# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

SMLMDriftCorrection.jl is a Julia package for fiducial-free drift correction in SMLM. It works on both 2D and 3D localization data and is part of the JuliaSMLM ecosystem (depends on SMLMData.jl for core data structures).

## Development Commands

```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Build documentation
julia --project=docs docs/make.jl
```

### Project Environments
- Root `Project.toml`: Main package
- `dev/Project.toml`: Development experiments
- `examples/Project.toml`: Example scripts
- `docs/Project.toml`: Documentation

## Code Architecture

### Algorithm Overview

The drift correction has two phases:
1. **Intra-dataset correction**: Corrects drift within each dataset using Legendre polynomial models (optimized via entropy minimization)
2. **Inter-dataset correction**: Aligns datasets to each other using constant shifts. In registered mode this is **CC-primary** (cross-correlation seed → entropy refine → overlap arbiter, per dataset), because the global merged-cloud entropy alone is ~1/N blind to a single dataset's offset — see Inter-dataset Alignment. Continuous mode instead chains polynomial endpoints.

The algorithm uses **entropy minimization** as the cost function with **adaptive KDTree neighbor rebuilding** for efficiency.

### Type Hierarchy

```
AbstractDriftModel
├── AbstractIntraInter (combined intra+inter models)
│   └── LegendrePolynomial (main model used by driftcorrect)
│
AbstractIntraDrift (per-dataset models)
└── IntraLegendre (wraps LegendrePoly1D per dimension)
│
AbstractIntraDrift1D (1D polynomial components)
└── LegendrePoly1D (normalized to [-1, 1] time domain)
│
InterShift (per-dataset constant shift)

DriftConfig <: AbstractSMLMConfig (input config, @kwdef)
DriftInfo{M} <: AbstractSMLMInfo (output struct with model, timing, convergence, roi_indices)

AbstractAlignTransform
├── ShiftTransform (pure translation, default)
├── AffineTransform2D (rotation + uniform scale + translation, 4 params)
└── AffineTransform3D (Euler rotation + uniform scale + translation, 7 params)

AlignConfig <: AbstractSMLMConfig (input config for align_smld, @kwdef)
AlignInfo <: AbstractSMLMInfo (output struct with shifts, transforms, timing)
```

### Key Data Flow

1. `driftcorrect(smld)` or `driftcorrect(smld, config::DriftConfig)` creates a `LegendrePolynomial` model
2. `findintra!()` optimizes intra-dataset drift per dataset (parallelized with `Threads.@threads`)
3. Inter-dataset alignment: registered = `_cc_primary_inter!` (per-dataset CC seed → `findinter!` entropy refine → overlap arbiter, threaded); continuous = `_ccseed_inter_continuous!` (consecutive-chunk CC → `findinter!` refine). `findinter!` is the entropy-refine primitive both wrap.
4. `correctdrift(smld, model)` applies the final corrections
5. Returns tuple `(smld_corrected, info::DriftInfo)`
6. Optional continuation: `driftcorrect(smld, info::DriftInfo)` refines from previous result

### Source Files

- `interface.jl`: Main `driftcorrect()` function with quality tiers - start here
- `legendre.jl`: `LegendrePolynomial`, `IntraLegendre`, `LegendrePoly1D` types and evaluation
- `intrainter.jl`: `findintra!()`, `findinter!()`, `applydrift()`, `correctdrift()`
- `costfuns.jl`: `NeighborState`, adaptive entropy cost functions
- `cost_entropy.jl`: Entropy calculations (KL divergence, `entropy_HD`, `ub_entropy`)
- `utilities.jl`: `filter_emitters()`, `chunk_smld()`, `drift_trajectory()`
- `crosscorr.jl`: Cross-correlation helpers (`findshift` with optional `maxcount` bin cap, `histimage2D`, `crosscorr2D`; `_shared_cc_roi` builds the histogram ROI: camera field in x/y, floor/ceil of both SMLDs' z range padded by one bin in 3D)
- `roi_selection.jl`: Auto-ROI subsampling (`calculate_n_locs_required`, `find_dense_roi`)
- `affine.jl`: Affine (similarity) transform functions: `apply_affine_2d/3d`, `correct_affine_2d/3d`, `apply_affine_transform!`
- `align.jl`: `align_smld()` for alignment of independent SMLDs (shift or affine transform)
- `typedefs.jl`: Abstract types, `InterShift`, `DriftInfo`, `AlignConfig`, `AlignInfo`, `AbstractAlignTransform`, `ShiftTransform`, `AffineTransform2D/3D`

### Threading

Intra-dataset correction is parallelized with `Threads.@threads` (each dataset independent). Registered inter alignment (`_cc_primary_inter!`) is also threaded — each dataset's CC seed + entropy refine + overlap arbiter runs against a precomputed corrected snapshot. Continuous CC-seeding (`_ccseed_inter_continuous!`) is sequential (a consecutive-chunk chain), while its inter refine pass (`findinter!`) is threaded.

### Adaptive Neighbor Optimization (Intra-dataset)

The `NeighborState` / `InterNeighborState` structs track KDTree neighbors and rebuild only when drift changes significantly (threshold: 100 nm). This avoids O(N log N) tree rebuilds on every optimizer iteration.

### Inter-dataset Alignment (registered mode: CC-primary)

Registered-mode inter alignment (`_cc_primary_inter!`) treats **cross-correlation as the primary** per-dataset objective and **entropy as the refinement**. For each dataset (including the reference), per Jacobi pass over one corrected snapshot:

1. **CC seed** — cross-correlation of the dataset (intra-corrected) against the corrected consensus of all *other* datasets, on **bin-count-capped histograms** (`findshift(...; maxcount=_CC_MAXCOUNT)`, 4 localizations per 50 nm bin). CC is globally robust: it finds the right alignment basin even for a dataset the merged-cloud entropy can't see; the cap stops a bright aggregate from capturing the peak (hot spots, below). The seed is the CC shift unless the incumbent shift's consensus overlap is strictly higher.
2. **Entropy refine** — `findinter!` minimizes the entropy of the dataset merged with that consensus, starting from the seed (no L2 reg — the seed is the anchor; `cc_init=false`, so a zero seed is used as chosen rather than replaced by findinter!'s own CC). The merged-cloud entropy properly incorporates localization uncertainties (σ) and gives sub-CC-bin precision *when it starts in the right basin*.
3. **Overlap arbiter** (backstop, `_arbitrate`) — on per-localization inlier flags (within `_CC_VERIFY_RADIUS` = 30 nm of a consensus localization) the candidate of {refined, CC, incumbent} with the largest **occupancy** picks the basin (ties keep refined, then CC, then incumbent); among the candidates within `_CC_VERIFY_RADIUS` of that winner (the winner included; evaluated only when there are two or more) the one with the lowest **entropy cost** is kept (ties in the same order). The cost is the objective `findinter!` minimizes, built by the shared `_inter_entropy_cost` and evaluated at each candidate's own shift with neighbors built there (`_arbiter_cost`). A lead of a few localizations between candidates this close is noise: on sparse data (~1300 localizations per dataset) 0.2.9 kept a CC that led the refine by 3 of 1260 and left 2-3 datasets ~15 nm off on two genmab cells, where the refine was within 1 nm. Occupancy (`_occupancy`, `_OCC_SEP` = 30 nm) is the number of inlier localizations kept when each is dropped if an already-kept inlier lies within `_OCC_SEP`, i.e. about one count per resolved site however often it was localized. A raw inlier count lets a wrong shift that lays sites localized many times onto the consensus outscore the true shift (the cross-basin fixture, test "cross-basin overlap decision", left a dataset 4.2 μm off in 0.2.9 and with a McNemar margin); occupancy cannot be bought by repeated localizations of the same site. The seed and the cold start / `:fft` guard (`_accept_if_better_overlap!`) score the same way and keep the 0.2.9 tie rule: the CC is used unless the incumbent's occupancy is strictly higher (ties go to the CC, typically both zero when a correct but bin-quantized CC lands just past the radius on sparse precise data; Codex review of 537b421: keeping the incumbent there left 213 nm). `_overlap_leads` / `_OVERLAP_Z` (McNemar) remain only for the stage-move warning. The arbiter is required because the entropy is ~1/N blind (below) and so cannot, on its own, reveal a refine that wandered to a wrong local minimum.

**Over-5-μm CCs.** In the registered CC-primary path (`:singlepass`, `:iterative`; not `:fft`, which has no 5 μm cut) a CC with a component over 5 μm is never applied, however well it overlaps (a shift that large is a collection failure, the stage moving between acquisitions, not drift; the dataset stays at its kept shift). When one is dropped and, scored against the final consensus (the cached CC, plus a fresh capped CC against that consensus, since the cached one was measured earlier), leads the shift the dataset ends with (`_overlap_leads`), `_cc_primary_inter!` emits one `@warn` per dataset per `driftcorrect` call (a `warned` vector threaded through the iterative tier) naming the CC shift and both overlap fractions and ending "the dataset was left at the kept shift (likely a collection failure: the stage moved between acquisitions; check the data)"; the applied shift is unchanged.

The reference (DS1) is included so a misaligned reference is fixable, then the result is re-anchored to `inter[1]=0` (an image-invariant global shift). `:singlepass` does a CC cold-start vs DS1 (kept unless the incumbent, i.e. zero, overlaps DS1 strictly better) then 2 passes; `:iterative` runs 1 pass per outer iteration. `:fft` is a separate, faster path (CC only, no entropy refine): every CC there is capped too, and each CC shift is overlap-guarded by `_accept_if_better_overlap!`.

**Hot spots capture a raw-count CC (fixed in 0.2.9).** `findshift` takes the argmax of an *unnormalized* histogram cross-correlation over the whole field (registered callers then drop a candidate with any component over 5 μm; the `:fft` tier has no such cut). A bright aggregate localized thousands of times in one 50 nm bin, present in some datasets only (it bleaches), dominates a raw-count CC: the peak lands wherever the other dataset's densest cluster overlays the hot spot, microns from the true shift. In 0.2.7–0.2.8 the cold start accepted any such peak under 5 μm and the arbiter only compared the CC seed with its own refine, so 8 of 102 genmab cohort cells (registered, `:iterative`, `shift_scale=0.05`) came back with datasets 1.5–6 μm off (reported as `converged` in 2 iterations, entropy unchanged to 4e-6); milder captures left single datasets tens of nm off on cells that looked fine (HeLa WT Cell_01 DS2 ~60 nm, HeLa E345R Cell_02 DS1-4). The `:fft` tier had the same exposure (the 2026-09 project-sr-spt report of datasets thrown 2-15 μm). 0.2.9 fixes the cause by capping the CC histograms (`_CC_MAXCOUNT = 4`), with the incumbent kept as an arbiter candidate as a backstop. Chosen over four rounds of richer arbiter designs (sub-bin grid search, count-then-distance ranking) because each round's review found a new failure and the cap measured best: a 40-scenario randomized SMLMSim sweep with known truth had 0 failures (worst 21.5 nm registered, 38.5 nm `:fft`; base 0.2.8 15/40, worst 5.6 μm), all 8 cells back at 0.97–1.08× April, and the best independent per-dataset co-localization on 18 real cells (`dev/output/decision-0.2.9-options.md`, local). Regression tests: CI "Hot spot in early datasets: CC-primary inter + :fft (registered)", "CC-primary inter: incumbent just past the radius vs captured CC" (includes raw vs capped `findshift`), "Overlap guard accepts a quantized CC (sub-bin shift, sparse)", "Overlap guard: zero-overlap tie goes to the CC", "findshift 3D: thin z stack"; extended "Registered driftcorrect: hot spot in early datasets" (< 30 nm) and "Unshared hot spot in the scored dataset".

**Why CC-primary — entropy blindness.** The merged-cloud entropy is a *global* cost: one dataset being offset is only ~1/N of the cloud, so a ~50 nm error in a single dataset moves the entropy by <1 part in 10⁶. Entropy-*only* inter alignment therefore has almost no gradient to align an individual dataset and can leave one badly misaligned (this produced the visible single-dataset ghost in some cohort cells); for the same reason the old `:iterative` loop could wander the inter parameters without lowering the cost (see Quality Tiers). Making CC the primary objective supplies the global signal entropy lacks, while the overlap arbiter keeps the entropy refine from regressing it. Validated on real cohort cells (reference-outlier DS1 58→3 nm, multi-outlier max 72→7 nm) and by an *independent* cross-dataset NN co-localization metric that improved on every dataset of the original bug cell. Note: on diffuse/low-contrast cells CC is imprecise (no structure to lock onto), but the overlap arbiter caps the damage by never accepting a shift that lowers consensus overlap. **Continuous mode uses a different mechanism** — consecutive-chunk cross-correlation seeding (see Dataset Modes / Continuous Mode Internals), not the merged-cloud consensus (whose cloud is smeared across chunks by the drift itself). Validated on real cohort data (DNA-PAINT 593→10 nm, ruler 23→5 nm).

### Quality Tiers

- `:fft`: Fast cross-correlation only (~10x faster, less accurate)
- `:singlepass` (default): Single pass of intra then inter correction
- `:iterative`: Full intra↔inter refinement, one inter pass per iteration (registered: a CC-primary pass; continuous: in-place inter refine, no per-iteration re-chain). Convergence trips when the max change across **both** inter and intra parameters falls below `convergence_tol`. In **both** modes an entropy **cost-plateau** early exit also applies (relative improvement < `_ENTROPY_REL_TOL`=1e-4): a movement-only test can run to the iteration cap at ~0% gain whenever the cost sits in a near-flat basin — registered via merged-cloud entropy blindness (above), continuous because the CC-seeded singlepass already solves the alignment so the iterate loop only wobbles params below the entropy scale. (This previously mis-reported `converged=false` on flat-basin continuous runs that had in fact reached the entropy floor — fixed in 0.2.8 by extending the plateau exit to continuous mode.)

### Dataset Modes

- `:registered` (default): Datasets are independent acquisitions with spatial overlap. Uses entropy-based inter-dataset alignment via `findinter!()`.
- `:continuous`: One long acquisition split into chunks. Inter-chunk alignment **measures** each boundary shift by cross-correlating consecutive chunks (`_ccseed_inter_continuous!`), with polynomial endpoint-chaining kept only as a per-boundary fallback where CC can't lock on. Consecutive chunks share structure (same FOV, adjacent in time), so CC is reliable — and unlike endpoint extrapolation it neither accumulates directional residual nor propagates a single bad-boundary blow-up down the chain (see Continuous Mode Internals).

**Chunking guidance for continuous mode**: Consider chunking when acquisitions exceed ~4000 frames, using `chunk_frames=4000` as a reasonable maximum. Shorter acquisitions can use a single polynomial with moderate degree. Each chunk's inter shift is measured against the previous (corrected) chunk by cross-correlation; a soft intra boundary prior keeps the per-chunk polynomials continuous across boundaries.
```julia
# Short acquisition (<4000 frames) - single polynomial
(smld_corrected, info) = driftcorrect(smld; dataset_mode=:continuous, degree=3)

# Long acquisition - chunk into ~4000 frame segments
(smld_corrected, info) = driftcorrect(smld; dataset_mode=:continuous, chunk_frames=4000)

# Multi-file data - datasets already separate, no explicit chunking needed
(smld_corrected, info) = driftcorrect(smld; dataset_mode=:continuous)
```

For trajectory plotting in continuous mode:
```julia
traj = drift_trajectory(info.model; cumulative=true)
```

## Usage Patterns

### Basic Usage
```julia
using SMLMDriftCorrection

# Returns tuple (smld_corrected, info::DriftInfo)
(smld_corrected, info) = driftcorrect(smld)

# Access model for trajectory extraction
traj = drift_trajectory(info.model)
# traj.frames, traj.x, traj.y ready for plotting
```

### Testing with Simulated Drift
```julia
DC = SMLMDriftCorrection

# Create random drift model
drift_model = DC.LegendrePolynomial(smld; degree=2, initialize="random", rscale=0.1)

# Apply drift (for testing)
smld_drifted = DC.applydrift(smld, drift_model)

# Correct drift (inverse operation - exact recovery)
smld_recovered = DC.correctdrift(smld_drifted, drift_model)
```

### Warm Start
```julia
# First dataset
(smld1_corrected, info1) = driftcorrect(smld1; degree=2)

# Second dataset - use model from first as starting point
(smld2_corrected, info2) = driftcorrect(smld2; warm_start=info1.model)
```

### Aligning Independent SMLDs
```julia
using SMLMDriftCorrection

# Align separate acquisitions to a common reference (smlds[1])
(aligned, info) = align_smld(smlds; method=:entropy)
info.shifts  # shift applied to each SMLD

# FFT-only (faster, less accurate)
(aligned, info) = align_smld(smlds; method=:fft)

# Affine alignment (rotation + scale + translation)
(aligned, info) = align_smld(smlds; transform=:affine)
info.transforms[2]  # AffineTransform2D(θ, scale, tx, ty)
info.shifts[2]      # translation component [tx, ty]

# Config struct form
config = AlignConfig(method=:entropy, transform=:affine, maxn=100)
(aligned, info) = align_smld(smlds, config)
```

### Filtering to ROI
```julia
x = [e.x for e in smld.emitters]
y = [e.y for e in smld.emitters]
mask = (x .> 64.0) .& (x .< 128.0) .& (y .> 64.0) .& (y .< 128.0)
smld_roi = filter_emitters(smld, mask)
```

## Key Parameters

- `quality=:singlepass`: Quality tier (`:fft`, `:singlepass`, `:iterative`)
- `degree=2`: Polynomial degree for intra-dataset drift
- `maxn=200`: Maximum neighbors for entropy calculation
- `dataset_mode=:registered`: How to handle multi-dataset alignment
- `chunk_frames` / `n_chunks`: Chunking for continuous mode
- `max_iterations=10`: Maximum iterations for `:iterative` mode
- `convergence_tol=0.001`: Convergence tolerance (μm) for `:iterative` mode
- `warm_start=nothing`: Previous model for warm starting optimization
- `shift_scale=1.0`: No effect — read by no dataset mode. It was the registered-mode L2 inter-shift prior (λ=1/σ²) through 0.2.5; CC-primary registered inter (released in 0.2.7) uses no L2 prior. Kept in `DriftConfig` so existing configs construct; a non-default value logs one warning (`maxlog=1`). Whether registered mode should use it again is an open design question (after 0.2.9).
- `verbose=0`: 0=quiet, 1=info, 2=debug

### Auto-ROI Parameters

- `auto_roi=false`: Set to `true` for faster processing using a dense spatial subset (~15% of data). Trades some accuracy (~1.4nm vs ~0.5nm RMSD) for speed.
- `σ_loc=0.010`: Typical localization precision (μm) for ROI sizing
- `σ_target=0.001`: Target drift precision (μm) for ROI sizing
- `roi_safety_factor=4.0`: Safety multiplier for required localizations

When `auto_roi=true`, selects a contiguous rectangular region from the densest part of the FOV. This preserves blink pairs from the same emitters which is essential for entropy-based optimization.

### Alignment Parameters (align_smld)

- `method=:entropy`: `:entropy` (CC + entropy refinement) or `:fft` (CC only)
- `transform=:shift`: `:shift` (translation only) or `:affine` (rotation + scale + translation)
- `maxn=100`: Maximum neighbors for entropy calculation
- `histbinsize=0.05`: Histogram bin size (μm) for cross-correlation

Note: `transform=:affine` requires `method=:entropy` (FFT can only recover translation). The affine transform stores forward parameters (how target differs from reference); `AffineTransform2D` has fields `θ` (radians), `scale`, `tx`, `ty`. The correction applies the inverse automatically.

## Units

All distance units are in **micrometers (μm)**.

### SMLMData Compatibility

The package supports both SMLMData 0.5 and 0.6+ via `_HAS_SIGMA_XY` (compile-time `const`). The `_make_emitter_2d` / `_make_emitter_3d` helpers in `utilities.jl` dispatch to positional (0.5) or keyword (0.6+) constructors. Current compat: SMLMData 0.7.

### Continuous Mode Internals

For continuous mode (singlepass), inter-shifts are **CC-seeded per boundary** (`_ccseed_inter_continuous!`): cross-correlate consecutive intra-corrected chunks, and keep whichever of {CC measurement, endpoint-chain prediction} better overlaps the corrected previous chunk (`_overlap_score` within `_CC_VERIFY_RADIUS`); then entropy-refine regularized by boundary-gap estimates (`_estimate_continuous_lambda`). Endpoint-chaining (`endpoint_drift()` − `startpoint_drift()`) survives only as the per-boundary fallback where CC can't lock on. Because CC *measures* each boundary, it avoids endpoint extrapolation's two failure modes — slow directional accumulation, and a single bad boundary (a degree-d polynomial blowing up at t=±1) propagated down the cumulative chain. The `:iterative` continuous path is separate (soft intra boundary priors + no per-iteration re-chain, `b463bd1`) and is left on its own machinery. Key functions: `endpoint_drift()`, `startpoint_drift()`, `evaluate_drift()` in `legendre.jl`.

## Key Dependencies

- **SMLMData.jl**: SMLD, Emitter types (compat: 0.7)
- **NearestNeighbors.jl**: KDTree for efficient spatial queries
- **Optim.jl**: BFGS for inter-dataset, Nelder-Mead for intra-dataset (10000 iterations)
- **LegendrePolynomials.jl**: Orthogonal polynomial basis (`Pl`)
- **SMLMSim.jl**: Test data generation (compat: 0.3-0.6)
