# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

SMLMDriftCorrection.jl is a Julia package for fiducial-free drift correction in Single Molecule Localization Microscopy (SMLM). It works on both 2D and 3D localization data and is part of the JuliaSMLM ecosystem (depends on SMLMData.jl for core data structures).

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
2. **Inter-dataset correction**: Aligns datasets to each other using constant shifts

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
3. `findinter!()` aligns datasets (threaded all-vs-DS1, then sequential refinement vs earlier)
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
- `crosscorr.jl`: Cross-correlation helpers (`findshift`, `histimage2D`, `crosscorr2D`)
- `roi_selection.jl`: Auto-ROI subsampling (`calculate_n_locs_required`, `find_dense_roi`)
- `affine.jl`: Affine (similarity) transform functions: `apply_affine_2d/3d`, `correct_affine_2d/3d`, `apply_affine_transform!`
- `align.jl`: `align_smld()` for alignment of independent SMLDs (shift or affine transform)
- `typedefs.jl`: Abstract types, `InterShift`, `DriftInfo`, `AlignConfig`, `AlignInfo`, `AbstractAlignTransform`, `ShiftTransform`, `AffineTransform2D/3D`

### Threading

Intra-dataset correction is parallelized with `Threads.@threads` (each dataset independent). The first inter-dataset pass (all vs DS1) is also threaded using a precomputed snapshot of corrected coordinates. The refinement pass (each vs all earlier) is sequential.

### Adaptive Neighbor Optimization (Intra-dataset)

The `NeighborState` / `InterNeighborState` structs track KDTree neighbors and rebuild only when drift changes significantly (threshold: 100 nm). This avoids O(N log N) tree rebuilds on every optimizer iteration.

### Inter-dataset Alignment (Merged Cloud Entropy)

Inter-dataset alignment uses a "merged cloud" entropy approach:
1. Combine shifted dataset with reference dataset(s)
2. Compute entropy of the combined point cloud
3. Optimizer finds shift that minimizes entropy (tighter combined cloud = better alignment)

This properly incorporates localization uncertainties (σ) and works well for real SMLM data where datasets image the same underlying structure.

### Quality Tiers

- `:fft`: Fast cross-correlation only (~10x faster, less accurate)
- `:singlepass` (default): Single pass of intra then inter correction
- `:iterative`: Full convergence with intra↔inter iteration

### Dataset Modes

- `:registered` (default): Datasets are independent acquisitions with spatial overlap. Uses entropy-based inter-dataset alignment via `findinter!()`.
- `:continuous`: One long acquisition split into files. Uses polynomial endpoint chaining (warmstart) for inter-dataset alignment since chunks have temporal but not spatial overlap.

**Chunking guidance for continuous mode**: Consider chunking when acquisitions exceed ~4000 frames, using `chunk_frames=4000` as a reasonable maximum. Shorter acquisitions can use a single polynomial with moderate degree. The warmstart mechanism initializes each chunk's polynomial from the previous chunk's endpoint for smooth transitions.
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

**Residual drift from chunk stitching (coarse-to-fine 2-pass)**: With `n_chunks > 1`, chunks are fit independently and stitched by endpoint chaining (inter-shift pinned to the chained value, see Continuous Mode Internals). Per-chunk fit bias therefore *accumulates across seams*, leaving a small coherent **directional** residual drift (observed ~17nm on a 20k-frame / 10-chunk DNA-PAINT nanoruler) that smears fine structure. Chunking can't simply be removed — a single global polynomial (`n_chunks=1`) cannot bootstrap from the uncorrected, drift-smeared cloud (flat entropy gradient → optimizer stalls). The robust, config-only fix is a two-pass coarse-to-fine: chunk to bootstrap, then refine the de-smeared output with a single low-degree global polynomial (no seams):
```julia
# Pass 1: chunked bootstrap
(smld_coarse, info1) = driftcorrect(smld; dataset_mode=:continuous, n_chunks=10, degree=3, quality=:iterative)
# Pass 2: single-poly refine on pass-1 output (removes the stitch-accumulation residual)
(smld_corrected, info2) = driftcorrect(smld_coarse; dataset_mode=:continuous, n_chunks=1, degree=2, quality=:iterative)
```
A low `degree` (2-3) suffices for pass 2 (the residual is smooth/low-frequency); Nelder-Mead converges it fine. Do NOT use `warm_start=info1.model` for pass 2 — a chunked model's per-chunk polynomials are normalized to each chunk's frame domain and can't be reinterpreted on the full domain; use the two sequential calls. Measure residual with consecutive/endpoint time-block cross-correlation (`findshift`), not block-vs-global (which under-reads on smeared data); and run a random-split control before attributing any residual spread to field-dependence (small-group CC spread is noise-inflated).

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
- `shift_scale=1.0`: Expected inter-shift scale (μm) for registered mode L2 regularization (λ=1/σ²)
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

For continuous mode, inter-shifts are warmstarted from polynomial endpoint chaining (`_warmstart_inter_continuous!`) and regularized using boundary gap estimates (`_estimate_continuous_lambda`). Key functions: `endpoint_drift()`, `startpoint_drift()`, `evaluate_drift()` in `legendre.jl`.

## Key Dependencies

- **SMLMData.jl**: SMLD, Emitter types (compat: 0.7)
- **NearestNeighbors.jl**: KDTree for efficient spatial queries
- **Optim.jl**: BFGS for inter-dataset, Nelder-Mead for intra-dataset (10000 iterations)
- **LegendrePolynomials.jl**: Orthogonal polynomial basis (`Pl`)
- **SMLMSim.jl**: Test data generation (compat: 0.3-0.6)
