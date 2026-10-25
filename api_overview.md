# SMLMDriftCorrection.jl API Overview

AI-parseable API reference for SMLMDriftCorrection.jl. All distance units are in micrometers (μm).

## Exports Summary

- **Types:** 8 (`DriftConfig`, `DriftInfo`, `AlignConfig`, `AlignInfo`, `AbstractAlignTransform`, `ShiftTransform`, `AffineTransform2D`, `AffineTransform3D`)
- **Functions:** 5 (`driftcorrect`, `align_smld`, `filter_emitters`, `drift_trajectory`, `position_frame_correlation`)

## Key Concepts

Drift correction uses entropy minimization to find polynomial drift models that minimize spatial entropy of localization data. The algorithm has two phases: intra-dataset (polynomial drift within each acquisition) and inter-dataset (constant shift between acquisitions). The primary interface follows the `f(data, config) -> (result, info)` tuple pattern used across JuliaSMLM.

`dataset_mode` selects the inter-dataset algorithm. Registered mode is **CC-primary**: a cross-correlation seed on bin-count-capped histograms is entropy-refined, and an overlap arbiter picks among {refined, CC, incumbent}. With two datasets only the second is aligned, against the first. With three or more, every dataset including the reference is aligned against the others; moves within 30 nm of the incumbent are damped by (N-1)/N (N = number of datasets) to remove the overcount from updating all datasets at once. Larger moves into another basin are applied in full. Continuous mode instead seeds each inter-chunk shift by cross-correlating consecutive chunks; endpoint-chaining remains a per-boundary fallback. `quality=:fft` ignores `dataset_mode` and runs registered-style CC passes.

No tier, `:fft` included, applies a cross-correlation shift with a component over 5 μm. Registered `:singlepass` and `:iterative`, and `:fft` regardless of `dataset_mode`, warn once per dataset per call when a dropped CC overlaps clearly better than the shift the dataset ends with, indicating a likely collection failure (the stage moved between acquisitions). Continuous `:singlepass` and `:iterative` drop such a boundary CC and fall back to endpoint chaining at that boundary without a warning.

The `:iterative` tier converges when both inter and intra parameter changes fall below `convergence_tol`, or when the entropy cost plateaus (relative improvement < 1e-4, both modes). Continuation (`driftcorrect(smld, info)`) reuses the original run's `dataset_mode` and chunking. See [Theory and Workflow](docs/src/theory_workflow.md) for the alignment rules.

## Types

### DriftConfig

```julia
@kwdef struct DriftConfig <: AbstractSMLMConfig
```

Configuration for drift correction. All parameters have sensible defaults.

**Fields:**
- `quality::Symbol = :singlepass`: Quality tier (`:fft`, `:singlepass`, `:iterative`)
- `degree::Int = 2`: Legendre polynomial degree for intra-dataset drift
- `dataset_mode::Symbol = :registered`: Multi-dataset handling (`:registered` or `:continuous`)
- `chunk_frames::Int = 0`: Split datasets into chunks of N frames (continuous mode)
- `n_chunks::Int = 0`: Alternative: number of chunks per dataset
- `maxn::Int = 100`: Maximum neighbors for entropy calculation
- `max_iterations::Int = 10`: Iterations per attempt for `:iterative` mode; a run that has not converged is retried from its current model up to 3 times
- `convergence_tol::Float64 = 0.001`: Convergence tolerance (μm) for `:iterative` mode
- `warm_start::Union{Nothing, AbstractIntraInter} = nothing`: Previous `info.model` for warm starting
- `verbose::Int = 0`: Verbosity (0=quiet, 1=info, 2=debug)
- `auto_roi::Bool = false`: Use dense ROI subset for faster estimation
- `σ_loc::Float64 = 0.010`: Typical localization precision (μm) for ROI sizing
- `σ_target::Float64 = 0.001`: Target drift precision (μm) for ROI sizing
- `roi_safety_factor::Float64 = 4.0`: Safety multiplier for required localizations

**Constructor:**
```julia
DriftConfig()                                        # all defaults
DriftConfig(quality=:iterative, degree=3, verbose=1) # keyword overrides
```

### DriftInfo

```julia
struct DriftInfo{M<:AbstractIntraInter} <: AbstractSMLMInfo
```

Output metadata from drift correction. Returned as second element of tuple.

**Fields:**
- `model::M`: Fitted `LegendrePolynomial` drift model
- `elapsed_s::Float64`: Wall time in seconds
- `backend::Symbol`: Computation backend (`:cpu`)
- `iterations::Int`: Iterations completed (0 for `:fft`, 1 for `:singlepass`)
- `converged::Bool`: Whether convergence was achieved
- `entropy::Float64`: Final entropy value (a sum over localizations; divide by `length(smld.emitters)` for a size-independent number)
- `history::Vector{Float64}`: Entropy after the singlepass start and after each iteration (`:iterative` only; empty for `:fft` and `:singlepass`)
- `roi_indices::Union{Nothing, Vector{Int}}`: Indices used for ROI subset (`nothing` if `auto_roi=false`)
- `residual_correlation::NamedTuple`: Position-frame correlation diagnostic (intra_summary, intra_per_dataset, inter)

### AlignConfig

```julia
@kwdef struct AlignConfig <: AbstractSMLMConfig
```

Configuration for alignment of independent SMLDs via `align_smld`.

**Fields:**
- `method::Symbol = :entropy`: Alignment method (`:entropy` or `:fft`)
- `transform::Symbol = :shift`: Transform model (`:shift` translation only, `:affine` 2D affine; 2D only, ignores `method`)
- `maxn::Int = 100`: Maximum neighbors for entropy calculation
- `histbinsize::Float64 = 0.05`: Histogram bin size (μm) for cross-correlation
- `verbose::Int = 0`: Verbosity (0=quiet, 1=info)

**Constructor:**
```julia
AlignConfig()                                        # all defaults
AlignConfig(transform=:affine)                       # affine alignment
AlignConfig(method=:fft, transform=:shift, maxn=50)  # keyword overrides
```

### AlignInfo

```julia
struct AlignInfo <: AbstractSMLMInfo
```

Output metadata from `align_smld`.

**Fields:**
- `shifts::Vector{Vector{Float64}}`: Translation component for each SMLD (shifts[1] = zeros)
- `transforms::Vector{<:AbstractAlignTransform}`: Shift transforms, or best-fit `AffineTransform2D` similarity approximations of the general affine corrections
- `elapsed_s::Float64`: Wall time in seconds
- `method::Symbol`: Method used (`:entropy` or `:fft`)
- `transform::Symbol`: Transform model used (`:shift` or `:affine`)
- `backend::Symbol`: Computation backend (`:cpu`)
- `diagnostic::Union{Nothing, NamedTuple}`: `nothing` for `:shift`; for `:affine` a NamedTuple with `affine_coeffs` (exact per-dataset `(a, b, c, d, e, f)`) and `global_shifts`

### AbstractAlignTransform Hierarchy

```julia
AbstractAlignTransform
├── ShiftTransform          # Pure translation: shift::Vector{Float64}
├── AffineTransform2D       # θ::Float64, scale::Float64, tx::Float64, ty::Float64
└── AffineTransform3D       # exported; not produced by `align_smld` (`:affine` is 2D only)
```

`AffineTransform2D` stores the geometric mean scale and rotation from the 6-parameter affine fit. The exact correction is available in `info.diagnostic.affine_coeffs` and `info.diagnostic.global_shifts`. `AffineTransform3D` stores Euler angles `θx`, `θy`, `θz`, a uniform `scale`, and translations `tx`, `ty`, `tz`.

## Functions

### driftcorrect

```julia
driftcorrect(smld::SMLD, config::DriftConfig) -> (SMLD, DriftInfo)
driftcorrect(smld::SMLD; kwargs...) -> (SMLD, DriftInfo)
driftcorrect(smld::SMLD, info::DriftInfo; kwargs...) -> (SMLD, DriftInfo)
```

Main interface for drift correction. The config struct form is preferred.

**Arguments:**
- `smld`: SMLD structure with localization coordinates (2D or 3D)
- `config`: `DriftConfig` with all algorithm parameters
- `info`: Previous `DriftInfo` for continuation (third dispatch); reuses the original run's `dataset_mode` and chunking

**Example:**
```julia
config = DriftConfig(quality=:iterative, degree=3, verbose=1)
(smld_corrected, info) = driftcorrect(smld, config)
```

### align_smld

```julia
align_smld(smlds::Vector{<:SMLD}; kwargs...) -> (Vector{<:SMLD}, AlignInfo)
align_smld(smlds::Vector{<:SMLD}, config::AlignConfig) -> (Vector{<:SMLD}, AlignInfo)
```

Align a vector of independent SMLD structures to a common reference (smlds[1]).

**Transform modes:**
- `transform=:shift` (default): Translation only. Methods: `:entropy` (CC + entropy refinement) or `:fft` (CC only).
- `transform=:affine`: Full 2D affine (anisotropic scale + rotation + shear + translation). Uses a 5×5 grid of local CC shifts and a quality-weighted least-squares affine fit, with two passes for refinement. See the `align_smld` docstring for requirements.

**Arguments:**
- `smlds`: Vector of 2+ SMLD structures (same dimensionality, non-empty)
- `config`: `AlignConfig` with alignment parameters

**Examples:**
```julia
# Shift only (fast, translation)
(aligned, info) = align_smld([smld_ch1, smld_ch2]; transform=:shift, method=:fft)

# Affine (anisotropic scale + rotation + translation)
(aligned, info) = align_smld([smld_ch1, smld_ch2]; transform=:affine)
info.shifts[2]      # translation component [tx, ty]
info.transforms[2]  # AffineTransform2D(θ, scale, tx, ty)
```

### filter_emitters

```julia
filter_emitters(smld::SMLD, keep::Union{AbstractVector,AbstractRange}) -> SMLD
filter_emitters(smld::SMLD, keep::Integer) -> SMLD
```

Select emitters by boolean mask, indices, or single index.

**Example:**
```julia
x = [e.x for e in smld.emitters]
mask = (x .> 10.0) .& (x .< 20.0)
smld_roi = filter_emitters(smld, mask)
```

### drift_trajectory

```julia
drift_trajectory(model::AbstractIntraInter; dataset=nothing, frames=nothing, cumulative=false) -> NamedTuple
```

Extract drift trajectory from fitted model for plotting.

**Keywords:**
- `dataset`: specific dataset to extract (default: all)
- `frames`: frame range to evaluate (default: 1:n_frames)
- `cumulative`: if true, chain datasets end-to-end showing total accumulated drift (useful for continuous mode)

**Returns:** `(frames, x, y, dataset)` or `(frames, x, y, z, dataset)` for 3D

**Example:**
```julia
(smld_corrected, info) = driftcorrect(smld)
traj = drift_trajectory(info.model)
traj = drift_trajectory(info.model; cumulative=true)  # for continuous mode
```

### position_frame_correlation

```julia
position_frame_correlation(smld::SMLD; K::Int=20, mode::Symbol=:intra) -> NamedTuple
```

Residual-drift diagnostic: correlates local position residuals (position minus the mean of the `K` nearest spatial neighbors) with frame number (`mode=:intra`, per dataset, with a `summary`) or dataset index (`mode=:inter`, across datasets). `driftcorrect` stores the result on the corrected data in `info.residual_correlation`.

## Non-Exported Public API

### LegendrePolynomial (Drift Model)
```julia
LegendrePolynomial(smld::SMLD; degree=2, initialize="zeros", rscale=0.1)
LegendrePolynomial(ndims::Int, ndatasets::Int, nframes::Int; degree=2, initialize="zeros", rscale=0.01)
```
Combined intra + inter drift model using Legendre polynomial basis.

**Fields:**
- `ndatasets::Int`: Number of datasets
- `n_frames::Int`: Frames per dataset
- `intra::Vector{IntraLegendre}`: Per-dataset polynomial models
- `inter::Vector{InterShift}`: Per-dataset constant shifts

**Initialize options:** `"zeros"` (default), `"random"` — `"random"` seeds **both** the intra polynomial coefficients and the inter shifts with `rscale * randn()`. That is only the *constructed* initial model; in the high-level `driftcorrect` pipeline the inter shifts are then determined by the alignment step (registered: CC-primary; continuous: CC-seeded per boundary by `_ccseed_inter_continuous!`), which overwrites any random inter values.

### applydrift / correctdrift
```julia
applydrift(smld::SMLD, model::AbstractIntraInter) -> SMLD
correctdrift(smld::SMLD, model::AbstractIntraInter) -> SMLD
```
Apply or correct drift using a model. `correctdrift` is the inverse of `applydrift`.

### findshift
```julia
findshift(smld1::SMLD, smld2::SMLD; histbinsize=1.0, maxcount=nothing) -> Vector{Float64}
```
Find shift between two SMLDs via cross-correlation of histogram images. Returns `[dx, dy]` or `[dx, dy, dz]`.
`maxcount` (a positive integer) clips each histogram bin to at most that many localizations before correlating,
so a bright aggregate cannot outweigh the structure; `nothing` correlates raw counts. `driftcorrect`'s registered CC-primary and
`:fft` alignment use `maxcount=4`; `align_smld` correlates raw counts.

### find_affine_shift_field
```julia
find_affine_shift_field(smld1, smld2; histbinsize=0.05, sub_histbinsize=0.005,
                        grid_n=5, min_locs=30) -> NamedTuple
```
Recover full 2D affine transform using local CC shifts on a grid. Returns `(a, b, c, d, e, f, global_shift, n_tiles, rms_residual, ...)` where the correction is `x_new = x - (a*x + b*y + c)`, `y_new = y - (d*x + e*y + f)`. Used internally by `align_smld(; transform=:affine)`.

### find_affine_fft
```julia
find_affine_fft(smld1, smld2; histbinsize=0.05, max_rotation=10.0) -> AffineTransform2D
```
Recover 2D similarity transform via Fourier-Mellin (log-polar CC of magnitude spectra). Fast (~2s) but assumes isotropic scale. Less accurate than shift-field method for small transforms.

### drift_at_frame
```julia
drift_at_frame(model::AbstractIntraInter, dataset::Int, frame::Int) -> Vector
```
Evaluate total drift (inter + intra) at a specific dataset and frame. Returns `[dx, dy]` or `[dx, dy, dz]`. Low-level function; for plotting trajectories use `drift_trajectory()`.

## Drift Evaluation Functions

### evaluate_at_frame
```julia
evaluate_at_frame(p::LegendrePoly1D, frame::Int) -> Float64
```
Evaluate 1D Legendre polynomial drift at a specific frame number. Returns the drift value (not the corrected coordinate).

### evaluate_drift
```julia
evaluate_drift(intra::IntraLegendre, frame::Int) -> Vector
```
Evaluate intra-dataset Legendre drift at a specific frame across all dimensions. Returns `[dx, dy]` or `[dx, dy, dz]`.

### endpoint_drift
```julia
endpoint_drift(intra::IntraLegendre, n_frames::Int) -> Vector
```
Evaluate Legendre drift at the last frame of a dataset. Evaluates at t=+1 in Legendre domain.

### startpoint_drift
```julia
startpoint_drift(intra::IntraLegendre) -> Vector
```
Evaluate Legendre drift at frame 1 of a dataset (uses `n_frames` from the polynomial). Evaluates at t=-1, which is NOT zero.

## Internal Types

### AbstractDriftModel Hierarchy
```
AbstractDriftModel
├── AbstractIntraInter
│   └── LegendrePolynomial
│
AbstractIntraDrift
└── IntraLegendre
│
AbstractIntraDrift1D
└── LegendrePoly1D
│
InterShift (per-dataset constant shift)

AbstractAlignTransform (exported)
├── ShiftTransform (exported)
├── AffineTransform2D (exported)
└── AffineTransform3D (exported)

AbstractSMLMConfig
├── DriftConfig (exported)
└── AlignConfig (exported)

AbstractSMLMInfo
├── DriftInfo{M} (exported)
└── AlignInfo (exported)
```

### IntraLegendre
```julia
IntraLegendre(ndims::Int, n_frames::Int; degree=2)
```
Per-dataset Legendre drift model wrapping one `LegendrePoly1D` per spatial dimension.

**Fields:**
- `ndims::Int`: Number of spatial dimensions (2 or 3)
- `dm::Vector{LegendrePoly1D}`: One polynomial per dimension

### LegendrePoly1D
```julia
LegendrePoly1D(degree::Int, n_frames::Int)
```
1D Legendre polynomial drift for a single spatial dimension. Coefficients are for P_1(t) through P_degree(t) where t ∈ [-1, 1] (normalized frame number). No P_0 constant term (handled by InterShift).

**Fields:**
- `degree::Int`: Polynomial degree
- `coefficients::Vector{<:Real}`: Coefficients for P_1, P_2, ..., P_degree
- `n_frames::Int`: Frame count for time normalization

### InterShift
```julia
InterShift(ndims::Int)
```
Per-dataset constant shift for inter-dataset alignment.

**Fields:**
- `ndims::Int`: Number of spatial dimensions
- `dm::Vector{<:Real}`: Shift values [dx, dy] or [dx, dy, dz]

## Entropy Functions

### entropy_HD
```julia
entropy_HD(σ_x::Vector, σ_y::Vector) -> Real
entropy_HD(σ_x::Vector, σ_y::Vector, σ_z::Vector) -> Real
```
Base entropy term H_i(D) from localization uncertainties only.

### ub_entropy
```julia
ub_entropy(x, y, σ_x, σ_y; maxn=200, divmethod="KL") -> Real
ub_entropy(x, y, z, σ_x, σ_y, σ_z; maxn=200, divmethod="KL") -> Real
ub_entropy(r::Matrix, σ::Matrix; maxn=200, divmethod="KL") -> Real
```
Upper bound on entropy using k-nearest neighbors. The matrix interface accepts N×K matrices (points as rows, dimensions as columns) for K=2 (2D) or K=3 (3D).

**divmethod options:** `"KL"` (default), `"Symmetric"`, `"Bhattacharyya"`, `"Mahalanobis"`

## Cross-Correlation Functions

### histimage2D / histimage3D
```julia
histimage2D(x, y; ROI=[-1.0], histbinsize=1.0) -> Matrix{Int}
histimage3D(x, y, z; ROI=[-1.0], histbinsize=1.0) -> Array{Int,3}
```
Create histogram image from localization coordinates.

### crosscorr2D / crosscorr3D
```julia
crosscorr2D(im1::Matrix, im2::Matrix) -> Matrix
crosscorr3D(im1::Array, im2::Array) -> Array
```
Compute cross-correlation between two images via FFT (FourierTools.ccorr). Zero-padded to 2x size to eliminate cyclic wrap-around artifacts.

### crosscorr2Dweighted
```julia
crosscorr2Dweighted(im1::AbstractMatrix, im2::AbstractMatrix) -> Matrix
```
Compute intensity-weighted cross-correlation between two 2D images.

### findshift_damped
```julia
findshift_damped(smld1::SMLD, smld2::SMLD; histbinsize=1.0, prior_shift=[0.0, 0.0], prior_sigma=1.0,
                 maxcount=nothing) -> Vector{Float64}
```
Find shift with Gaussian damping centered at `prior_shift` (`maxcount` as in `findshift`). Used to refine outlier shifts by searching near an expected location. Applies a Gaussian weight to the cross-correlation centered at `prior_shift` with width `prior_sigma`.

## Internal Optimization

### findintra!
```julia
findintra!(intra::AbstractIntraDrift, smld::SMLD, dataset::Int, maxn::Int; skip_init::Bool=false)
```
Optimize intra-dataset drift using entropy minimization with adaptive KDTree rebuilding.

**Keywords:**
- `skip_init=false`: If true, skip random initialization (use when warmstarted externally)

### findinter!
```julia
findinter!(dm::AbstractIntraInter, smld::SMLD, dataset_n::Int, ref_datasets::Vector{Int}, maxn::Int;
           precomputed_corrected=nothing, regularization_target=nothing, regularization_lambda=0.0,
           cc_init=true)
```
Optimize inter-dataset shift for `dataset_n` against `ref_datasets` using merged-cloud entropy. Optional regularization: `entropy + λ*||shift - target||²`.
The optimizer starts from the current shift; when that shift is exactly zero and `cc_init=true` (default) it starts from a
cross-correlation instead. Pass `cc_init=false` to start from a chosen zero seed (the CC-primary path does).

### NeighborState / InterNeighborState
```julia
NeighborState(N::Int, k::Int, rebuild_threshold::Real)
InterNeighborState(N_n::Int, k::Int, rebuild_threshold::Real)
InterNeighborState3D(N_n::Int, k::Int, rebuild_threshold::Real)
```
Track KDTree neighbors for adaptive rebuilding during optimization. Rebuilds only when drift changes by more than `rebuild_threshold` (default: 0.1 μm). `InterNeighborState3D` initializes `last_shift` with 3 elements for 3D datasets (vs 2 for the default `InterNeighborState`).

### filter_by_dataset
```julia
filter_by_dataset(smld::SMLD, dataset::Int) -> SMLD
filter_by_dataset(smld::SMLD, datasets::Vector{Int}) -> SMLD
```
Filter SMLD to include only emitters from the specified dataset(s).

## Parameter Conversion (Internal)

```julia
intra2theta(p::IntraLegendre) -> Vector{Real}
theta2intra!(p::IntraLegendre, θ::Vector{Real})
inter2theta(s::InterShift) -> Vector{Real}
theta2inter!(s::InterShift, θ::Vector{Real})
```
Convert between drift models and flat parameter vectors for optimization.

## Chunking Utilities

```julia
chunk_smld(smld; chunk_frames=0, n_chunks=0) -> NamedTuple
compute_chunk_params(n_frames; chunk_frames=0, n_chunks=0) -> NamedTuple
```
Split SMLD into chunks for finer-grained drift correction in continuous mode.

## Threading

- **Intra-dataset correction** is parallelized with `Threads.@threads` (each dataset independent).
- **Registered inter-dataset alignment** is also threaded (each dataset's CC seed + entropy refine runs against a precomputed snapshot of corrected coordinates).
- **Continuous CC seeding** walks the chunks sequentially (each placement builds on the previous chunk's); its entropy refine is threaded against chunk 1, then sequential against earlier chunks.
- Launch Julia with `julia -t N` or set `JULIA_NUM_THREADS=N` to use N threads.

## Common Workflows

### Basic Drift Correction
```julia
using SMLMDriftCorrection

(smld_corrected, info) = driftcorrect(smld, DriftConfig())
traj = drift_trajectory(info.model)
```

### Iterative with Custom Config
```julia
config = DriftConfig(quality=:iterative, degree=3, verbose=1)
(smld_corrected, info) = driftcorrect(smld, config)
```

### Continuous Mode with Chunking
```julia
config = DriftConfig(dataset_mode=:continuous, chunk_frames=4000)
(smld_corrected, info) = driftcorrect(smld, config)
traj = drift_trajectory(info.model; cumulative=true)
```

### Warm Start from Previous Result
```julia
config1 = DriftConfig(degree=2)
(smld1, info1) = driftcorrect(smld1, config1)

config2 = DriftConfig(warm_start=info1.model)
(smld2, info2) = driftcorrect(smld2, config2)
```

### Continue Iterating
```julia
(smld1, info1) = driftcorrect(smld, DriftConfig(quality=:singlepass))
(smld2, info2) = driftcorrect(smld, info1; max_iterations=5)
```

Pass the original, uncorrected `smld`. Continuation reuses the original run's `dataset_mode` and chunking, including for chunked continuous results.

### Align Independent SMLDs (Shift Only)
```julia
(aligned, info) = align_smld([smld_ch1, smld_ch2]; transform=:shift, method=:fft)
info.shifts  # [zeros(2), [dx, dy]]
```

### Align with Affine (Multi-Channel Registration)
```julia
# Recovers anisotropic scale, rotation, shear, translation
(aligned, info) = align_smld([smld_ch1, smld_ch2]; transform=:affine)
info.transforms[2]  # AffineTransform2D(θ, scale, tx, ty)
# ~3nm median residual, ~13s on 1M locs
```
