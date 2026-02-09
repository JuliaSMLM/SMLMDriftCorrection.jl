# SMLMDriftCorrection.jl API Overview

AI-parseable API reference for SMLMDriftCorrection.jl. All distance units are in micrometers (μm).

## Exports Summary

- **Types:** 2 (`DriftConfig`, `DriftInfo`)
- **Functions:** 3 (`driftcorrect`, `filter_emitters`, `drift_trajectory`)

## Key Concepts

Drift correction uses entropy minimization to find polynomial drift models that minimize spatial entropy of localization data. The algorithm has two phases: intra-dataset (polynomial drift within each acquisition) and inter-dataset (constant shift between acquisitions). The primary interface follows the `f(data, config) -> (result, info)` tuple pattern used across JuliaSMLM.

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
- `maxn::Int = 200`: Maximum neighbors for entropy calculation
- `max_iterations::Int = 10`: Maximum iterations for `:iterative` mode
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
- `entropy::Float64`: Final entropy value
- `history::Vector{Float64}`: Entropy per iteration (for diagnostics)
- `roi_indices::Union{Nothing, Vector{Int}}`: Indices used for ROI subset (`nothing` if `auto_roi=false`)

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
- `info`: Previous `DriftInfo` for continuation (third dispatch)

**Example:**
```julia
config = DriftConfig(quality=:iterative, degree=3, verbose=1)
(smld_corrected, info) = driftcorrect(smld, config)
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
traj = drift_trajectory(info.model)
traj = drift_trajectory(info.model; cumulative=true)  # for continuous mode
```

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

**Initialize options:** `"zeros"` (default), `"random"`. Continuous mode warmstart is handled by `initialize_from_endpoint!`, not an initialize string.

### applydrift / correctdrift
```julia
applydrift(smld::SMLD, model::AbstractIntraInter) -> SMLD
correctdrift(smld::SMLD, model::AbstractIntraInter) -> SMLD
```
Apply or correct drift using a model. `correctdrift` is the inverse of `applydrift`.

### findshift
```julia
findshift(smld1::SMLD, smld2::SMLD; histbinsize=0.1) -> Vector{Float64}
```
Find shift between two SMLDs via cross-correlation of histogram images. Returns `[dx, dy]` or `[dx, dy, dz]`.

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

### initialize_from_endpoint!
```julia
initialize_from_endpoint!(intra_new::IntraLegendre, intra_prev::IntraLegendre, n_frames_prev::Int)
```
Initialize polynomial for continuous mode warmstart. Copies coefficients from `intra_prev` and shifts so `intra_new`'s startpoint (frame 1) equals `intra_prev`'s endpoint (frame `n_frames_prev`). Initialization only - optimizer refines from here.

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

AbstractSMLMConfig
└── DriftConfig (exported)

AbstractSMLMInfo
└── DriftInfo{M} (exported)
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
findshift_damped(smld1::SMLD, smld2::SMLD; histbinsize=1.0, prior_shift=[0.0, 0.0], prior_sigma=1.0) -> Vector{Float64}
```
Find shift with Gaussian damping centered at `prior_shift`. Used to refine outlier shifts by searching near an expected location. Applies a Gaussian weight to the cross-correlation centered at `prior_shift` with width `prior_sigma`.

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
           precomputed_corrected=nothing, regularization_target=nothing, regularization_lambda=0.0)
```
Optimize inter-dataset shift for `dataset_n` against `ref_datasets` using merged-cloud entropy. Optional regularization: `entropy + λ*||shift - target||²`.

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
- **First inter-dataset pass** (all datasets vs DS1) is threaded using a precomputed snapshot of corrected coordinates.
- **Refinement pass** (each dataset vs all earlier datasets) is sequential to preserve ordering dependencies.
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
(smld2, info2) = driftcorrect(smld1, info1; max_iterations=5)
```
