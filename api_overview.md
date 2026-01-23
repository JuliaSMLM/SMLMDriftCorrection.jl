# SMLMDriftCorrection.jl API Overview

AI-parseable API reference for SMLMDriftCorrection.jl. All distance units are in micrometers (μm).

## Exported Functions

### driftcorrect
```julia
driftcorrect(smld::SMLD; degree=2, dataset_mode=:registered, chunk_frames=0, n_chunks=0, maxn=200, verbose=0) -> NamedTuple
```
Main interface for drift correction. Returns `(smld=corrected_smld, model=LegendrePolynomial)`.

**Parameters:**
- `smld`: SMLD structure with localization coordinates
- `degree`: Polynomial degree for intra-dataset drift (default: 2)
- `dataset_mode`: `:registered` (independent datasets) or `:continuous` (drift accumulates)
- `chunk_frames`: Split datasets into chunks of N frames (0 = no chunking)
- `n_chunks`: Alternative - specify number of chunks per dataset
- `maxn`: Max neighbors for entropy calculation (default: 200)
- `verbose`: 0=quiet, 1=info, 2=debug

### filter_emitters
```julia
filter_emitters(smld::SMLD, keep::Union{AbstractVector,AbstractRange}) -> SMLD
filter_emitters(smld::SMLD, keep::Integer) -> SMLD
```
Select emitters by boolean mask, indices, or single index.

### drift_trajectory
```julia
drift_trajectory(model::AbstractIntraInter; dataset=nothing, frames=nothing) -> NamedTuple
```
Extract drift trajectory from fitted model for plotting.

**Returns:** `(frames, x, y, dataset)` or `(frames, x, y, z, dataset)` for 3D

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

**Initialize options:** `"zeros"`, `"random"`, `"continuous"`

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

## Types

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
- `coefficients::Vector{Real}`: Coefficients for P_1, P_2, ..., P_degree
- `n_frames::Int`: Frame count for time normalization

### InterShift
```julia
InterShift(ndims::Int)
```
Per-dataset constant shift for inter-dataset alignment.

**Fields:**
- `ndims::Int`: Number of spatial dimensions
- `dm::Vector{Real}`: Shift values [dx, dy] or [dx, dy, dz]

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
```
Upper bound on entropy using k-nearest neighbors.

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
Compute cross-correlation between two images via FFT (FourierTools.ccorr).

## Internal Optimization

### findintra!
```julia
findintra!(intra::AbstractIntraDrift, smld::SMLD, dataset::Int, maxn::Int)
```
Optimize intra-dataset drift using entropy minimization with adaptive KDTree rebuilding.

### findinter!
```julia
findinter!(dm::AbstractIntraInter, smld::SMLD, dataset1::Int, dataset2::Vector{Int}, maxn::Int)
```
Optimize inter-dataset shift between dataset1 and reference datasets in dataset2.

### NeighborState
```julia
NeighborState(N::Int, k::Int, rebuild_threshold::Real)
```
Tracks KDTree neighbors for adaptive rebuilding during optimization. Rebuilds only when drift changes by more than `rebuild_threshold` (default: 0.5 μm).

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
