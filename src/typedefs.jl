abstract type AbstractDriftModel end

abstract type AbstractIntraDrift1D end

abstract type AbstractIntraDrift end

abstract type AbstractIntraInter <: AbstractDriftModel end

mutable struct InterShift
    ndims::Int
    dm::Vector{<:Real}
end

"""
    DriftConfig <: AbstractSMLMConfig

Configuration for drift correction, holding all algorithm parameters.

# Fields
- `quality::Symbol`: Quality tier (`:fft`, `:singlepass`, `:iterative`)
- `degree::Int`: Polynomial degree for intra-dataset drift model
- `dataset_mode::Symbol`: Multi-dataset handling (`:registered` or `:continuous`)
- `chunk_frames::Int`: For continuous mode, split each dataset into chunks of this many frames
- `n_chunks::Int`: Alternative to chunk_frames - specify number of chunks per dataset
- `maxn::Int`: Maximum number of neighbors for entropy calculation
- `max_iterations::Int`: Maximum iterations for `:iterative` mode
- `convergence_tol::Float64`: Convergence tolerance (μm) for `:iterative` mode
- `warm_start::Union{Nothing, AbstractIntraInter}`: Previous model for warm starting
- `verbose::Int`: Verbosity level (0=quiet, 1=info, 2=debug)
- `auto_roi::Bool`: Use dense ROI subset for faster estimation
- `σ_loc::Float64`: Typical localization precision (μm) for ROI sizing
- `σ_target::Float64`: Target drift precision (μm) for ROI sizing
- `roi_safety_factor::Float64`: Safety multiplier for required localizations

# Example
```julia
config = DriftConfig(; quality=:iterative, degree=3, verbose=1)
(smld_corrected, info) = driftcorrect(smld, config)
```
"""
@kwdef struct DriftConfig <: AbstractSMLMConfig
    quality::Symbol = :singlepass
    degree::Int = 2
    dataset_mode::Symbol = :registered
    chunk_frames::Int = 0
    n_chunks::Int = 0
    maxn::Int = 200
    max_iterations::Int = 10
    convergence_tol::Float64 = 0.001
    warm_start::Union{Nothing, AbstractIntraInter} = nothing
    verbose::Int = 0
    auto_roi::Bool = false
    σ_loc::Float64 = 0.010
    σ_target::Float64 = 0.001
    roi_safety_factor::Float64 = 4.0
end

"""
    DriftInfo <: AbstractSMLMInfo

Metadata from drift correction, returned as second element of tuple.
Supports warm start via `info.model`.

# Fields
- `model::LegendrePolynomial`: Fitted drift model (intra + inter)
- `elapsed_s::Float64`: Wall time in seconds
- `backend::Symbol`: Computation backend (`:cpu`)
- `iterations::Int`: Number of iterations completed (0 for :fft, 1 for :singlepass, N for :iterative)
- `converged::Bool`: Whether convergence criterion was met (always true for :fft/:singlepass)
- `entropy::Float64`: Final entropy value after correction
- `history::Vector{Float64}`: Entropy per iteration (empty for :fft)
- `roi_indices::Union{Nothing, Vector{Int}}`: Indices used for ROI subsampling (nothing if not used)

# Usage
```julia
(smld_corrected, info) = driftcorrect(smld)
info.converged    # check convergence
info.entropy      # final value
info.elapsed_s    # timing
plot(info.history)  # diagnostics
info.roi_indices  # ROI used for estimation (nothing if auto_roi=false)

# Warm start from previous result
(smld2, info2) = driftcorrect(smld2; warm_start=info.model)
```
"""
struct DriftInfo{M<:AbstractIntraInter} <: AbstractSMLMInfo
    model::M
    elapsed_s::Float64
    backend::Symbol
    iterations::Int
    converged::Bool
    entropy::Float64
    history::Vector{Float64}
    roi_indices::Union{Nothing, Vector{Int}}
end
