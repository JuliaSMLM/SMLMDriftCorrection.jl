abstract type AbstractDriftModel end

abstract type AbstractIntraDrift1D end

abstract type AbstractIntraDrift end

abstract type AbstractIntraInter <: AbstractDriftModel end

mutable struct InterShift
    ndims::Int
    dm::Vector{Float64}
end

"""
    DriftConfig <: AbstractSMLMConfig

Configuration for drift correction, holding all algorithm parameters. Constructed with keyword
arguments; all fields have sensible defaults.

# Fields

| Field | Default | Description |
|:------|:--------|:------------|
| `quality` | `:singlepass` | Quality tier: `:fft`, `:singlepass`, or `:iterative` |
| `degree` | `2` | Legendre polynomial degree for intra-dataset drift |
| `dataset_mode` | `:registered` | Multi-dataset handling: `:registered` or `:continuous` |
| `chunk_frames` | `0` | Split datasets into chunks of N frames (0 = no chunking) |
| `n_chunks` | `0` | Alternative: number of chunks per dataset (0 = use chunk_frames) |
| `maxn` | `100` | Maximum neighbors for entropy calculation |
| `max_iterations` | `10` | Maximum iterations for `:iterative` mode |
| `convergence_tol` | `0.001` | Convergence tolerance in μm for `:iterative` mode |
| `warm_start` | `nothing` | Previous `info.model` for warm starting optimization |
| `verbose` | `0` | Verbosity level: 0=quiet, 1=info, 2=debug |
| `auto_roi` | `false` | Use dense ROI subset for faster estimation |
| `σ_loc` | `0.010` | Typical localization precision (μm) for ROI sizing |
| `σ_target` | `0.001` | Target drift precision (μm) for ROI sizing |
| `roi_safety_factor` | `4.0` | Safety multiplier for required localizations |

# Example
```jldoctest
julia> config = DriftConfig(quality=:iterative, degree=3);

julia> config.quality
:iterative

julia> config.degree
3

julia> config.convergence_tol
0.001
```
"""
@kwdef struct DriftConfig <: AbstractSMLMConfig
    quality::Symbol = :singlepass
    degree::Int = 2
    dataset_mode::Symbol = :registered
    chunk_frames::Int = 0
    n_chunks::Int = 0
    maxn::Int = 100
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
