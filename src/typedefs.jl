abstract type AbstractDriftModel end

abstract type AbstractIntraDrift1D end

abstract type AbstractIntraDrift end

abstract type AbstractIntraInter <: AbstractDriftModel end

mutable struct InterShift
    ndims::Int
    dm::Vector{<:Real}
end

"""
    DriftInfo

Metadata from drift correction optimization.

# Fields
- `model::AbstractIntraInter`: Fitted drift model (for warm starts or trajectory extraction)
- `elapsed_ns::UInt64`: Wall-clock time in nanoseconds
- `backend::Symbol`: Computation backend (`:cpu`)
- `iterations::Int`: Total optimization iterations across all datasets
- `converged::Bool`: Whether all optimizations converged
- `entropy::Float64`: Final entropy/cost value (sum across datasets)
- `history::Vector{Float64}`: Per-dataset final cost values
"""
struct DriftInfo
    model::AbstractIntraInter
    elapsed_ns::UInt64
    backend::Symbol
    iterations::Int
    converged::Bool
    entropy::Float64
    history::Vector{Float64}
end
