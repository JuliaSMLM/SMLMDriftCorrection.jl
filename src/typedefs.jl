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

# Usage
```julia
(smld_corrected, info) = driftcorrect(smld)
info.converged    # check convergence
info.entropy      # final value
info.elapsed_ns   # timing
plot(info.history)  # diagnostics

# Warm start from previous result
(smld2, info2) = driftcorrect(smld2; warm_start=info.model)
```
"""
struct DriftInfo{M<:AbstractIntraInter}
    model::M
    elapsed_s::Float64
    backend::Symbol
    iterations::Int
    converged::Bool
    entropy::Float64
    history::Vector{Float64}
end
