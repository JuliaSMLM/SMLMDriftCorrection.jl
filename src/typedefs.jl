abstract type AbstractDriftModel end

abstract type AbstractIntraDrift1D end

abstract type AbstractIntraDrift end

abstract type AbstractIntraInter <: AbstractDriftModel end

mutable struct InterShift
    ndims::Int
    dm::Vector{<:Real}
end

"""
    DriftResult

Result type for drift correction, providing a unified interface across all quality modes.
Supports dispatch for continuation (driftcorrect(result; kw...)).

# Fields
- `smld::SMLD`: Drift-corrected localization data
- `model::LegendrePolynomial`: Fitted drift model (intra + inter)
- `iterations::Int`: Number of iterations completed (0 for :fft, 1 for :singlepass, N for :iterative)
- `converged::Bool`: Whether convergence criterion was met (always true for :fft/:singlepass)
- `entropy::Float64`: Final entropy value after correction
- `history::Vector{Float64}`: Entropy per iteration (empty for :fft)

# Usage
```julia
result = driftcorrect(smld)
result.smld        # corrected data
result.converged   # check convergence
result.entropy     # final value
plot(result.history)  # diagnostics
```
"""
mutable struct DriftResult{S<:SMLD, M<:AbstractIntraInter}
    smld::S
    model::M
    iterations::Int
    converged::Bool
    entropy::Float64
    history::Vector{Float64}
end
