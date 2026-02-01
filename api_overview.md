# SMLMDriftCorrection API Overview

AI-parseable API reference for SMLMDriftCorrection.jl.

## Main Function

### driftcorrect

```julia
(smld_corrected, info) = driftcorrect(smld::SMLD; kwargs...)
```

Performs fiducial-free drift correction on SMLM localization data.

**Arguments:**
- `smld::SMLD`: Input localization data (2D or 3D)

**Keyword Arguments:**
- `intramodel::String = "Polynomial"`: Model type (`"Polynomial"` or `"LegendrePoly"`)
- `cost_fun::String = "Kdtree"`: Cost function (`"Kdtree"` or `"Entropy"`)
- `cost_fun_intra::String = ""`: Override for intra-dataset cost function
- `cost_fun_inter::String = ""`: Override for inter-dataset cost function
- `degree::Int = 2`: Polynomial degree for intra-dataset drift
- `d_cutoff::AbstractFloat = 0.01`: Distance cutoff in μm (Kdtree)
- `maxn::Int = 200`: Max neighbors (Entropy)
- `histbinsize::AbstractFloat = -1.0`: Histogram bin size for cross-correlation (< 0 disables)
- `verbose::Int = 0`: Verbosity level (0=quiet, 1=info, 2=debug)

**Returns:**
- `smld_corrected::SMLD`: Drift-corrected localization data
- `info::DriftInfo`: Optimization metadata

**Example:**
```julia
using SMLMDriftCorrection

(smld_corrected, info) = driftcorrect(smld; degree=2, cost_fun="Entropy")
println("Converged: $(info.converged), Iterations: $(info.iterations)")
```

## Types

### DriftInfo

```julia
struct DriftInfo
    model::AbstractIntraInter   # Fitted drift model
    elapsed_ns::UInt64          # Wall time in nanoseconds
    backend::Symbol             # :cpu
    iterations::Int             # Total optimization iterations
    converged::Bool             # Convergence status
    entropy::Float64            # Final total cost
    history::Vector{Float64}    # Per-dataset costs
end
```

**Fields:**
- `model`: The fitted drift model, can be used for trajectory extraction or warm starts
- `elapsed_ns`: Wall-clock time in nanoseconds
- `backend`: Computation backend (currently always `:cpu`)
- `iterations`: Sum of optimization iterations across all datasets
- `converged`: `true` if all optimizations converged
- `entropy`: Sum of final cost values across all datasets
- `history`: Vector of per-dataset final cost values

## Utility Functions

### filter_emitters

```julia
smld_filtered = filter_emitters(smld::SMLD, mask::BitVector)
```

Filter emitters by boolean mask.

**Arguments:**
- `smld::SMLD`: Input localization data
- `mask::BitVector`: Boolean mask for emitters to keep

**Returns:**
- `smld_filtered::SMLD`: Filtered localization data

## Internal Functions (Not Exported)

### applydrift / correctdrift

```julia
smld_drifted = applydrift(smld::SMLD, model::AbstractIntraInter)
smld_corrected = correctdrift(smld::SMLD, model::AbstractIntraInter)
```

Apply or correct drift using a model. Useful for testing.

### Polynomial / LegendrePolynomial

```julia
model = Polynomial(smld; degree=2, initialize="zeros", rscale=0.1)
model = LegendrePolynomial(smld; degree=2, initialize="zeros", rscale=0.1)
```

Create drift models. `initialize` can be `"zeros"`, `"random"`, or `"continuous"`.

## Units

All distance units are in **micrometers (μm)**.
