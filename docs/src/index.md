```@meta
CurrentModule = SMLMDriftCorrection
```

# SMLMDriftCorrection

Documentation for [SMLMDriftCorrection](https://github.com/JuliaSMLM/SMLMDriftCorrection.jl).

## Overview

SMLMDriftCorrection.jl provides fiducial-free drift correction for Single Molecule Localization Microscopy (SMLM) data. The algorithm corrects both intra-dataset drift (within each acquisition) and inter-dataset drift (between acquisitions).

## Quick Start

```julia
using SMLMDriftCorrection

# Correct drift - returns (corrected_data, info) tuple
(smld_corrected, info) = driftcorrect(smld)

# Access optimization metadata
println("Elapsed: $(info.elapsed_ns / 1e9) seconds")
println("Converged: $(info.converged)")
println("Iterations: $(info.iterations)")
```

## Return Value

The `driftcorrect` function returns a tuple `(smld_corrected, info)`:

- `smld_corrected`: The drift-corrected SMLD structure
- `info`: A `DriftInfo` struct containing optimization metadata

### DriftInfo Fields

| Field | Type | Description |
|-------|------|-------------|
| `model` | `AbstractIntraInter` | Fitted drift model |
| `elapsed_ns` | `UInt64` | Wall time in nanoseconds |
| `backend` | `Symbol` | Computation backend (`:cpu`) |
| `iterations` | `Int` | Total optimization iterations |
| `converged` | `Bool` | Whether all optimizations converged |
| `entropy` | `Float64` | Final total cost value |
| `history` | `Vector{Float64}` | Per-dataset final costs |

## Cost Functions

Two cost functions are available:

- **Kdtree** (default): Sum of nearest neighbor distances using KD-tree spatial queries
- **Entropy**: Upper bound on entropy of Gaussian mixture model

```julia
# Using Kdtree (default)
(smld_dc, info) = driftcorrect(smld; cost_fun="Kdtree")

# Using Entropy
(smld_dc, info) = driftcorrect(smld; cost_fun="Entropy", maxn=100)
```

## API Reference

```@index
```

```@autodocs
Modules = [SMLMDriftCorrection]
```
