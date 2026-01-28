# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

SMLMDriftCorrection.jl is a Julia package for fiducial-free drift correction in Single Molecule Localization Microscopy (SMLM). It works on both 2D and 3D localization data and is part of the JuliaSMLM ecosystem (depends on SMLMData.jl for core data structures).

## Development Commands

```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Build documentation
julia --project=docs docs/make.jl
```

### Project Environments
- Root `Project.toml`: Main package
- `dev/Project.toml`: Development experiments
- `examples/Project.toml`: Example scripts
- `docs/Project.toml`: Documentation

## Code Architecture

### Algorithm Overview

The drift correction has two phases:
1. **Intra-dataset correction**: Corrects drift within each dataset using Legendre polynomial models (optimized via entropy minimization)
2. **Inter-dataset correction**: Aligns datasets to each other using constant shifts

The algorithm uses **entropy minimization** as the cost function with **adaptive KDTree neighbor rebuilding** for efficiency.

### Type Hierarchy

```
AbstractDriftModel
├── AbstractIntraInter (combined intra+inter models)
│   └── LegendrePolynomial (main model used by driftcorrect)
│
AbstractIntraDrift (per-dataset models)
└── IntraLegendre (wraps LegendrePoly1D per dimension)
│
AbstractIntraDrift1D (1D polynomial components)
└── LegendrePoly1D (normalized to [-1, 1] time domain)
│
InterShift (per-dataset constant shift)
```

### Key Data Flow

1. `driftcorrect(smld)` creates a `LegendrePolynomial` model
2. `findintra!()` optimizes intra-dataset drift per dataset (parallelized with `Threads.@threads`)
3. `findinter!()` sequentially aligns datasets
4. `correctdrift(smld, model)` applies the final corrections
5. Returns `NamedTuple` with `smld` (corrected) and `model` (for trajectory extraction)

### Source Files

- `interface.jl`: Main `driftcorrect()` function - start here
- `legendre.jl`: `LegendrePolynomial`, `IntraLegendre`, `LegendrePoly1D` types and evaluation
- `intrainter.jl`: `findintra!()`, `findinter!()`, `applydrift()`, `correctdrift()`
- `costfuns.jl`: `NeighborState`, adaptive entropy cost functions
- `cost_entropy.jl`: Entropy calculations (KL divergence, `entropy_HD`, `ub_entropy`)
- `utilities.jl`: `filter_emitters()`, `chunk_smld()`, `drift_trajectory()`
- `crosscorr.jl`: Cross-correlation helpers (`findshift`, `histimage2D`, `crosscorr2D`)
- `typedefs.jl`: Abstract types and `InterShift`

### Adaptive Neighbor Optimization (Intra-dataset)

The `NeighborState` struct tracks KDTree neighbors and rebuilds only when drift changes significantly (threshold: 0.5 μm). This avoids O(N log N) tree rebuilds on every optimizer iteration.

### Inter-dataset Alignment (Merged Cloud Entropy)

Inter-dataset alignment uses a "merged cloud" entropy approach:
1. Combine shifted dataset with reference dataset(s)
2. Compute entropy of the combined point cloud
3. Optimizer finds shift that minimizes entropy (tighter combined cloud = better alignment)

This properly incorporates localization uncertainties (σ) and works well for real SMLM data where datasets image the same underlying structure.

### Dataset Modes

- `:registered` (default): Datasets are independent (stage registered between acquisitions)
- `:continuous`: Drift accumulates across datasets (one long acquisition split into files)

For continuous mode, optional chunking splits datasets for finer-grained correction:
```julia
result = driftcorrect(smld; dataset_mode=:continuous, n_chunks=10)
```

## Usage Patterns

### Basic Usage
```julia
using SMLMDriftCorrection

# Returns NamedTuple with smld and model
result = driftcorrect(smld)
smld_corrected = result.smld

# Or destructure:
(; smld, model) = driftcorrect(smld)

# Extract drift trajectory for plotting
traj = drift_trajectory(model)
# traj.frames, traj.x, traj.y ready for plotting
```

### Testing with Simulated Drift
```julia
DC = SMLMDriftCorrection

# Create random drift model
drift_model = DC.LegendrePolynomial(smld; degree=2, initialize="random", rscale=0.1)

# Apply drift (for testing)
smld_drifted = DC.applydrift(smld, drift_model)

# Correct drift (inverse operation - exact recovery)
smld_recovered = DC.correctdrift(smld_drifted, drift_model)
```

### Filtering to ROI
```julia
x = [e.x for e in smld.emitters]
y = [e.y for e in smld.emitters]
mask = (x .> 64.0) .& (x .< 128.0) .& (y .> 64.0) .& (y .< 128.0)
smld_roi = filter_emitters(smld, mask)
```

## Key Parameters

- `degree=2`: Polynomial degree for intra-dataset drift
- `maxn=200`: Maximum neighbors for entropy calculation
- `dataset_mode=:registered`: How to handle multi-dataset alignment
- `chunk_frames` / `n_chunks`: Chunking for continuous mode
- `verbose=0`: 0=quiet, 1=info, 2=debug

## Units

All distance units are in **micrometers (μm)**.

## Key Dependencies

- **SMLMData.jl**: SMLD, Emitter types
- **NearestNeighbors.jl**: KDTree for efficient spatial queries
- **Optim.jl**: Optimization backend (10000 iterations, convergence tolerances)
- **LegendrePolynomials.jl**: Orthogonal polynomial basis
- **SMLMSim.jl**: Test data generation

## Branch Notes

This branch (`adaptive-neighbors`) has simplified the codebase to use only:
- Legendre polynomial models (better optimization conditioning than standard polynomials)
- Entropy-based cost function (removed Kdtree cost function option)
- Adaptive neighbor rebuilding for intra-dataset optimization
- Merged cloud entropy for inter-dataset alignment (combines datasets, minimizes joint entropy)
