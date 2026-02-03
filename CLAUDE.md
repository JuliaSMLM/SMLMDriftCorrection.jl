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
│
DriftInfo (output struct with model, timing, convergence info)
```

### Key Data Flow

1. `driftcorrect(smld)` creates a `LegendrePolynomial` model
2. `findintra!()` optimizes intra-dataset drift per dataset (parallelized with `Threads.@threads`)
3. `findinter!()` sequentially aligns datasets
4. `correctdrift(smld, model)` applies the final corrections
5. Returns tuple `(smld_corrected, info::DriftInfo)`

### Source Files

- `interface.jl`: Main `driftcorrect()` function with quality tiers - start here
- `legendre.jl`: `LegendrePolynomial`, `IntraLegendre`, `LegendrePoly1D` types and evaluation
- `intrainter.jl`: `findintra!()`, `findinter!()`, `applydrift()`, `correctdrift()`
- `costfuns.jl`: `NeighborState`, adaptive entropy cost functions
- `cost_entropy.jl`: Entropy calculations (KL divergence, `entropy_HD`, `ub_entropy`)
- `utilities.jl`: `filter_emitters()`, `chunk_smld()`, `drift_trajectory()`
- `crosscorr.jl`: Cross-correlation helpers (`findshift`, `histimage2D`, `crosscorr2D`)
- `typedefs.jl`: Abstract types, `InterShift`, `DriftInfo`

### Adaptive Neighbor Optimization (Intra-dataset)

The `NeighborState` struct tracks KDTree neighbors and rebuilds only when drift changes significantly (threshold: 0.5 μm). This avoids O(N log N) tree rebuilds on every optimizer iteration.

### Inter-dataset Alignment (Merged Cloud Entropy)

Inter-dataset alignment uses a "merged cloud" entropy approach:
1. Combine shifted dataset with reference dataset(s)
2. Compute entropy of the combined point cloud
3. Optimizer finds shift that minimizes entropy (tighter combined cloud = better alignment)

This properly incorporates localization uncertainties (σ) and works well for real SMLM data where datasets image the same underlying structure.

### Quality Tiers

- `:fft`: Fast cross-correlation only (~10x faster, less accurate)
- `:singlepass` (default): Single pass of intra then inter correction
- `:iterative`: Full convergence with intra↔inter iteration

### Dataset Modes

Both modes use the same entropy-based alignment algorithm. The difference is semantic:

- `:registered` (default): Datasets are independent acquisitions. Use default trajectory plotting.
- `:continuous`: One long acquisition split into files. Use `drift_trajectory(model; cumulative=true)` for plotting.

For continuous mode, optional chunking splits datasets for finer-grained correction:
```julia
(smld_corrected, info) = driftcorrect(smld; dataset_mode=:continuous, n_chunks=10)
```

## Usage Patterns

### Basic Usage
```julia
using SMLMDriftCorrection

# Returns tuple (smld_corrected, info::DriftInfo)
(smld_corrected, info) = driftcorrect(smld)

# Access model for trajectory extraction
traj = drift_trajectory(info.model)
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

### Warm Start
```julia
# First dataset
(smld1_corrected, info1) = driftcorrect(smld1; degree=2)

# Second dataset - use model from first as starting point
(smld2_corrected, info2) = driftcorrect(smld2; warm_start=info1.model)
```

### Filtering to ROI
```julia
x = [e.x for e in smld.emitters]
y = [e.y for e in smld.emitters]
mask = (x .> 64.0) .& (x .< 128.0) .& (y .> 64.0) .& (y .< 128.0)
smld_roi = filter_emitters(smld, mask)
```

## Key Parameters

- `quality=:singlepass`: Quality tier (`:fft`, `:singlepass`, `:iterative`)
- `degree=2`: Polynomial degree for intra-dataset drift
- `maxn=200`: Maximum neighbors for entropy calculation
- `dataset_mode=:registered`: How to handle multi-dataset alignment
- `chunk_frames` / `n_chunks`: Chunking for continuous mode
- `max_iterations=10`: Maximum iterations for `:iterative` mode
- `convergence_tol=0.001`: Convergence tolerance (μm) for `:iterative` mode
- `warm_start=nothing`: Previous model for warm starting optimization
- `verbose=0`: 0=quiet, 1=info, 2=debug

## Units

All distance units are in **micrometers (μm)**.

## Key Dependencies

- **SMLMData.jl**: SMLD, Emitter types
- **NearestNeighbors.jl**: KDTree for efficient spatial queries
- **Optim.jl**: Optimization backend (10000 iterations, convergence tolerances)
- **LegendrePolynomials.jl**: Orthogonal polynomial basis
- **SMLMSim.jl**: Test data generation
