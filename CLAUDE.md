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
DriftInfo (output struct with model, timing, convergence, roi_indices)
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
- `roi_selection.jl`: Auto-ROI subsampling (`calculate_n_locs_required`, `find_dense_roi`)
- `typedefs.jl`: Abstract types, `InterShift`, `DriftInfo`

### Adaptive Neighbor Optimization (Intra-dataset)

The `NeighborState` struct tracks KDTree neighbors and rebuilds only when drift changes significantly (threshold: 100 nm). This avoids O(N log N) tree rebuilds on every optimizer iteration.

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

- `:registered` (default): Datasets are independent acquisitions with spatial overlap. Uses entropy-based inter-dataset alignment via `findinter!()`.
- `:continuous`: One long acquisition split into files. Uses polynomial endpoint chaining (warmstart) for inter-dataset alignment since chunks have temporal but not spatial overlap.

**Chunking guidance for continuous mode**: Consider chunking when acquisitions exceed ~4000 frames, using `chunk_frames=4000` as a reasonable maximum. Shorter acquisitions can use a single polynomial with moderate degree. The warmstart mechanism initializes each chunk's polynomial from the previous chunk's endpoint for smooth transitions.
```julia
# Short acquisition (<4000 frames) - single polynomial
(smld_corrected, info) = driftcorrect(smld; dataset_mode=:continuous, degree=3)

# Long acquisition - chunk into ~4000 frame segments
(smld_corrected, info) = driftcorrect(smld; dataset_mode=:continuous, chunk_frames=4000)

# Multi-file data - datasets already separate, no explicit chunking needed
(smld_corrected, info) = driftcorrect(smld; dataset_mode=:continuous)
```

For trajectory plotting in continuous mode:
```julia
traj = drift_trajectory(info.model; cumulative=true)
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

### Auto-ROI Parameters

- `auto_roi=false`: Set to `true` for faster processing using a dense spatial subset (~15% of data). Trades some accuracy (~1.4nm vs ~0.5nm RMSD) for speed.
- `σ_loc=0.010`: Typical localization precision (μm) for ROI sizing
- `σ_target=0.001`: Target drift precision (μm) for ROI sizing
- `roi_safety_factor=4.0`: Safety multiplier for required localizations

When `auto_roi=true`, selects a contiguous rectangular region from the densest part of the FOV. This preserves blink pairs from the same emitters which is essential for entropy-based optimization.

## Units

All distance units are in **micrometers (μm)**.

## Key Dependencies

- **SMLMData.jl**: SMLD, Emitter types
- **NearestNeighbors.jl**: KDTree for efficient spatial queries
- **Optim.jl**: Optimization backend (10000 iterations, convergence tolerances)
- **LegendrePolynomials.jl**: Orthogonal polynomial basis
- **SMLMSim.jl**: Test data generation
