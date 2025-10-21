# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

SMLMDriftCorrection.jl is a Julia package for drift correction in Single Molecule Localization Microscopy (SMLM). It implements fiducial-free drift correction algorithms that work on both 2D and 3D localization data. The package is part of the JuliaSMLM ecosystem and depends on SMLMData.jl for core data structures.

## Development Commands

### Testing
```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run tests from REPL
julia --project
using Pkg
Pkg.test()
```

### Documentation
```bash
# Build documentation locally
julia --project=docs docs/make.jl
```

### Development Environment
The package includes several project environments:
- Root `Project.toml`: Main package dependencies
- `dev/Project.toml`: Development experiments
- `dev/loc_entropy/Project.toml`: Entropy-specific development
- `examples/Project.toml`: Example scripts
- `docs/Project.toml`: Documentation dependencies

Activate the appropriate environment:
```julia
using Pkg
Pkg.activate(".")  # or "dev", "examples", "docs"
```

## Code Architecture

### Core Algorithm Structure

The drift correction algorithm has two phases:
1. **Intra-dataset correction**: Corrects drift within each dataset (collection of movie frames) using polynomial models
2. **Inter-dataset correction**: Aligns datasets to each other, typically to dataset 1 as reference

### Type Hierarchy

```
AbstractDriftModel
├── AbstractIntraInter (combined intra+inter models)
│   ├── Polynomial (polynomial drift model)
│   └── LegendrePolynomial (Legendre polynomial model)
│
AbstractIntraDrift (intra-dataset models)
├── IntraPolynomial
└── IntraLegendrePolynomial (if implemented)
│
AbstractIntraDrift1D (1D drift components)
├── Polynomial1D
└── LegendrePoly1D (if implemented)
│
InterShift (inter-dataset shift)
```

### Key Data Structures

- **SMLD**: From SMLMData.jl - contains localization coordinates with fields:
  - `emitters`: Vector of Emitter2DFit or Emitter3DFit
  - `n_datasets`: Number of datasets
  - `n_frames`: Number of frames per dataset

- **Polynomial**: Main drift model combining intra and inter corrections
  - `intra`: Vector of IntraPolynomial (one per dataset)
  - `inter`: Vector of InterShift (one per dataset)

- **IntraPolynomial**: Per-dataset polynomial drift
  - `ndims`: Number of spatial dimensions (2 or 3)
  - `dm`: Vector of Polynomial1D (one per dimension)

### Source File Organization

- `src/SMLMDriftCorrection.jl`: Main module, exports `driftcorrect` and `filter_emitters`
- `src/interface.jl`: Main user-facing `driftcorrect()` function
- `src/intrainter.jl`: Core drift application/correction and `findintra!`/`findinter!` optimizers
- `src/polynomial.jl`: Polynomial drift model definitions and conversion functions
- `src/typedefs.jl`: Abstract types and InterShift structure
- `src/costfuns.jl`: Cost functions for optimization (Kdtree and Entropy methods)
- `src/cost_entropy.jl`: Entropy-based cost calculations
- `src/crosscorr.jl`: Cross-correlation drift correction (histbinsize > 0)
- `src/utilities.jl`: Helper functions

### Cost Functions

Two main approaches for measuring drift quality:

1. **Kdtree** (default): Sum of negative exponentials of k-nearest neighbor distances
   - Parameter: `d_cutoff` (distance cutoff in μm, default 0.01)
   - Uses NearestNeighbors.jl for efficient spatial queries

2. **Entropy**: Upper bound on statistical entropy of Gaussian mixture model
   - Parameter: `maxn` (max neighbors considered, default 200)
   - Based on [Cnossen2021] approach

3. **Cross-correlation**: Optional histogram-based correction for inter-dataset
   - Parameter: `histbinsize` (μm, default -1.0 means disabled)
   - Uses FourierTools.jl

### Parameter Conversion

The optimization uses flat parameter vectors (θ) internally:
- `intra2theta()` / `theta2intra!()`: Convert IntraPolynomial ↔ coefficients
- `inter2theta()` / `theta2inter!()`: Convert InterShift ↔ shift vector

### Optimization Flow

1. Initialize drift model with `Polynomial(smld; degree=2, initialize="zeros")`
2. For each dataset, optimize intra-dataset drift using `findintra!()`:
   - Convert model to θ vector
   - Minimize cost function using Optim.jl
   - Convert optimized θ back to model
3. Sequentially optimize inter-dataset drift using `findinter!()`:
   - First align all datasets to dataset 1
   - Then align each dataset to all earlier datasets
4. Apply final corrections with `correctdrift(smld, driftmodel)`

### Threading

- Intra-dataset corrections use `Threads.@threads` for parallelization
- Inter-dataset corrections run sequentially (commented threading code exists)

## Working with SMLD Data

All distance units are in micrometers (μm). The package works with SMLMData.jl structures:

```julia
using SMLMData
using SMLMDriftCorrection

# From simulation
smld = ... # SMLD structure from SMLMSim

# From SMITE MATLAB files
smd = SmiteSMD(path, file)   # *_Results.mat
smld = load_smite_2d(smd)

# Filter to ROI
x = [e.x for e in smld.emitters]
y = [e.y for e in smld.emitters]
roi_mask = (x .> 64.0) .& (x .< 128.0) .& (y .> 64.0) .& (y .< 128.0)
smld_roi = SMLMDriftCorrection.filter_emitters(smld, roi_mask)
```

## Common Patterns

### Applying and correcting drift
```julia
# Create drift model
drift_model = SMLMDriftCorrection.Polynomial(smld; degree=2, initialize="random")

# Apply drift (for testing)
smld_drifted = SMLMDriftCorrection.applydrift(smld, drift_model)

# Correct drift (inverse operation)
smld_corrected = SMLMDriftCorrection.correctdrift(smld_drifted, drift_model)
```

### Main drift correction interface
```julia
# Basic usage (Kdtree cost function)
smld_corrected = driftcorrect(smld)

# With Entropy cost function
smld_corrected = driftcorrect(smld; cost_fun="Entropy", maxn=100)

# With cross-correlation inter-dataset correction
smld_corrected = driftcorrect(smld; histbinsize=0.1)

# Custom cost functions for intra vs inter
smld_corrected = driftcorrect(smld; cost_fun_intra="Entropy", cost_fun_inter="Kdtree")
```

## Testing Strategy

The test suite (`test/runtests.jl`) uses SMLMSim to generate synthetic data:
1. Create simulated SMLD with known structure (Nmers)
2. Apply synthetic drift with random polynomial coefficients
3. Run drift correction
4. Compute RMSD between corrected and original coordinates
5. Assert RMSD is within tolerance (typically 1.0 μm for recovered drift)

Tests cover:
- Entropy calculations (2D and 3D)
- Cross-correlation shift finding (identity and imposed shift)
- Polynomial drift application/correction (exact recovery)
- Full drift correction pipeline with Kdtree, Entropy, and cross-correlation methods
- Both 2D and 3D workflows

## Key Dependencies

- **SMLMData.jl**: Core SMLM data structures (SMLD, Emitter types)
- **NearestNeighbors.jl**: KDTree for efficient spatial queries
- **Optim.jl**: Optimization backend (used with default settings, 10000 iterations)
- **FourierTools.jl**: Cross-correlation via FFT
- **SMLMSim.jl**: Simulation for testing

## References

The algorithms are based on:
- [Wester2021]: Kdtree-based drift correction (Scientific Reports)
- [Cnossen2021]: Entropy minimization approach (Optics Express)
- [SchodtWester2023]: SMITE MATLAB toolbox

See README.md for full citations.

## Branch Information

The `SMLMData` branch contains updates to work with newer versions of SMLMData.jl. Check the current branch when working on compatibility issues.
- This is part of the larger JuliaSMLM envriornment