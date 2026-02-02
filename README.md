# SMLMDriftCorrection

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSMLM.github.io/SMLMDriftCorrection.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSMLM.github.io/SMLMDriftCorrection.jl/dev)
[![Build Status](https://github.com/JuliaSMLM/SMLMDriftCorrection.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaSMLM/SMLMDriftCorrection.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/JuliaSMLM/SMLMDriftCorrection.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSMLM/SMLMDriftCorrection.jl)

## Overview

Fiducial-free drift correction for Single Molecule Localization Microscopy (SMLM). The algorithm uses entropy minimization with Legendre polynomial drift models to correct both intra-dataset drift (within each movie segment) and inter-dataset drift (between segments). All distance units are in μm.

## Installation
```julia
using Pkg
Pkg.add("SMLMDriftCorrection")
```

## Basic Example

```julia
using SMLMSim
using SMLMDriftCorrection
DC = SMLMDriftCorrection

# Simulate an Nmer dataset
smld_true, smld_model, smld_noisy = simulate(;
    ρ=1.0,                # emitters per μm²
    σ_psf=0.13,           # PSF width in μm (130nm)
    minphotons=50,        # minimum photons for detection
    ndatasets=10,         # number of independent datasets
    nframes=1000,         # frames per dataset
    framerate=50.0,       # frames per second
    pattern=Nmer2D(n=6, d=0.2),  # hexamer with 200nm diameter
    molecule=GenericFluor(; q=[0 50; 1e-2 0]),
    camera=IdealCamera(1:256, 1:256, 0.1)
)

# Create and apply synthetic drift for testing
drift_true = DC.LegendrePolynomial(smld_noisy; degree=2, initialize="random", rscale=0.1)
smld_drifted = DC.applydrift(smld_noisy, drift_true)

# Correct drift
(smld_corrected, info) = driftcorrect(smld_drifted; verbose=1)

# Extract drift trajectory for plotting
traj = drift_trajectory(info.model)
# traj.frames, traj.x, traj.y are ready for plotting

# Compute RMSD between original and corrected
N = length(smld_noisy.emitters)
rmsd = sqrt(sum(
    ([e.x for e in smld_corrected.emitters] .- [e.x for e in smld_noisy.emitters]).^2 .+
    ([e.y for e in smld_corrected.emitters] .- [e.y for e in smld_noisy.emitters]).^2
) / N)
println("RMSD = $rmsd μm")
```

## Common Workflows

### Generic 2D or 3D Data
```julia
using SMLMDriftCorrection

(smld_corrected, info) = driftcorrect(smld)

# Access model for trajectory extraction:
traj = drift_trajectory(info.model)
```

### SMITE Results.mat File
```julia
using SMLMData
using SMLMDriftCorrection

smd = SmiteSMD(path, file)   # *_Results.mat file
smld = load_smite_2d(smd)
(smld_corrected, info) = driftcorrect(smld; verbose=1)
```

### Selecting a ROI to Analyze
```julia
using SMLMDriftCorrection

x = [e.x for e in smld.emitters]
y = [e.y for e in smld.emitters]
mask = (x .> 64.0) .& (x .< 128.0) .& (y .> 64.0) .& (y .< 128.0)
smld_roi = filter_emitters(smld, mask)

(smld_corrected, info) = driftcorrect(smld_roi)
```

### Continuous Acquisition (Drift Accumulates Across Datasets)
```julia
using SMLMDriftCorrection

# For data where drift accumulates across files (one long acquisition)
(smld_corrected, info) = driftcorrect(smld; dataset_mode=:continuous)

# With chunking for finer-grained correction
(smld_corrected, info) = driftcorrect(smld; dataset_mode=:continuous, n_chunks=10)
```

### Warm Start (Reuse Model from Previous Correction)
```julia
using SMLMDriftCorrection

# First dataset
(smld1_corrected, info1) = driftcorrect(smld1; degree=2)

# Second dataset - use model from first as starting point
(smld2_corrected, info2) = driftcorrect(smld2; warm_start=info1.model)
```

## Interface

**driftcorrect** is the main interface for drift correction.

```julia
function driftcorrect(smld::SMLD;
    quality::Symbol = :singlepass,
    degree::Int = 2,
    dataset_mode::Symbol = :registered,
    chunk_frames::Int = 0,
    n_chunks::Int = 0,
    maxn::Int = 200,
    warm_start = nothing,
    verbose::Int = 0) -> (smld_corrected, info)
```

### Input
- **smld**: SMLD structure containing (X, Y) or (X, Y, Z) localization coordinates (μm)

### Keyword Arguments
- **quality**: Quality tier for correction:
  - `:fft`: Fast cross-correlation only (~10x faster, less accurate)
  - `:singlepass` (default): Single pass of intra then inter correction
  - `:iterative`: Full convergence with intra↔inter iteration
- **degree**: Polynomial degree for intra-dataset drift model (default: 2)
- **dataset_mode**: Semantic label for multi-dataset handling (algorithm is identical):
  - `:registered` (default): Datasets are independent acquisitions
  - `:continuous`: One long acquisition split into files (use `drift_trajectory(model; cumulative=true)` for plotting)
- **chunk_frames**: For continuous mode, split each dataset into chunks of this many frames (0 = no chunking)
- **n_chunks**: Alternative to chunk_frames - specify number of chunks per dataset (0 = no chunking)
- **maxn**: Maximum number of neighbors for entropy calculation (default: 200)
- **warm_start**: Previous model (`info.model`) for warm starting optimization (default: nothing)
- **verbose**: Verbosity level (0=quiet, 1=info, 2=debug)

### Output
Returns a tuple `(smld_corrected, info)` where `info::DriftInfo` contains:
- **model**: Fitted `LegendrePolynomial` drift model (use `drift_trajectory(info.model)` for plotting)
- **elapsed_ns**: Wall time in nanoseconds
- **backend**: Computation backend (`:cpu`)
- **iterations**: Number of iterations completed
- **converged**: Whether convergence was achieved
- **entropy**: Final entropy value
- **history**: Entropy per iteration (for diagnostics)

## Other Functions

- **drift_trajectory(model)**: Extract drift trajectory from model for plotting. Returns NamedTuple with `frames`, `x`, `y` (and `z` for 3D)
- **filter_emitters(smld, mask)**: Select emitters by boolean mask or indices
- **LegendrePolynomial(smld; degree, initialize, rscale)**: Create drift model directly
- **applydrift(smld, model)**: Apply drift model to data (for simulation/testing)
- **correctdrift(smld, model)**: Correct drift using model (inverse of applydrift)
- **findshift(smld1, smld2; histbinsize)**: Find shift between two SMLDs via cross-correlation
- **histimage2D/histimage3D**: Create histogram images from coordinates
- **crosscorr2D/crosscorr3D**: Cross-correlation via FFT
- **entropy_HD(σ_x, σ_y)**: Base entropy of localization uncertainties
- **ub_entropy(x, y, σ_x, σ_y; maxn)**: Upper bound entropy for drift optimization

## Algorithm

The algorithm uses **entropy minimization** with **Legendre polynomial** drift models:

1. **Intra-dataset correction**: For each dataset, fits a Legendre polynomial (degree 2 by default) to model drift over time. Uses KL divergence-based entropy as the cost function, with adaptive KDTree neighbor rebuilding for efficiency.

2. **Inter-dataset correction**: Aligns datasets to each other using constant shifts optimized via entropy minimization. All datasets align to dataset 1, then refine against all previous datasets.

The Legendre polynomial basis provides better optimization conditioning than standard polynomials because the basis functions are orthogonal over the normalized time domain [-1, 1].

## References

- **[Cnossen2021]** Jelmer Cnossen, Tao Ju Cui, Chirlmin Joo and Carlas Smith,
  "Drift correction in localization microscopy using entropy minimization",
  *Optics Express*, Volume 29, Number 18, August 30, 2021, 27961-27974,
  https://doi.org/10.1364/OE.426620

- **[Wester2021]** Michael J. Wester, David J. Schodt, Hanieh Mazloom-Farsibaf,
  Mohamadreza Fazel, Sandeep Pallikkuth and Keith A. Lidke, "Robust,
  fiducial-free drift correction for super-resolution imaging", *Scientific
  Reports*, Volume 11, Article 23672, December 8, 2021,
  https://doi.org/10.1038/s41598-021-02850-7

- **[SchodtWester2023]** David J. Schodt*, Michael J. Wester*, et al.,
  "SMITE: Single Molecule Imaging Toolbox Extraordinaire (MATLAB)",
  *Journal of Open Source Software*, Volume 8, Number 90, 2023, p. 5563,
  https://doi.org/10.21105/joss.05563
