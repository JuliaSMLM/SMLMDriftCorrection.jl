```@meta
CurrentModule = SMLMDriftCorrection
```

# SMLMDriftCorrection.jl

Fiducial-free drift correction for Single Molecule Localization Microscopy (SMLM).

## Overview

SMLMDriftCorrection.jl implements entropy-based drift correction using Legendre polynomial models. The algorithm corrects both:

- **Intra-dataset drift**: Drift within each dataset (movie segment) modeled as a polynomial over time
- **Inter-dataset drift**: Constant shifts between datasets to align them

All distance units are in **micrometers (μm)**.

## Installation

```julia
using Pkg
Pkg.add("SMLMDriftCorrection")
```

## Quick Start

```julia
using SMLMDriftCorrection

# Basic drift correction
result = driftcorrect(smld)
smld_corrected = result.smld

# Or destructure directly
(; smld, model) = driftcorrect(smld)

# Extract drift trajectory for plotting
traj = drift_trajectory(model)
# traj.frames, traj.x, traj.y ready for plotting
```

## Usage Examples

### With Simulated Data

```julia
using SMLMSim
using SMLMDriftCorrection
DC = SMLMDriftCorrection

# Simulate data
smld_true, smld_model, smld_noisy = simulate(;
    ρ=1.0, σ_psf=0.13, minphotons=50,
    ndatasets=10, nframes=1000, framerate=50.0,
    pattern=Nmer2D(n=6, d=0.2),
    molecule=GenericFluor(; q=[0 50; 1e-2 0]),
    camera=IdealCamera(1:256, 1:256, 0.1)
)

# Create and apply synthetic drift
drift_true = DC.LegendrePolynomial(smld_noisy; degree=2, initialize="random", rscale=0.1)
smld_drifted = DC.applydrift(smld_noisy, drift_true)

# Correct drift
result = driftcorrect(smld_drifted; verbose=1)
```

### With SMITE Data

```julia
using SMLMData
using SMLMDriftCorrection

smd = SmiteSMD(path, file)   # *_Results.mat file
smld = load_smite_2d(smd)
result = driftcorrect(smld; verbose=1)
```

### Filtering to ROI

```julia
x = [e.x for e in smld.emitters]
y = [e.y for e in smld.emitters]
mask = (x .> 64.0) .& (x .< 128.0) .& (y .> 64.0) .& (y .< 128.0)
smld_roi = filter_emitters(smld, mask)

result = driftcorrect(smld_roi)
```

### Continuous Acquisition Mode

For data where drift accumulates across files (one long acquisition split into multiple datasets):

```julia
# Continuous mode - drift chains across datasets
result = driftcorrect(smld; dataset_mode=:continuous)

# With chunking for finer-grained correction
result = driftcorrect(smld; dataset_mode=:continuous, n_chunks=10)
```

## Dataset Modes

- **`:registered`** (default): Datasets are independent acquisitions with stage registration between them. Each dataset aligns to dataset 1.

- **`:continuous`**: One long acquisition split into multiple files. Drift accumulates across datasets, and inter-dataset shifts chain sequentially.

## Algorithm

The algorithm uses **entropy minimization** (Cnossen et al., 2021) with **Legendre polynomial** basis functions:

1. **Intra-dataset**: For each dataset, fit a Legendre polynomial (degree 2 by default) to model drift over time. Uses KL divergence-based entropy as cost function with adaptive KDTree neighbor rebuilding.

2. **Inter-dataset**: Align datasets using constant shifts optimized via entropy minimization.

The Legendre polynomial basis provides better optimization conditioning than standard polynomials because the basis functions are orthogonal over the normalized time domain [-1, 1].

## References

- Cnossen J, et al. "Drift correction in localization microscopy using entropy minimization", *Optics Express* 29(18), 2021. [DOI: 10.1364/OE.426620](https://doi.org/10.1364/OE.426620)

- Wester MJ, et al. "Robust, fiducial-free drift correction for super-resolution imaging", *Scientific Reports* 11, 2021. [DOI: 10.1038/s41598-021-02850-7](https://doi.org/10.1038/s41598-021-02850-7)

## API Reference

### Main Interface

```@docs
driftcorrect
filter_emitters
drift_trajectory
```

### Drift Models

```@docs
LegendrePolynomial
IntraLegendre
LegendrePoly1D
InterShift
```

### Drift Application

```@docs
applydrift
correctdrift
```

### Entropy Functions

```@docs
entropy_HD
ub_entropy
```

### Cross-Correlation

```@docs
findshift
histimage2D
crosscorr2D
```

### Index

```@index
```
