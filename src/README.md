# SMLMDriftCorrection

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSMLM.github.io/SMLMDriftCorrection.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSMLM.github.io/SMLMDriftCorrection.jl/dev)
[![Build Status](https://github.com/JuliaSMLM/SMLMDriftCorrection.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaSMLM/SMLMDriftCorrection.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/JuliaSMLM/SMLMDriftCorrection.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSMLM/SMLMDriftCorrection.jl)

## Overview

Drift correction.  The main algorithm (*driftcorrect*) consists of an
intra-dataset portion and an inter-dataset portion.  The drift corrected
coordinates are returned as output.  All distance units are in μm.

## Quick Start

```julia
using SMLMSim
using SMLMDriftCorrection

# Make an Nmer dataset.  Simulation parameters use physical units, while smld
# structures are in units of pixels and frames.
# Original noisy data ...
smld_true, smld_model, smld_noisy = simulate(;
    ρ=1.0,                # emitters per μm²
    σ_psf=0.13,           # PSF width in μm (130nm)
    minphotons=50,        # minimum photons for detection
    ndatasets=10,         # number of independent datasets
    nframes=1000,         # frames per dataset
    framerate=50.0,       # frames per second
    pattern=Nmer2D(n=6, d=0.2),  # hexamer with 200nm diameter
    molecule=GenericFluor(; q=[0 50; 1e-2 0]),  # rates in 1/s
    camera=IdealCamera(1:256, 1:256, 0.1)  # pixelsize in μm
)

## Set up the drift model
drift_true = Polynomial(smld_noisy; degree=2, initialize="random", rscale=0.1)

# Produce drifted data ...
smld_drifted = SMLMDriftCorrection.applydrift(smld_noisy, drift_true)

## Correct drift (Kdtree cost function by default)
smld_corrected = driftcorrect(smld_drifted; verbose=1)

# Compute the RMSD between the original and corrected SMLD structures
N = length(smld_noisy.emitters)
smld_noisy_x = [e.x for e in smld_noisy.emitters]
smld_noisy_y = [e.y for e in smld_noisy.emitters]
smld_corrected_x = [e.x for e in smld_corrected.emitters]
smld_corrected_y = [e.y for e in smld_corrected.emitters]
rmsd = sqrt(sum((smld_corrected_x .- smld_noisy_x).^2 .+
		(smld_corrected_y .- smld_noisy_y).^2) ./ N)
print("rmsd 2D [driftcorrect] = $rmsd\n")

```

## COmmon Workflows

## Algorithms

- Kdtree

