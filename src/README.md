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
DC = SMLMDriftCorrection

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
drift_true = DC.Polynomial(smld_noisy; degree=2, initialize="random", rscale=0.1)

# Produce drifted data ...
smld_drifted = DC.applydrift(smld_noisy, drift_true)

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

## Common Workflows

### Generic 2D or 3D data
```julia
using SMLMDriftCorrection

smld = ...
smld_DC = driftcorrect(smld)
```

### SMITE file
```julia
using SMLMData
using SMLMDriftCorrection

smd = SmiteSMD(path, file)   # *_Results.mat file
smld2 = load_smite_2d(smd)   # To check keys, use: varnames = keys(smld2)
smld2_DC = driftcorrect(smld2; verbose = 1, cost_fun = "Kdtree")
```

### Selecting a ROI to analyze
```julia
using SMLMData
using SMLMDriftCorrection

println("N_smld2    = $(length(smld2.emitters))")
smld2_x = [e.x for e in smld2.emitters]
smld2_y = [e.y for e in smld2.emitters]
subind = (smld2_x .> 64.0) .& (smld2_x .< 128.0) .&
         (smld2_y .> 64.0) .& (smld2_y .< 128.0)
smld2roi = DC.filter_emitters(smld2, subind)
println("N_smld2roi = $(length(smld2roi.emitters))")

 smld2_DC = DC.driftcorrect(smld2roi)
```

## Interface

Main interface for drift correction (DC).  This algorithm consists of an
intra-dataset portion and an inter-dataset portion.  The drift corrected
coordinates are returned as output.  All distance units are in μm.

"""
function driftcorrect(smld::SMLD;
    intramodel::String = "Polynomial",
    cost_fun::String = "Kdtree",
    cost_fun_intra::String = "",
    cost_fun_inter::String = "",
    degree::Int = 2,
    d_cutoff::AbstractFloat = 0.01,
    maxn::Int = 200,
    histbinsize::AbstractFloat = -1.0,
    verbose::Int = 0)

# Fields
- smld:           structure containing (X, Y) or (X, Y, Z) localization
                  coordinates (μm) - see SMLMData
- intramodel:     model for intra-dataset DC:
                  {"Polynomial", "LegendrePoly"} = "Polynomial"
- cost_fun:       intra/inter cost function: {"Kdtree", "Entropy"} = "Kdtree"
- cost_fun_intra: intra cost function override: ""
- cost_fun_inter: inter cost function override: ""
- degree:         degree for polynomial intra-dataset DC = 2
- d_cutoff:       distance cutoff (μm) = 0.01 [Kdtree cost function]
- maxn:           maximum number of neighbors considered = 200
                  [Entropy cost function]
- histbinsize:    histogram bin size for inter-datset cross-correlation
                  correction (μm) = -1.0 [< 0 means no correction]
- verbose:        flag for more output = 0
# Output
- smld_found:     structure containing drift corrected coordinates (μm)
"""

## Installation
```julia
using Pkg
Pkg.add(:SMLMDriftCorrection)
```

## Algorithms

The procedure divides the problem into intra-dataset and inter-dataset drift
correction, both minimizing a cost function that is based on the predicted
emitter positions in a dataset, either with respect to itself for intra-dataset
drift correction (noting that different sets of localizations over time are
coming from the blinking fluorophores), or the first dataset for inter-dataset
drift correction. Intra-dataset drift correction is performed first, with the
results saved for the next phase. For inter-dataset drift correction, the
datasets are drift corrected in sequence against the first dataset.  Here,
"dataset" means a segment or collection of movie frames.  Several datasets make
up a full movie.

- Kdtree: cost function is simply the thresholded sum of the nearest neighbor
  distances for all the predicted emitter positions in a dataset, either with
  respect to itself for intra-dataset drift correction (noting that different
  sets of localizations over time are coming from the blinking fluorophores),
  or the first dataset for inter-dataset drift correction.  The fast nearest
  neighbor search is done using a k-dimensional tree data structure to
  partition the image.  See [Wester2021].
- Entropy; cost function (Drift at Minimum Entropy as described in
  [Cnossen2021].
- histbinsize > 0: performs cross correlation between two histogram images
  formed from dataset sum images with the specified histogram bin size.

## References

- [Cnossen2021] Jelmer Cnossen, Tao Ju Cui, Chirlmin Joo and Carlas Smith,
  "Drift correction in localization microscopy using entropy minimization",
  *Optics Express*, Volume 29, Number 18, August 30, 2021, 27961-27974,
  PMID: 34614938,
  ([https://doi.org/10.1364/OE.426620)[https://doi.org/10.1364/OE.426620]
  (DOI: 10.1364/OE.426620)
- [Wester2021] Michael J. Wester, David J. Schodt, Hanieh Mazloom-Farsibaf,
  Mohamadreza Fazel, Sandeep Pallikkuth and Keith A. Lidke, "Robust,
  fiducial-free drift correction for super-resolution imaging", *Scientific
  Reports*, Volume 11, Article 23672, December 8, 2021, 1-14,
  (https://www.nature.com/articles/s41598-021-02850-7)
  [https://www.nature.com/articles/s41598-021-02850-7]
  (DOI: 10.1038/s41598-021-02850-7).
