# SMLMDriftCorrection

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSMLM.github.io/SMLMDriftCorrection.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSMLM.github.io/SMLMDriftCorrection.jl/dev)
[![Build Status](https://github.com/JuliaSMLM/SMLMDriftCorrection.jl/workflows/CI/badge.svg)](https://github.com/JuliaSMLM/SMLMDriftCorrection.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaSMLM/SMLMDriftCorrection.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSMLM/SMLMDriftCorrection.jl)

## Overview

SMLMDriftCorrection is a Julia package that implements algorithms for
correcting sample drift in SMLM (Single Molecule Localization Microscopy) data.
The algorithm consist of an intra-dataset portion and an inter-dataset portion.
The main entry to this package is driftcorrect in src/interfaces.jl .

## Installation
```julia
using Pkg
Pkg.add("SMSLDriftCorrection")
```

## Basic Example
```julia
using SMLMData
using SMLMDriftCorrection
DC = SMLMDriftCorrection

# Generate or load localization data (smld).
# Here, start with a basic simulation with default parameters.
cam = IdealCamera(1:128, 1:128, 0.1)  # 128×128 pixels, 100nm pixels
smld_true, smld_model, smld_noisy = simulate(
    camera=cam
)
smld = smld_noisy
# Count the number of localizations.
N = length(smld.emitters)
# Create a degree 2 polynomial model for intra-dataset drift correction with
# normalized random coefficients.
driftmodel = DC.Polynomial(smld; degree=2, initialize="random")
# Apply the drift model to the localizations with random adjustments.
smld_drift = DC.applydrift(smld, driftmodel)
# Perform drift correction with default parameters.
smld_DC = DC.driftcorrect(smld_drift)
# Extract emitter localization coordinates.
smld_x = [e.x for e in smld.emitters]
smld_y = [e.y for e in smld.emitters]
smld_DC_x = [e.x for e in smld_DC.emitters]
smld_DC_y = [e.y for e in smld_DC.emitters]
# Compute the RMSD (root mean square deviation) between the original and drift
# corrected coordinates.
rmsd = sqrt(sum((smld_DC_x .- smld_x).^2 .+ (smld_DC_y .- smld_y).^2) ./ N)
```
## Cost Functions

- **Kdtree**: Michael J. Wester et al (2021): "Robust, fiducial-free drift
  correction for super-resolution imaging"; sum of the scaled negative
  exponentials of nearest neighbor distances for each localization;
  see costfuns.jl
- **Entropy**: Jelmer Cnossen et al (2021): "Drift correction in localization
  microscopy using entropy minimization"; upper bound on the statistical
  entropy of a Gaussian Mixture Model characterizing the sum of the
  localization probability distributions; see costfuns.jl and cost_entropy.jl

Compare using examples/finddrift.jl, which has the basic calls to set up
the drift model and then correct it using the default Kdtree cost function,
where smd_noisy is gnerated by SMLMSim.

```
## driftcorrect interface function

**driftcorrect**(***smld***::SMLMData.SMLD;  
  ***intramodel***::String = "Polynomial",  
  ***cost_fun***::String = "Kdtree",  
  ***degree***::Int = 2,  
  ***d_cutoff***::AbstractFloat = 0.1,  
  ***maxn***::Int = 200,
  ***histbinsize***::AbstractFloat = -1.0, 
  ***verbose***::Int = 0)

### INPUT
- ***smld***:        structure containing (X, Y) or (X, Y, Z) localization
                     coordinates (μm)
### Optional keyword INPUTs
- ***intramodel***:  model for intra-dataset DC: {"Polynomial", "LegendrePoly"} = "Polynomial"
- ***cost_fun***:    intra/inter cost function: {"Kdtree", "Entropy"} = "Kdtree"
- *** cost_fun_intra***: intra cost function override: ""
- *** cost_fun_inter***: inter cost function override: ""
- ***degree***:      degree for polynomial intra-dataset DC = 2
- ***d_cutoff***:    distance cutoff (μm) = 0.01 (Kdtree cost function)
- ***maxn***:        maximum number of neighbors considered = 200 (Entropy cost
                     function)
- ***histbinsize***: histogram bin size for inter-datset cross-correlation
                     correction (μm) = -1.0 [< 0 means no correction]
- ***verbose***:     flag for more output = 0
### OUTPUT
- ***smd_found***:   structure containing drift corrected coordinates (μm)

## Other entry points into SMLMDriftCorrect
- Polynomial
- applydrift
- ub_entropy is an upper bound on the entropy based on nearest neighbors
- entropy_HD is the entropy summed over all/nearest neighbor localizations
