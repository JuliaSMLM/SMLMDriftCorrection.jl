# SMLMDriftCorrection

Algorithms for correcting sample drift in SMLM data.  Main entry is driftcorrect in src/interfaces.jl
which consists of an intra-dataset portion and an inter-dataset portion.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSMLM.github.io/SMLMDriftCorrection.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSMLM.github.io/SMLMDriftCorrection.jl/dev)
[![Build Status](https://github.com/JuliaSMLM/SMLMDriftCorrection.jl/workflows/CI/badge.svg)](https://github.com/JuliaSMLM/SMLMDriftCorrection.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaSMLM/SMLMDriftCorrection.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSMLM/SMLMDriftCorrection.jl)

## Cost Functions

- **Kdtree**: Michael J. Wester et al (2021): "Robust, fiducial-free drift correction for super-resolution imaging";
  sum of the scaled negative exponentials of nearest neighbor distances for each localization;
  see costfuns.jl
- **Entropy**: Jelmer Cnossen et al (2021): "Drift correction in localization microscopy using entropy minimization";
  upper bound on the statistical entropy of a Gaussian Mixture Model characterizing the sum of the localization probability distributions;
  see costfuns.jl and cost_entropy.jl

Compare using examples/finddrift.jl, which has the basic calls to set up
the drift model and then correct it using the default Kdtree cost function,
where smd_noisy is gnerated by SMLMSim.

```
using SMLMDriftCorrection
DC = SMLMDriftCorrection

smd_noisy = ...
drift_true = DC.Polynomial(smd_noisy; degree = 2, 
               initialize = "random", rscale = 0.1)
smd_drift = DC.applydrift(smd_noisy, drift_true)
smld_corrected = DC.driftcorrect(smd_drift)
```

## driftcorrect interface function

**driftcorrect**(***smld***::SMLMData.SMLD;  
  ***intramodel***::String = "Polynomial",  
  ***cost_fun***::String = "Kdtree",  
  ***degree***::Int = 2,  
  ***d_cutoff***::AbstractFloat = 0.1,  
  ***maxn***::Int = 200,  
  ***verbose***::Int = 0)

### INPUT
- ***smld***:       structure containing (X, Y) coordinates (pixel)
### Optional keyword INPUTs
- ***intramodel***: model for intra-dataset DC: {"Polynomial", "LegendrePoly"} = "Polynomial"
- ***cost_fun***:   intra/inter cost function: {"Kdtree", "Entropy"} = "Kdtree"
- ***degree***:     degree for polynomial intra-dataset DC = 2
- ***d_cutoff***:   distance cutoff (pixel) = 0.1
- ***maxn***:       maximum number of neighbors considered = 200
- ***verbose***:    flag for more output = 0
### OUTPUT
- ***smd_found***:  structure containing drift corrected (X, Y) coordinates (pixel)
