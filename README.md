# SMLMDriftCorrection

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSMLM.github.io/SMLMDriftCorrection.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSMLM.github.io/SMLMDriftCorrection.jl/dev)
[![Build Status](https://github.com/JuliaSMLM/SMLMDriftCorrection.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaSMLM/SMLMDriftCorrection.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/JuliaSMLM/SMLMDriftCorrection.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSMLM/SMLMDriftCorrection.jl)

Fiducial-free drift correction for single-molecule localization microscopy (SMLM): correcting both intra-dataset drift (within each acquisition) and inter-dataset drift (between acquisitions) using entropy minimization with Legendre polynomial models. Works on 2D and 3D localization data. All distance units are in μm.

## Installation

```julia
using Pkg
Pkg.add("SMLMDriftCorrection")
```

## Quick Start

```julia
using SMLMDriftCorrection

# Default configuration
(smld_corrected, info) = driftcorrect(smld, DriftConfig())

# Custom configuration
config = DriftConfig(quality=:iterative, degree=3, verbose=1)
(smld_corrected, info) = driftcorrect(smld, config)

# Extract drift trajectory for plotting
traj = drift_trajectory(info.model)
# traj.frames, traj.x, traj.y ready for plotting
```

For complete SMLM workflows (detection + fitting + frame-connection + drift correction + rendering), see [SMLMAnalysis.jl](https://github.com/JuliaSMLM/SMLMAnalysis.jl).

## Configuration

`driftcorrect(smld, config::DriftConfig)` is the primary interface. `DriftConfig` is a `@kwdef` struct:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quality` | `:singlepass` | Quality tier (`:fft`, `:singlepass`, `:iterative`) |
| `degree` | `2` | Legendre polynomial degree for intra-dataset drift |
| `dataset_mode` | `:registered` | Multi-dataset handling (`:registered` or `:continuous`) |
| `chunk_frames` | `0` | Split datasets into chunks of N frames (continuous mode) |
| `n_chunks` | `0` | Alternative: number of chunks per dataset |
| `maxn` | `200` | Maximum neighbors for entropy calculation |
| `max_iterations` | `10` | Maximum iterations for `:iterative` mode |
| `convergence_tol` | `0.001` | Convergence tolerance (μm) for `:iterative` mode |
| `warm_start` | `nothing` | Previous `info.model` for warm starting |
| `auto_roi` | `false` | Use dense ROI subset for faster estimation |
| `verbose` | `0` | Verbosity (0=quiet, 1=info, 2=debug) |

```julia
# Config struct (preferred - reusable, composable)
config = DriftConfig(quality=:iterative, degree=3, verbose=1)
(smld_corrected, info) = driftcorrect(smld, config)

# Keyword convenience form also supported
(smld_corrected, info) = driftcorrect(smld; quality=:iterative, degree=3, verbose=1)
```

**Quality tier guidance:** `:singlepass` works well for most data. Use `:fft` for fast previews (~10x faster, less accurate). Use `:iterative` when maximum accuracy is needed (iterates intra↔inter until convergence).

**Dataset mode guidance:** Use `:registered` (default) when datasets are independent acquisitions of the same FOV. Use `:continuous` when data is one long acquisition split across files, with `chunk_frames=4000` for long acquisitions.

## Output Format

`driftcorrect()` returns `(smld_corrected::SMLD, info::DriftInfo)`.

| Output | Description |
|--------|-------------|
| `smld_corrected` | Drift-corrected localizations (main output) |
| `info.model` | Fitted `LegendrePolynomial` drift model |
| `info.elapsed_s` | Wall time (seconds) |
| `info.backend` | Computation backend (`:cpu`) |
| `info.iterations` | Iterations completed (0 for `:fft`, 1 for `:singlepass`) |
| `info.converged` | Whether convergence was achieved |
| `info.entropy` | Final entropy value |
| `info.history` | Entropy per iteration (for diagnostics) |
| `info.roi_indices` | Indices used for ROI subset (`nothing` if `auto_roi=false`) |

## Algorithm Pipeline

1. **Intra-dataset**: For each dataset, fit Legendre polynomial drift over time via entropy minimization with adaptive KDTree neighbor rebuilding (threaded across datasets)
2. **Inter-dataset**: Align datasets using constant shifts optimized via merged-cloud entropy. First pass aligns all to dataset 1 (threaded), then sequential refinement against all earlier datasets
3. **Iterative** (`:iterative` only): Repeat intra↔inter until inter-shift changes < `convergence_tol`

The Legendre polynomial basis provides better optimization conditioning than standard polynomials because the basis functions are orthogonal over the normalized time domain [-1, 1].

## Example

```julia
using SMLMSim
using SMLMDriftCorrection
DC = SMLMDriftCorrection

# Simulate localization data
params = StaticSMLMConfig(10.0, 0.13, 30, 3, 1000, 50.0, 2, [0.0, 1.0])
(smld_noisy, _) = simulate(params;
    pattern=Nmer2D(n=6, d=0.2),
    molecule=GenericFluor(; photons=5000.0, k_on=0.02, k_off=50.0),
    camera=IdealCamera(1:64, 1:64, 0.1)
)

# Create and apply synthetic drift for testing
drift_true = DC.LegendrePolynomial(smld_noisy; degree=2, initialize="random", rscale=0.1)
smld_drifted = DC.applydrift(smld_noisy, drift_true)

# Correct drift
config = DriftConfig(verbose=1)
(smld_corrected, info) = driftcorrect(smld_drifted, config)

# Extract trajectory for plotting
traj = drift_trajectory(info.model)

# Compute RMSD
N = length(smld_noisy.emitters)
rmsd = sqrt(sum(
    ([e.x for e in smld_corrected.emitters] .- [e.x for e in smld_noisy.emitters]).^2 .+
    ([e.y for e in smld_corrected.emitters] .- [e.y for e in smld_noisy.emitters]).^2
) / N)
println("RMSD = $(round(rmsd * 1000, digits=1)) nm")
```

## Utility Functions

### drift_trajectory
```julia
traj = drift_trajectory(info.model)
traj = drift_trajectory(info.model; cumulative=true)  # for continuous mode
```
Extract drift trajectory for plotting. Returns NamedTuple with `frames`, `x`, `y` (and `z` for 3D), `dataset`.

### filter_emitters
```julia
smld_roi = filter_emitters(smld, mask)  # boolean mask or indices
```
Select emitters by boolean mask or index vector. Useful for ROI selection before correction.

### Warm Start / Continuation
```julia
# Warm start from previous model
config = DriftConfig(warm_start=info1.model)
(smld2, info2) = driftcorrect(smld2, config)

# Continue iterating from previous result
(smld2, info2) = driftcorrect(smld1, info1; max_iterations=5)
```

## Algorithm References

> Cnossen, J., Cui, T.J., Joo, C. and Smith, C. "Drift correction in localization microscopy using entropy minimization." *Optics Express*, 29(18), 27961-27974, 2021. [DOI: 10.1364/OE.426620](https://doi.org/10.1364/OE.426620)

> Wester, M.J., Schodt, D.J., Mazloom-Farsibaf, H., Fazel, M., Pallikkuth, S. and Lidke, K.A. "Robust, fiducial-free drift correction for super-resolution imaging." *Scientific Reports*, 11, 23672, 2021. [DOI: 10.1038/s41598-021-02850-7](https://doi.org/10.1038/s41598-021-02850-7)

> Schodt, D.J.\*, Wester, M.J.\*, et al. "SMITE: Single Molecule Imaging Toolbox Extraordinaire (MATLAB)." *Journal of Open Source Software*, 8(90), 5563, 2023. [DOI: 10.21105/joss.05563](https://doi.org/10.21105/joss.05563)

## Related Packages

- **[SMLMAnalysis.jl](https://github.com/JuliaSMLM/SMLMAnalysis.jl)** - Complete SMLM workflow (detection + fitting + frame-connection + rendering)
- **[SMLMData.jl](https://github.com/JuliaSMLM/SMLMData.jl)** - Core data types for SMLM
- **[SMLMFrameConnection.jl](https://github.com/JuliaSMLM/SMLMFrameConnection.jl)** - Frame connection (linking blinks across frames)
- **[SMLMSim.jl](https://github.com/JuliaSMLM/SMLMSim.jl)** - SMLM data simulation

## License

MIT License - see [LICENSE](LICENSE) file for details.
