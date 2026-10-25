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
| `maxn` | `100` | Maximum neighbors for entropy calculation |
| `max_iterations` | `10` | Iterations per attempt for `:iterative` mode; a run that has not converged is retried from its current model up to 3 times |
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

**Dataset mode guidance:** Use `:registered` (default) when datasets are independent acquisitions of the same FOV. Use `:continuous` when data is one long acquisition split across files, with `chunk_frames=4000` for long acquisitions. The two modes run different inter-dataset algorithms; `quality=:fft` ignores `dataset_mode`.

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
| `info.entropy` | Final entropy value; scales with the number of localizations. Divide by `length(smld.emitters)` for a size-independent number |
| `info.history` | Entropy after the singlepass start and after each iteration (`:iterative` only; empty for `:fft` and `:singlepass`) |
| `info.residual_correlation` | Position-frame correlation diagnostic on the corrected data (`intra_summary`, `intra_per_dataset`, `inter`); see `position_frame_correlation` below |
| `info.roi_indices` | Indices used for ROI subset (`nothing` if `auto_roi=false`) |

## Algorithm Pipeline

1. **Intra-dataset**: For each dataset, fit Legendre polynomial drift over time via entropy minimization with adaptive KDTree neighbor rebuilding (threaded across datasets)
2. **Inter-dataset**: Align datasets using constant shifts. Registered mode uses a cross-correlation seed, entropy refinement, and an overlap arbiter. With two datasets only the second is aligned, against the first. With three or more, every dataset including the reference is aligned against the others; a move within 30 nm of the current shift is damped by (N-1)/N (N = number of datasets) to remove the overcount from updating all datasets at once. Larger moves into another basin are applied in full. Continuous mode aligns consecutive-chunk boundaries by cross-correlation, with endpoint-chaining as a per-boundary fallback.
3. **Iterative** (`:iterative` only): Repeat intra↔inter until the inter+intra parameter changes fall below `convergence_tol` or the entropy cost plateaus (relative improvement < 1e-4, both modes)

No tier, `:fft` included, applies a cross-correlation shift with a component over 5 μm. Registered `:singlepass` and `:iterative`, and `:fft` regardless of `dataset_mode`, warn once per dataset per call when a dropped CC overlaps clearly better than the shift the dataset ends with, indicating a likely collection failure (the stage moved between acquisitions). Continuous `:singlepass` and `:iterative` drop such a boundary CC and fall back to endpoint chaining at that boundary without a warning. See [Theory and Workflow](https://juliasmlm.github.io/SMLMDriftCorrection.jl/stable/theory_workflow/) for the alignment rules.

The Legendre polynomial basis provides better optimization conditioning than standard polynomials because the basis functions are orthogonal over the normalized time domain [-1, 1].

## Aligning Independent SMLDs

`align_smld` aligns a vector of independent SMLD structures to a common reference (the first) using rigid shifts (or, in 2D, an affine transform). Use case: aligning separate acquisitions or multi-color channels that image the same FOV.

```julia
using SMLMDriftCorrection

# Entropy method (CC initial guess + entropy refinement, default)
(aligned, info) = align_smld([smld_ch1, smld_ch2]; method=:entropy)
info.shifts  # [zeros(2), [dx, dy]] -- shift applied to each SMLD

# FFT method (cross-correlation only, faster)
(aligned, info) = align_smld([smld_ch1, smld_ch2]; method=:fft)

# Config struct form
config = AlignConfig(method=:entropy, maxn=100, histbinsize=0.05)
(aligned, info) = align_smld(smlds, config)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `:entropy` | `:entropy` (CC + entropy refinement) or `:fft` (CC only) |
| `transform` | `:shift` | `:shift` (2D or 3D translation) or `:affine` (2D affine correction) |
| `maxn` | `100` | Maximum neighbors for entropy calculation |
| `histbinsize` | `0.05` | Histogram bin size (μm) for cross-correlation |
| `verbose` | `0` | Verbosity (0=quiet, 1=info) |

Shift alignment works on both 2D and 3D data; `transform=:affine` is 2D only. Shift alignment is threaded across datasets. The entropy method uses regularization toward the CC result to prevent optimizer divergence. See the [`align_smld` API reference](https://juliasmlm.github.io/SMLMDriftCorrection.jl/stable/api/#SMLMDriftCorrection.align_smld) for affine requirements.

| Alignment output | Description |
|------------------|-------------|
| `aligned` | Aligned SMLDs (the first is unchanged) |
| `info.shifts` | Translation component for each SMLD |
| `info.transforms` | Shift transforms, or similarity approximations of the affine corrections |
| `info.diagnostic` | Exact affine coefficients and global shifts for `:affine`; `nothing` for `:shift` |

## Utility Functions

### drift_trajectory
```julia
traj = drift_trajectory(info.model)
traj = drift_trajectory(info.model; cumulative=true)  # for continuous mode
```
Extract drift trajectory for plotting. Returns NamedTuple with `frames`, `x`, `y` (and `z` for 3D), `dataset`.

### position_frame_correlation
```julia
diag = position_frame_correlation(smld_corrected; K=20, mode=:intra)  # or mode=:inter
```
Residual-drift diagnostic: correlation of local position residuals with frame number (`:intra`, per dataset) or dataset index (`:inter`). `driftcorrect` stores it in `info.residual_correlation`.

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
(smld2, info2) = driftcorrect(smld, info1; max_iterations=5)
```
Pass the original, uncorrected `smld` with its `info1`. Continuation reuses the original run's `dataset_mode` and chunking, so it also works on chunked continuous results.

## Algorithm References

> Cnossen, J., Cui, T.J., Joo, C. and Smith, C. "Drift correction in localization microscopy using entropy minimization." *Optics Express*, 29(18), 27961-27974, 2021. [DOI: 10.1364/OE.426620](https://doi.org/10.1364/OE.426620)

> Wester, M.J., Schodt, D.J., Mazloom-Farsibaf, H., Fazel, M., Pallikkuth, S. and Lidke, K.A. "Robust, fiducial-free drift correction for super-resolution imaging." *Scientific Reports*, 11, 23672, 2021. [DOI: 10.1038/s41598-021-02850-7](https://doi.org/10.1038/s41598-021-02850-7)

> Schodt, D.J.\*, Wester, M.J.\*, et al. "SMITE: Single Molecule Imaging Toolbox Extraordinaire (MATLAB)." *Journal of Open Source Software*, 8(90), 5563, 2023. [DOI: 10.21105/joss.05563](https://doi.org/10.21105/joss.05563) — MATLAB implementation of this algorithm

## Related Packages

- **[SMLMAnalysis.jl](https://github.com/JuliaSMLM/SMLMAnalysis.jl)** - Complete SMLM workflow (detection + fitting + frame-connection + rendering)
- **[SMLMData.jl](https://github.com/JuliaSMLM/SMLMData.jl)** - Core data types for SMLM
- **[SMLMFrameConnection.jl](https://github.com/JuliaSMLM/SMLMFrameConnection.jl)** - Frame connection (linking blinks across frames)
- **[SMLMSim.jl](https://github.com/JuliaSMLM/SMLMSim.jl)** - SMLM data simulation

## License

MIT License - see [LICENSE](LICENSE) file for details.
