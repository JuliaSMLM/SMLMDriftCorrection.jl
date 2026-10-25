```@meta
CurrentModule = SMLMDriftCorrection
```

# SMLMDriftCorrection.jl

Fiducial-free drift correction for Single Molecule Localization Microscopy (SMLM).

## Overview

SMLMDriftCorrection.jl implements entropy-based drift correction using Legendre polynomial models. The package provides:

- **Intra-dataset drift correction**: Drift within each dataset (movie segment) modeled as a polynomial over time
- **Inter-dataset drift correction**: Constant shifts between datasets to align them
- **SMLD alignment** (`align_smld`): Shift alignment in 2D or 3D, or affine alignment in 2D, of independent SMLD structures (e.g., multi-color channels)

All distance units are in **micrometers (μm)**.

## Installation

```julia
using Pkg
Pkg.add("SMLMDriftCorrection")
```

## Quick Start

```@example quickstart
using SMLMSim
using SMLMDriftCorrection
using SMLMRender
DC = SMLMDriftCorrection
using Random; Random.seed!(42)  # hide

# Simulate blinking data (hexamers, 2 datasets)
params = StaticSMLMConfig(7.5, 0.13, 50, 2, 1000, 50.0, 2, [0.0, 1.0])
(smld_noisy, _) = simulate(params;
    pattern=Nmer2D(n=6, d=0.2),
    molecule=GenericFluor(; photons=35000.0, k_on=0.25, k_off=50.0),
    camera=IdealCamera(1:32, 1:32, 0.1)
)

# Create synthetic drift (~200nm diagonal) and apply it
drift_true = DC.LegendrePolynomial(smld_noisy; degree=2, initialize="zeros")
for ds in 1:smld_noisy.n_datasets
    drift_true.intra[ds].dm[1].coefficients .= [0.10, 0.03]   # x drift
    drift_true.intra[ds].dm[2].coefficients .= [0.08, -0.02]   # y drift
end
drift_true.inter[2].dm .= [0.05, -0.03]
smld_drifted = DC.applydrift(smld_noisy, drift_true)

# Correct drift
(smld_corrected, info) = driftcorrect(smld_drifted)

# Results
N = length(smld_noisy.emitters)
rmsd = sqrt(sum(
    ([e.x for e in smld_corrected.emitters] .- [e.x for e in smld_noisy.emitters]).^2 .+
    ([e.y for e in smld_corrected.emitters] .- [e.y for e in smld_noisy.emitters]).^2
) / N)
println("Localizations: ", N)
println("RMSD: ", round(rmsd * 1000, digits=1), " nm")
println("Elapsed: ", round(info.elapsed_s, digits=1), " s")
```

### Before and After

Histogram renders colored by frame number show drift smearing (before) and recovery (after):

**Drifted** — localizations smeared by ~200 nm of drift:

```@example quickstart
(img_drifted, _) = render(smld_drifted, strategy=HistogramRender(), color_by=:frame, colormap=:turbo, zoom=10)
save_image("drifted.png", img_drifted)
nothing  # hide
```

![Drifted](drifted.png)

**Corrected** — hexamer structure recovered (RMSD below 1 nm):

```@example quickstart
(img_corrected, _) = render(smld_corrected, strategy=HistogramRender(), color_by=:frame, colormap=:turbo, zoom=10)
save_image("corrected.png", img_corrected)
nothing  # hide
```

![Corrected](corrected.png)

## Usage Examples

### With Simulated Data

```julia
using SMLMSim
using SMLMDriftCorrection
DC = SMLMDriftCorrection

# Simulate data
params = StaticSMLMConfig(10.0, 0.13, 30, 3, 1000, 50.0, 2, [0.0, 1.0])
(smld_noisy, _) = simulate(params;
    pattern=Nmer2D(n=6, d=0.2),
    molecule=GenericFluor(; photons=5000.0, k_on=0.02, k_off=50.0),
    camera=IdealCamera(1:64, 1:64, 0.1)
)

# Create and apply synthetic drift
drift_true = DC.LegendrePolynomial(smld_noisy; degree=2, initialize="random", rscale=0.1)
smld_drifted = DC.applydrift(smld_noisy, drift_true)

# Correct drift
config = DriftConfig(verbose=1)
(smld_corrected, info) = driftcorrect(smld_drifted, config)
```

### With SMITE Data

```julia
using SMLMData
using SMLMDriftCorrection

smd = SmiteSMD(path, file)   # *_Results.mat file
smld = load_smite_2d(smd)
(smld_corrected, info) = driftcorrect(smld, DriftConfig(verbose=1))
```

### Filtering to ROI

```julia
x = [e.x for e in smld.emitters]
y = [e.y for e in smld.emitters]
# thresholds are in μm (here a 6.4-12.8 μm square)
mask = (x .> 6.4) .& (x .< 12.8) .& (y .> 6.4) .& (y .< 12.8)
smld_roi = filter_emitters(smld, mask)

(smld_corrected, info) = driftcorrect(smld_roi, DriftConfig())
```

### Continuous Acquisition Mode

For data where drift accumulates across files (one long acquisition split into multiple datasets):

```julia
# Continuous mode - one acquisition split across datasets
config = DriftConfig(dataset_mode=:continuous)
(smld_corrected, info) = driftcorrect(smld, config)

# With chunking for finer-grained correction
config = DriftConfig(dataset_mode=:continuous, chunk_frames=4000)
(smld_corrected, info) = driftcorrect(smld, config)
```

### Aligning Independent SMLDs

For aligning separate acquisitions or multi-color channels:

```julia
using SMLMDriftCorrection

(aligned, info) = align_smld([smld_ch1, smld_ch2]; method=:entropy)
info.shifts  # [zeros(2), [dx, dy]]
```

Use `transform=:affine` for a 2D affine correction. `info.transforms` then holds similarity approximations; `info.diagnostic` holds the exact affine coefficients and global shifts. See [`align_smld`](@ref) for requirements.

### Residual Drift Diagnostics

`driftcorrect` returns position-frame correlation diagnostics in `info.residual_correlation`, with per-dataset summaries and an inter-dataset check. Use [`position_frame_correlation`](@ref) to compute these diagnostics directly on any SMLD:

```julia
diag = position_frame_correlation(smld_corrected; K=20, mode=:intra)
diag.summary.mean_abs_corr_x
```

### Warm Start / Continuation

```julia
# Warm start from previous model
config1 = DriftConfig(degree=2)
(smld1, info1) = driftcorrect(smld, config1)

config2 = DriftConfig(warm_start=info1.model)
(smld2, info2) = driftcorrect(smld2, config2)

# Continue iterating from previous result
(smld3, info3) = driftcorrect(smld, info1; max_iterations=5)
```

Pass the original, uncorrected `smld` with its `info1`. Continuation reuses the original run's `dataset_mode` and chunking, so it also works on chunked continuous results.

## Dataset Modes

The modes use different inter-dataset algorithms; see Dataset Modes in [Theory and Workflow](@ref "Theory and Workflow"). `quality=:fft` ignores `dataset_mode`.

- **`:registered`** (default): datasets are independent acquisitions (CC-primary inter alignment).
- **`:continuous`**: one long acquisition split into files or chunks (consecutive-chunk CC). Use `drift_trajectory(info.model; cumulative=true)` for plotting accumulated drift.

## Algorithm

The algorithm uses **entropy minimization** (Cnossen et al., 2021) with **Legendre polynomial** basis functions:

1. **Intra-dataset**: For each dataset, fit a Legendre polynomial (degree 2 by default) to model drift over time. Uses KL divergence-based entropy as cost function with adaptive KDTree neighbor rebuilding. Threaded across datasets.

2. **Inter-dataset**: Align datasets using constant shifts. Registered mode uses a cross-correlation seed, entropy refinement, and an overlap arbiter. With two datasets only the second is aligned, against the first. With three or more, every dataset including the reference is aligned against the others; a move within 30 nm of the current shift is damped by (N-1)/N (N = number of datasets) to remove the overcount from updating all datasets at once. Larger moves into another basin are applied in full. Continuous mode aligns consecutive-chunk boundaries by cross-correlation, with endpoint-chaining as a per-boundary fallback.

3. **Iterative** (`:iterative` quality only): Repeat intra↔inter until the inter and intra parameter changes fall below `convergence_tol` or the entropy cost plateaus (relative improvement < 1e-4, both modes).

No tier, `:fft` included, applies a cross-correlation shift with a component over 5 μm. Registered `:singlepass` and `:iterative`, and `:fft` regardless of `dataset_mode`, warn once per dataset per call when a dropped CC overlaps clearly better than the shift the dataset ends with, indicating a likely collection failure (the stage moved between acquisitions). Continuous `:singlepass` and `:iterative` drop such a boundary CC and fall back to endpoint chaining at that boundary without a warning. See [Theory and Workflow](@ref "Theory and Workflow") for the alignment rules.

The Legendre polynomial basis provides better optimization conditioning than standard polynomials because the basis functions are orthogonal over the normalized time domain [-1, 1].

!!! tip "Threading"
    Intra-dataset correction is parallelized with `Threads.@threads` (each dataset is independent). Registered inter-dataset alignment is also threaded (each dataset's CC seed and entropy refine run against a snapshot of corrected coordinates); continuous CC seeding walks the chunks sequentially and its refine is threaded, then sequential. Start Julia with multiple threads (e.g., `julia -t auto`) for best performance on multi-dataset data.

## References

- Cnossen J, et al. "Drift correction in localization microscopy using entropy minimization", *Optics Express* 29(18), 2021. [DOI: 10.1364/OE.426620](https://doi.org/10.1364/OE.426620)

- Wester MJ, et al. "Robust, fiducial-free drift correction for super-resolution imaging", *Scientific Reports* 11, 2021. [DOI: 10.1038/s41598-021-02850-7](https://doi.org/10.1038/s41598-021-02850-7)

## Documentation

- [Configuration](@ref) - Full description of `DriftConfig` input and `DriftInfo` output
- [Theory & Workflow](@ref "Theory and Workflow") - Algorithm background, dataset modes, and quality tiers
- [API Reference](@ref) - Complete API documentation
