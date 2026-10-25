```@meta
CurrentModule = SMLMDriftCorrection
```

# Theory and Workflow

## Background

Sample drift is a fundamental challenge in single molecule localization microscopy (SMLM). During acquisition, thermal fluctuations, mechanical settling, and other perturbations cause the sample to move relative to the optical system. Because super-resolution images are constructed from thousands of individual localizations accumulated over minutes to hours, even nanometer-scale drift degrades the final resolution.

This package implements a **fiducial-free** drift correction algorithm -- no reference markers are needed. Instead, the algorithm exploits the statistical redundancy inherent in SMLM data: the same fluorophores blink multiple times across different frames, creating repeated observations of fixed structures. By finding the drift trajectory that produces the tightest spatial clustering of these repeated observations, the drift can be estimated and corrected.

## Drift Model

### Algorithmic Framework

This package is based on [Wester et al. (2021)](https://doi.org/10.1038/s41598-021-02850-7), which introduced a two-phase parametric approach to fiducial-free drift correction:

1. **Intra-dataset correction**: Drift within each acquisition segment is modeled as a polynomial function of frame number (a proxy for time), with no constant term since global offsets are handled separately.

2. **Inter-dataset correction**: Constant lateral shifts between datasets account for registration errors or repositioning between acquisition segments.

This decomposition separates two physically distinct drift sources: continuous thermal/mechanical drift during acquisition (intra) and discrete repositioning errors between acquisitions (inter).

The current implementation makes two key updates to the original Wester et al. algorithm:

### Update 1: Entropy Cost Function

Wester et al. used a **saturated nearest-neighbor distance** cost function:

```math
C(\theta) = \sum_{i=1}^{N} \min(d_i, \ell)
```

where ``d_i`` is the nearest-neighbor distance for localization ``i`` after drift correction and ``\ell`` is a saturation threshold. The saturation prevents distant pairs (from different emitters) from dominating the cost. While effective, this cost function does not account for the varying localization precisions (``\sigma``) of individual localizations.

We adopted the **entropy minimization** cost function of [Cnossen et al. (2021)](https://doi.org/10.1364/OE.426620), which models the SMLM reconstruction as a Gaussian mixture:

```math
p(\mathbf{r}) = \frac{1}{N} \sum_{i=1}^{N} \mathcal{N}(\mathbf{r};\, \boldsymbol{\mu}_i - \mathbf{d}(t_i),\, \boldsymbol{\Sigma}_i)
```

where ``\boldsymbol{\mu}_i`` is the measured position, ``\boldsymbol{\Sigma}_i`` is the diagonal covariance from localization uncertainty, and ``\mathbf{d}(t_i)`` is the drift at frame ``t_i``.

Since the entropy of a Gaussian mixture has no closed form, we optimize a **variational upper bound**. The implementation reports the objective with a summed component-entropy term and an averaged neighbor term:

```math
H_{\text{ub}}(\mathbf{D}) = \sum_i H_i - \frac{1}{N}\sum_i \log\!\Bigl(\frac{1}{N}\sum_{j \neq i} e^{-D_{\text{KL}}(p_i \| p_j)}\Bigr)
```

The summed first term makes `info.entropy` scale with the number of localizations; divide by `length(smld.emitters)` (``N``) for a size-independent diagnostic. Each ``H_i`` depends only on localization uncertainties and is constant during optimization. The second term depends on pairwise KL divergences between Gaussian localizations:

```math
D_{\text{KL}}(i, j) = \frac{1}{2}\sum_{k=1}^{K} \left[\log\frac{\sigma_{j,k}^2}{\sigma_{i,k}^2} + \frac{\sigma_{i,k}^2}{\sigma_{j,k}^2} + \frac{(\mu_{i,k} - \mu_{j,k})^2}{\sigma_{j,k}^2} - 1\right]
```

The drift parameters enter through the corrected positions ``\mu_{i,k}``. When drift is correctly removed, localizations from the same emitter cluster tightly, the KL divergences shrink, and the entropy decreases.

For computational efficiency, the pairwise sum is truncated to the **k nearest neighbors** of each localization (default ``k = 100``), computed via KDTree. The KDTree is rebuilt adaptively -- only when the drift estimate changes by more than 100 nm -- avoiding ``O(N \log N)`` rebuilds on every optimizer iteration.

### Update 2: Legendre Polynomial Basis

Wester et al. used standard monomials (``f, f^2, \ldots``) for the drift polynomial. We use **Legendre polynomials** (``P_1(t), P_2(t), \ldots``) evaluated on normalized time ``t \in [-1, 1]``:

```math
t = \frac{2(f - 1)}{n_{\text{frames}} - 1} - 1
```

The orthogonality of the Legendre basis provides better optimization conditioning, especially for higher polynomial degrees. Each coefficient captures independent variation, preventing the numerical ill-conditioning that arises with standard polynomial bases at high degree.

As in Wester et al., the constant term (``P_0``) is excluded from the intra-dataset model. Global offsets are handled by the inter-dataset shifts.

## Dataset Modes

The package supports three acquisition scenarios through the `dataset_mode` parameter and chunking options. `dataset_mode` selects the inter-dataset algorithm: registered mode uses CC-primary alignment against the consensus, while continuous mode uses consecutive-chunk CC. `quality=:fft` ignores `dataset_mode` and uses registered-style CC passes. Continuation, `driftcorrect(smld, info)`, reuses the original run's `dataset_mode` and chunking.

### Continuous (Single Polynomial)

For short acquisitions (fewer than ~4000 frames), drift is modeled as a single polynomial over the entire dataset:

```julia
config = DriftConfig(dataset_mode=:continuous, degree=3)
(smld_corrected, info) = driftcorrect(smld, config)
```

This fits one ``n``-th order Legendre polynomial per spatial dimension to capture the smooth, continuous drift trajectory. Best when the drift is well-described by a low-order polynomial over the full acquisition.

### Continuous (Piecewise Chunked)

For long acquisitions, a single polynomial may not capture complex drift patterns or may become numerically unstable at high degree. The data is arbitrarily split into temporal chunks, each fit with its own polynomial:

```julia
config = DriftConfig(dataset_mode=:continuous, chunk_frames=4000)
(smld_corrected, info) = driftcorrect(smld, config)
```

Each chunk is treated as a separate "dataset" internally. The inter-chunk shift at each boundary is **measured by cross-correlating consecutive chunks** rather than extrapolated from polynomial endpoints -- consecutive chunks image the same structure (same field of view, adjacent in time), so cross-correlation locks on reliably. The endpoint-chained prediction (``\text{inter}[n] = \text{inter}[n-1] + \text{endpoint}(n-1) - \text{startpoint}(n)``) is kept only as a per-boundary fallback for a sparse or featureless boundary where cross-correlation cannot, chosen by a consensus-overlap test, and the result is entropy-refined with regularization from boundary-gap estimates. Measuring each boundary independently avoids the two failure modes of endpoint extrapolation: slow directional residual accumulation across many seams, and a single mis-fit boundary (a polynomial extrapolating poorly at ``t = \pm 1``) propagating downstream through the cumulative chain. This piecewise approach provides the flexibility of high-order modeling without requiring a single high-degree polynomial over the full acquisition.

### Registered

For instruments that periodically register the sample to a reference position between acquisition segments -- such as the Sequential Super-resolution Microscope (SeqSRM) described in [Schodt et al. (2023)](https://doi.org/10.1364/BOE.477501) -- the datasets are independent acquisitions with bounded drift:

```julia
config = DriftConfig(dataset_mode=:registered)
(smld_corrected, info) = driftcorrect(smld, config)
```

Between acquisition segments, the microscope acquires a brightfield z-stack, computes 3D cross-correlation against a reference, and iteratively moves the stage to realign the sample. This bounds the inter-dataset drift to the registration precision (typically 5-10 nm lateral). However, residual registration errors and intra-segment drift still require computational correction.

In registered mode, datasets are spatially overlapping images of the same field of view. The inter-dataset alignment is **CC-primary**. For each dataset being aligned, a cross-correlation against the corrected consensus of all *other* datasets gives a shift estimate (the **seed**). Histogram bins are capped at 4 localizations per 50 nm bin so a bright aggregate localized thousands of times in one bin cannot capture the correlation peak. The seed is the cross-correlation shift unless the current shift overlaps strictly better. It is then **refined** by minimizing the merged-cloud entropy of that dataset combined with the consensus, incorporating localization uncertainties ``\sigma`` for sub-bin precision.

An **overlap arbiter** chooses among {refined, cross-correlation, current}. The largest *occupancy* picks the basin: localizations within 30 nm of a consensus localization count as inliers, weighted by inverse local density so a compact site localized many times counts about once. Occupancy ties prefer refined, then cross-correlation, then current. Among candidates within 30 nm of that winner that it does not lead significantly in occupied sites, the lowest merged-cloud entropy cost wins. Equal costs prefer current, then refined, then cross-correlation **among those eligible candidates**. Cross-correlation supplies the global signal because merged-cloud entropy alone is only about ``1/N`` sensitive to a single dataset's offset (here ``N`` is the number of datasets). The arbiter provides a backstop against a captured CC peak or a refinement into a wrong basin; the near-tie rule can accept a small occupancy decrease.

With **two datasets**, only the second is aligned, against the first; the first is held fixed. With **three or more**, every dataset including the reference is aligned against the others. In the small-offset regime, each dataset's refine measures its offset from the mean of the others, which move in the same pass, so a simultaneous update overcounts the relative correction by ``N/(N-1)``. A move within 30 nm of the incumbent is therefore damped by ``(N-1)/N`` to remove that overcount; larger moves into another basin are applied in full. The result is then re-anchored to keep the reference shift at zero, preserving the global frame. This damping applies to the registered CC-primary passes; the cold start, `:fft`, and continuous mode use their own update rules.

No tier, `:fft` included, applies a cross-correlation shift with a component over 5 μm. Registered `:singlepass` and `:iterative`, and `:fft` regardless of `dataset_mode`, warn once per dataset per call when a dropped CC overlaps clearly better than the shift the dataset ends with, indicating a likely collection failure (the stage moved between acquisitions); the warning leaves the kept shift unchanged. Continuous `:singlepass` and `:iterative` drop such a boundary CC and fall back to endpoint chaining at that boundary without a warning.

## Quality Tiers

The package provides three quality tiers that trade speed for accuracy. All three share the same drift model (Legendre polynomials + inter-shifts); they differ in how the model parameters are estimated.

!!! tip "Multi-threading"
    Intra-dataset correction is parallelized with `Threads.@threads` (each dataset is independent). Registered inter-dataset alignment is also threaded (each dataset's CC seed and entropy refine run against a precomputed snapshot of corrected coordinates). Continuous CC seeding walks the chunks sequentially; its entropy refine is threaded against chunk 1, then sequential against earlier chunks. Start Julia with multiple threads for best performance:
    ```
    julia -t auto        # use all available cores
    julia -t 8           # use 8 threads
    ```

### FFT (`:fft`)

The fastest tier. Uses **cross-correlation** of histogram images to estimate inter-dataset shifts. No intra-dataset polynomial fitting is performed.

**Procedure:**
1. Build 2D (or 3D) histogram images from each dataset's localizations
2. **Pass 1**: Compute cross-correlation of each dataset against dataset 1 via FFT; extract shift from the correlation peak with sub-pixel Gaussian refinement
3. **Pass 2**: Refine each dataset's shift against a merged histogram of all other (shifted) datasets
4. **Pass 3**: Detect outlier shifts (more than `median + 5*max(MAD, 100 nm)`, with at least 4 datasets) and re-align them against dataset 1 using a Gaussian-damped cross-correlation prior

Every CC shift, in all three passes, is computed on bin-count-capped histograms. Subject to the 5 μm component limit, it replaces the current shift unless the current shift has strictly higher occupancy near the reference (ties go to the CC). A CC with a component over 5 μm is not applied; when a dropped CC overlaps clearly better than the shift the dataset ends with, `:fft` warns once per dataset per call of a likely collection failure (the stage moved between acquisitions). `:fft` ignores `dataset_mode`: `:fft` with `:continuous` runs the same registered-style passes against dataset 1 and the merged others.

Best for: quick previews, very large datasets, or as initialization for entropy-based methods.

### Singlepass (`:singlepass`)

The default tier. Performs one pass of entropy-based intra-dataset correction followed by inter-dataset alignment.

**Procedure:**
1. **Intra-dataset correction** (threaded across datasets):
   - Initialize polynomial coefficients with small random values
   - Minimize entropy upper bound using Nelder-Mead optimization (10,000 iteration limit)
   - Adaptive KDTree rebuilding avoids unnecessary recomputation
2. **Inter-dataset alignment** -- mode-specific:
   - **Registered**: CC-primary, as described under Dataset Modes: a CC cold start against dataset 1 followed by two passes of CC seeding, BFGS entropy refinement, and overlap arbitration against the corrected consensus. With two datasets only the second is aligned; with three or more, all datasets including the reference are aligned, with small moves damped as described above.
   - **Continuous**: CC-seeded per boundary. The boundary cross-correlations walk the chunks **sequentially** (each placement builds on the previous chunk's), falling back to endpoint-chaining only where CC cannot lock on; the entropy refine that follows (with boundary-gap regularization) is threaded against chunk 1, then sequential against earlier chunks.
3. Apply corrections to produce the final SMLD

### Iterative (`:iterative`)

The most accurate tier. Iterates between intra and inter correction until convergence.

**Procedure:**
1. Run the full singlepass procedure as initialization (this establishes the CC-seeded inter shifts -- continuous benefits from the CC seed here too)
2. **Iterate until convergence** (default: 10 iterations per attempt, tolerance 1 nm):
   a. Re-run intra-dataset correction with inter-shifts applied (shifted coordinates), threaded across datasets
   b. Re-run inter-dataset alignment over a corrected snapshot -- registered: one CC-primary pass; continuous: refine the inter shifts in place (the boundary cross-correlation is *not* re-chained every iteration, which would otherwise overwrite the refined shifts and stall convergence)
   c. Check convergence: the maximum change across **both** inter-shift and intra-drift components (the latter sampled at frame start/mid/end) falls below `convergence_tol`. In **both** modes an additional early exit fires when the merged-cloud entropy cost plateaus (relative improvement below 1e-4) -- the entropy is nearly flat with respect to the inter shifts there, so a movement-only test could otherwise run to the iteration cap at ~0% gain.
   d. Each attempt runs at most `max_iterations` iterations; a run that has not converged is retried from its current model up to 3 times, so `info.iterations` can exceed `max_iterations`.
3. Track entropy history for diagnostics (`info.history`: the entropy after the singlepass start and after each iteration; empty for `:fft` and `:singlepass`)

The iteration resolves the coupling between intra and inter corrections: the optimal polynomial depends on the inter-shifts, and vice versa. For data with significant inter-dataset drift, iterative mode can improve accuracy substantially over singlepass.

## Known limits

- Dilution: unmatched localizations within 30 nm of matched ones lower the correct shift's occupancy, and on a hand-placed arrangement `:fft` can then accept a wrong basin.
- A reference dataset made mostly of structure the other dataset lacks can prefer a wrong basin in registered mode.
- On the extended iterative fixture the sub-pixel CC fit costs about 8.7 nm mean RMSD over 16 seeds against 0.2.9 (not significant); real iterative runs show no loss.

## References

- Cnossen J, Cui TJ, Joo C, Smith C. "Drift correction in localization microscopy using entropy minimization." *Optics Express* 29(18):27961-27974, 2021. [DOI: 10.1364/OE.426620](https://doi.org/10.1364/OE.426620)

- Wester MJ, Schodt DJ, Mazloom-Farsibaf H, Fazel M, Pallikkuth S, Lidke KA. "Robust, fiducial-free drift correction for super-resolution imaging." *Scientific Reports* 11:23672, 2021. [DOI: 10.1038/s41598-021-02850-7](https://doi.org/10.1038/s41598-021-02850-7)

- Schodt DJ, Farzam F, Liu S, Lidke KA. "Automated multi-target super-resolution microscopy with trust regions." *Biomedical Optics Express* 14(1):429-440, 2023. [DOI: 10.1364/BOE.477501](https://doi.org/10.1364/BOE.477501)
