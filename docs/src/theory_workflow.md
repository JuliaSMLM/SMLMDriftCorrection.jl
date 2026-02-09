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

Since the entropy of a Gaussian mixture has no closed form, we minimize a **variational upper bound**:

```math
H_{\text{ub}}(\mathbf{D}) = \frac{1}{N}\sum_i H_i - \frac{1}{N}\sum_i \log\!\Bigl(\frac{1}{N}\sum_{j \neq i} e^{-D_{\text{KL}}(p_i \| p_j)}\Bigr)
```

The first term ``H_i`` is the entropy of each individual Gaussian component (determined only by localization uncertainties, constant during optimization). The second term depends on pairwise KL divergences between Gaussian localizations:

```math
D_{\text{KL}}(i, j) = \frac{1}{2}\sum_{k=1}^{K} \left[\log\frac{\sigma_{j,k}^2}{\sigma_{i,k}^2} + \frac{\sigma_{i,k}^2}{\sigma_{j,k}^2} + \frac{(\mu_{i,k} - \mu_{j,k})^2}{\sigma_{j,k}^2} - 1\right]
```

The drift parameters enter through the corrected positions ``\mu_{i,k}``. When drift is correctly removed, localizations from the same emitter cluster tightly, the KL divergences shrink, and the entropy decreases.

For computational efficiency, the pairwise sum is truncated to the **k nearest neighbors** of each localization (default ``k = 200``), computed via KDTree. The KDTree is rebuilt adaptively -- only when the drift estimate changes by more than 100 nm -- avoiding ``O(N \log N)`` rebuilds on every optimizer iteration.

### Update 2: Legendre Polynomial Basis

Wester et al. used standard monomials (``f, f^2, \ldots``) for the drift polynomial. We use **Legendre polynomials** (``P_1(t), P_2(t), \ldots``) evaluated on normalized time ``t \in [-1, 1]``:

```math
t = \frac{2(f - 1)}{n_{\text{frames}} - 1} - 1
```

The orthogonality of the Legendre basis provides better optimization conditioning, especially for higher polynomial degrees. Each coefficient captures independent variation, preventing the numerical ill-conditioning that arises with standard polynomial bases at high degree.

As in Wester et al., the constant term (``P_0``) is excluded from the intra-dataset model. Global offsets are handled by the inter-dataset shifts.

## Dataset Modes

The package supports three acquisition scenarios through the `dataset_mode` parameter and chunking options:

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

Each chunk is treated as a separate "dataset" internally. **Warm-starting** ensures continuity: each chunk's polynomial is initialized from the endpoint of the previous chunk's fit. The inter-dataset shifts chain the chunks together, with regularization from boundary gap estimates to prevent discontinuities. This piecewise approach provides the flexibility of high-order modeling without requiring a single high-degree polynomial over the full acquisition.

### Registered

For instruments that periodically register the sample to a reference position between acquisition segments -- such as the Sequential Super-resolution Microscope (SeqSRM) described in [Schodt et al. (2023)](https://doi.org/10.1364/BOE.477501) -- the datasets are independent acquisitions with bounded drift:

```julia
config = DriftConfig(dataset_mode=:registered)
(smld_corrected, info) = driftcorrect(smld, config)
```

Between acquisition segments, the microscope acquires a brightfield z-stack, computes 3D cross-correlation against a reference, and iteratively moves the stage to realign the sample. This bounds the inter-dataset drift to the registration precision (typically 5-10 nm lateral). However, residual registration errors and intra-segment drift still require computational correction.

In registered mode, datasets are spatially overlapping images of the same field of view. The inter-dataset alignment uses **merged-cloud entropy**: the shifted dataset's localizations are combined with reference dataset localizations into a single point cloud, and the entropy of the combined cloud is minimized. This finds the constant shift that produces the tightest merged distribution.

## Quality Tiers

The package provides three quality tiers that trade speed for accuracy. All three share the same drift model (Legendre polynomials + inter-shifts); they differ in how the model parameters are estimated.

!!! tip "Multi-threading"
    Intra-dataset correction is parallelized with `Threads.@threads` (each dataset is independent). The first inter-dataset pass (all vs dataset 1) is also threaded using a precomputed snapshot of corrected coordinates. Start Julia with multiple threads for best performance:
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
4. **Pass 3**: Detect outlier shifts (>5 MAD from median) and re-align them using a Gaussian-damped cross-correlation prior

Best for: quick previews, very large datasets, or as initialization for entropy-based methods.

### Singlepass (`:singlepass`)

The default tier. Performs one pass of entropy-based intra-dataset correction followed by inter-dataset alignment.

**Procedure:**
1. **Intra-dataset correction** (threaded across datasets):
   - Initialize polynomial coefficients with small random values
   - Minimize entropy upper bound using Nelder-Mead optimization (10,000 iteration limit)
   - Adaptive KDTree rebuilding avoids unnecessary recomputation
2. **Inter-dataset alignment, Pass 1** (threaded, all vs dataset 1):
   - For each dataset ``n > 1``, apply intra-correction to dataset ``n`` and full correction to dataset 1
   - Use cross-correlation for initial shift estimate
   - Refine via BFGS optimization of merged-cloud entropy
3. **Inter-dataset alignment, Pass 2** (sequential, each vs all earlier):
   - For each dataset ``n``, re-optimize the shift against all datasets ``1, \ldots, n-1``
   - This incorporates information from the intermediate datasets
4. Apply corrections to produce the final SMLD

For continuous mode, inter-shifts are warm-started from polynomial endpoint chaining and regularized using boundary gap estimates.

### Iterative (`:iterative`)

The most accurate tier. Iterates between intra and inter correction until convergence.

**Procedure:**
1. Run the full singlepass procedure as initialization
2. **Iterate until convergence** (default: max 10 iterations, tolerance 1 nm):
   a. Re-run intra-dataset correction with inter-shifts applied (shifted coordinates), threaded across datasets
   b. Re-run inter-dataset alignment using Jacobi-style updates (snapshot corrected data, then thread all-vs-others)
   c. Check convergence: maximum change in any inter-shift component < `convergence_tol`
3. Track entropy history for diagnostics (`info.history`)

The iteration resolves the coupling between intra and inter corrections: the optimal polynomial depends on the inter-shifts, and vice versa. For data with significant inter-dataset drift, iterative mode can improve accuracy substantially over singlepass.

## References

- Cnossen J, Cui TJ, Joo C, Smith C. "Drift correction in localization microscopy using entropy minimization." *Optics Express* 29(18):27961-27974, 2021. [DOI: 10.1364/OE.426620](https://doi.org/10.1364/OE.426620)

- Wester MJ, Schodt DJ, Mazloom-Farsibaf H, Fazel M, Pallikkuth S, Lidke KA. "Robust, fiducial-free drift correction for super-resolution imaging." *Scientific Reports* 11:23672, 2021. [DOI: 10.1038/s41598-021-02850-7](https://doi.org/10.1038/s41598-021-02850-7)

- Schodt DJ, Farzam F, Liu S, Lidke KA. "Automated multi-target super-resolution microscopy with trust regions." *Biomedical Optics Express* 14(1):429-440, 2023. [DOI: 10.1364/BOE.477501](https://doi.org/10.1364/BOE.477501)

