```@meta
CurrentModule = SMLMDriftCorrection
```

# Configuration

The primary interface follows the JuliaSMLM tuple pattern:

```julia
(smld_corrected, info) = driftcorrect(smld, config)
```

where `config` is a [`DriftConfig`](@ref) and `info` is a [`DriftInfo`](@ref).

## Input: DriftConfig

```@docs
DriftConfig
```

### Quality Tiers

The `quality` parameter selects the algorithm complexity:

- **`:fft`** -- Fast cross-correlation of histogram images. No intra-dataset correction. Best for quick previews or very large datasets. It ignores `dataset_mode`.
- **`:singlepass`** (default) -- One pass of entropy-based intra-dataset correction followed by cross-correlation-seeded, entropy-refined inter-dataset alignment. Good balance of speed and accuracy.
- **`:iterative`** -- Iterates intra and inter correction until both parameter changes fall below `convergence_tol` or the entropy cost plateaus (relative improvement < 1e-4, both dataset modes). `max_iterations` is the number of iterations per attempt; a run that has not converged is retried from its current model up to 3 times.

### Dataset Modes

The `dataset_mode` parameter selects the inter-dataset algorithm. In `:iterative`, continuous mode also adds boundary priors to the intra fit. `quality=:fft` ignores `dataset_mode` and uses the registered-style CC passes:

- **`:registered`** (default) -- Datasets are independent acquisitions of the same field of view (e.g., SeqSRM). With two datasets only the second is aligned, against the first. With three or more, every dataset including the reference is aligned against the others; moves within 30 nm of the current shift are damped by (N-1)/N (N = number of datasets) to remove the overcount from updating all datasets at once. Larger moves into another basin are applied in full.
- **`:continuous`** -- One long acquisition split into multiple files or chunks. Each inter-chunk shift is measured by cross-correlating consecutive chunks (with endpoint-chaining kept only as a per-boundary fallback), then entropy-refined with continuity regularization.

No tier, `:fft` included, applies a cross-correlation shift with a component over 5 μm. Registered `:singlepass` and `:iterative`, and `:fft` regardless of `dataset_mode`, warn once per dataset per call when a dropped CC overlaps clearly better than the shift the dataset ends with, indicating a likely collection failure (the stage moved between acquisitions). Continuous `:singlepass` and `:iterative` drop such a boundary CC and fall back to endpoint chaining at that boundary without a warning. See [Theory and Workflow](@ref "Theory and Workflow") for the alignment rules.

### Chunking (Continuous Mode)

For long continuous acquisitions, a single polynomial may not capture complex drift. Use `chunk_frames` or `n_chunks` to split each dataset into temporal segments, each modeled independently:

```julia
# Split into ~4000-frame chunks
config = DriftConfig(dataset_mode=:continuous, chunk_frames=4000)

# Or specify number of chunks
config = DriftConfig(dataset_mode=:continuous, n_chunks=3)
```

Each chunk gets its own polynomial; consecutive chunks are aligned by cross-correlation (endpoint-chaining only as a fallback), which keeps a single mis-fit boundary from propagating down the chain.

### Auto-ROI

When `auto_roi=true`, the algorithm selects a dense rectangular subregion of the field of view for drift estimation, then applies the fitted model to all localizations. This can significantly speed up processing for large datasets while trading some accuracy (~1.4 nm vs ~0.5 nm RMSD in testing).

The ROI size is determined by `σ_loc`, `σ_target`, and `roi_safety_factor`:

```julia
config = DriftConfig(auto_roi=true, σ_loc=0.010, σ_target=0.001)
```

The estimated ROI indices are stored in `info.roi_indices`.

### Warm Starting

Pass a previously fitted model to initialize optimization:

```julia
# From a previous correction
(smld1, info1) = driftcorrect(smld1, DriftConfig(degree=2))

# Use as starting point for new data
config2 = DriftConfig(warm_start=info1.model)
(smld2, info2) = driftcorrect(smld2, config2)
```

## Output: DriftInfo

```@docs
DriftInfo
```

### Accessing Results

```julia
(smld_corrected, info) = driftcorrect(smld, DriftConfig())

# Diagnostics
info.converged     # true if convergence criterion met
info.entropy       # final entropy value (sum over localizations; divide by length(smld.emitters) for a size-independent number)
info.elapsed_s     # wall time in seconds
info.iterations    # iterations completed (0 for :fft, 1 for :singlepass)
info.residual_correlation  # position-frame correlation diagnostic (intra_summary, intra_per_dataset, inter)
info.history       # entropy after the singlepass start and each iteration (:iterative only; empty otherwise)
info.roi_indices   # indices used for ROI subset (nothing if auto_roi=false)
```

### Drift Trajectory

Extract the fitted drift model for plotting:

```julia
traj = drift_trajectory(info.model)
# traj.frames, traj.x, traj.y (and traj.z for 3D)

# For continuous mode, chain datasets end-to-end
traj = drift_trajectory(info.model; cumulative=true)
```

### Continuation

Pass `DriftInfo` directly to continue iterating from a previous result:

```julia
(smld1, info1) = driftcorrect(smld, DriftConfig(quality=:singlepass))

# Continue with more iterations
(smld2, info2) = driftcorrect(smld, info1; max_iterations=5)
```

Pass the original, uncorrected `smld` with its `info1`. Continuation reuses the original run's `dataset_mode` and chunking, so it also works on chunked continuous results.

## SMLD Alignment

For aligning independent SMLD structures (e.g., multi-color channels or separate acquisitions of the same FOV), use `align_smld` with `AlignConfig`.

### Input: AlignConfig

```@docs
AlignConfig
```

### Output: AlignInfo

```@docs
AlignInfo
```

### Usage

```julia
# Entropy method (default) -- CC initial guess + entropy refinement
(aligned, info) = align_smld([smld_ch1, smld_ch2])
info.shifts  # [zeros(2), [dx, dy]]

# FFT method -- cross-correlation only
(aligned, info) = align_smld(smlds; method=:fft)

# Config struct form
config = AlignConfig(method=:entropy, maxn=100)
(aligned, info) = align_smld(smlds, config)
```

The entropy method regularizes toward the cross-correlation initial guess to prevent optimizer divergence. Shift alignment works on both 2D and 3D data, threaded across datasets. Use `transform=:affine` for a 2D affine correction; see [`align_smld`](@ref) for requirements. `info.transforms` then holds similarity approximations, while `info.diagnostic` holds the exact affine coefficients and global shifts.
