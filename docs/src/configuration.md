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

- **`:fft`** -- Fast cross-correlation of histogram images. No intra-dataset correction. Best for quick previews or very large datasets.
- **`:singlepass`** (default) -- One pass of entropy-based intra-dataset correction followed by inter-dataset alignment. Good balance of speed and accuracy.
- **`:iterative`** -- Iterates intra and inter correction until convergence. Most accurate; resolves the coupling between intra and inter drift estimates.

### Dataset Modes

The `dataset_mode` parameter controls how multiple datasets are related:

- **`:registered`** (default) -- Datasets are independent acquisitions of the same field of view (e.g., SeqSRM). Inter-dataset alignment finds the best constant shift for each dataset against the others.
- **`:continuous`** -- One long acquisition split into multiple files or chunks. Polynomials are warm-started from the previous chunk's endpoint, and inter-shifts are regularized to maintain continuity.

### Chunking (Continuous Mode)

For long continuous acquisitions, a single polynomial may not capture complex drift. Use `chunk_frames` or `n_chunks` to split each dataset into temporal segments, each modeled independently:

```julia
# Split into ~4000-frame chunks
config = DriftConfig(dataset_mode=:continuous, chunk_frames=4000)

# Or specify number of chunks
config = DriftConfig(dataset_mode=:continuous, n_chunks=3)
```

Each chunk gets its own polynomial. Warm-starting from the previous chunk's endpoint ensures smooth transitions.

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
info.entropy       # final entropy value
info.elapsed_s     # wall time in seconds
info.iterations    # iterations completed (0 for :fft, 1 for :singlepass)
info.history       # entropy per iteration (for plotting convergence)
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
