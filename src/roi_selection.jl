# Automatic ROI subsampling for faster drift correction
# Stage drift is spatially uniform, so models estimated from a dense subset apply everywhere

"""
    calculate_n_locs_required(n_frames; kwargs...) -> Int

Calculate minimum number of localizations needed for accurate drift estimation.

The formula accounts for:
- Polynomial degree (more coefficients = more data needed)
- Localization precision vs target drift precision
- Temporal density of pairs (depends on blink rate and frame count)

# Keyword Arguments
- `degree=2`: Polynomial degree for drift model
- `σ_loc=0.010`: Typical localization precision (μm), default 10 nm
- `σ_target=0.001`: Target drift precision (μm), default 1 nm
- `safety_factor=2.0`: Safety multiplier for required pairs
- `n_frames_target=1000`: Target frame window for temporal pairing

# Returns
Integer number of localizations required.

# Details
Based on error propagation: to achieve σ_target from σ_loc measurements,
need n_pairs = 2*(degree+1)*(σ_loc/σ_target)² pairs. The number of pairs
available depends on the temporal density of emitter blinks.
"""
function calculate_n_locs_required(n_frames::Int;
    degree::Int = 2,
    σ_loc::Float64 = 0.010,
    σ_target::Float64 = 0.001,
    safety_factor::Float64 = 2.0,
    n_frames_target::Int = 1000)

    # Required pairs for target precision
    n_pairs = 2 * (degree + 1) * (σ_loc / σ_target)^2 * safety_factor

    # Temporal scaling: ~3 blinks/emitter spread over n_frames
    # Pairs form within a ~1000-frame window
    λ_window = 3.0 * (n_frames_target / n_frames)

    # Pairs per emitter (Poisson pairing)
    pairs_per_emitter = max(λ_window^2 / 2, 0.01)  # guard against very small

    # Convert to emitters, then to localizations (~10 locs/emitter)
    n_emitters = n_pairs / pairs_per_emitter
    return ceil(Int, n_emitters * 10)
end

"""
    find_dense_roi(smld, n_locs_target; k=10) -> indices

Find indices of the `n_locs_target` densest localizations using k-NN density estimation.

Uses inverse mean k-nearest-neighbor distance as a density proxy.
Returns indices sorted by density (highest first), truncated to n_locs_target.

# Arguments
- `smld`: SMLD structure with localization data
- `n_locs_target`: Number of localizations to select

# Keyword Arguments
- `k=10`: Number of neighbors for density estimation

# Returns
Vector of integer indices into smld.emitters for the densest localizations.
"""
function find_dense_roi(smld::SMLD, n_locs_target::Int; k::Int = 10)
    n_locs = length(smld.emitters)

    # Edge case: if target >= total, return all
    if n_locs_target >= n_locs
        return collect(1:n_locs)
    end

    # Build coordinate matrix
    is_3d = nDims(smld) == 3
    if is_3d
        coords = Matrix{Float64}(undef, 3, n_locs)
        for (i, e) in enumerate(smld.emitters)
            coords[1, i] = e.x
            coords[2, i] = e.y
            coords[3, i] = e.z
        end
    else
        coords = Matrix{Float64}(undef, 2, n_locs)
        for (i, e) in enumerate(smld.emitters)
            coords[1, i] = e.x
            coords[2, i] = e.y
        end
    end

    # Build KDTree and query k+1 neighbors (includes self)
    tree = KDTree(coords)
    k_query = min(k + 1, n_locs)
    _, dists = knn(tree, coords, k_query)

    # Compute density as inverse of mean k-NN distance (excluding self at dist=0)
    density = Vector{Float64}(undef, n_locs)
    for i in 1:n_locs
        # Skip self (first neighbor at distance ~0)
        d = dists[i]
        if length(d) > 1
            mean_dist = sum(d[2:end]) / (length(d) - 1)
            density[i] = 1.0 / max(mean_dist, 1e-10)
        else
            density[i] = 0.0
        end
    end

    # Sort by density (descending) and return top n_locs_target indices
    sorted_indices = sortperm(density, rev=true)
    return sorted_indices[1:n_locs_target]
end
