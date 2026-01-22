"""
Cost functions for drift correction optimization.
Optimized to avoid unnecessary allocations.
"""

# ============================================================================
# Adaptive Neighbor State for efficient intra-dataset optimization
# ============================================================================

"""
    NeighborState

Mutable state for adaptive neighbor-based cost functions.
Tracks neighbor indices and triggers rebuilds when drift changes significantly.

Instead of rebuilding KDTree every iteration (O(N log N)), we:
1. Build neighbors once from initial coordinates
2. Only rebuild when total drift magnitude changes by more than threshold
3. Compute distances only between fixed neighbor pairs (O(N × k))

This gives ~100x speedup for typical drift correction scenarios.
"""
mutable struct NeighborState{T<:Real}
    neighbor_indices::Vector{Vector{Int}}  # neighbor_indices[i] = indices of neighbors of point i
    last_rebuild_drift::T                   # max drift magnitude at last rebuild
    rebuild_threshold::T                    # drift change that triggers rebuild
    rebuild_count::Int                      # number of rebuilds (for diagnostics)
    k::Int                                  # number of neighbors
end

function NeighborState(N::Int, k::Int, rebuild_threshold::T) where {T<:Real}
    neighbor_indices = [Int[] for _ in 1:N]
    return NeighborState{T}(neighbor_indices, T(0), rebuild_threshold, 0, k)
end

"""
    build_neighbors!(state, x, y)

Build neighbor indices from 2D coordinates using KDTree.
"""
function build_neighbors!(state::NeighborState{T}, x::Vector{T}, y::Vector{T}) where {T<:Real}
    N = length(x)
    k = min(state.k, N - 1)
    if k < 1
        return
    end

    data = Matrix{T}(undef, 2, N)
    @inbounds for i in 1:N
        data[1, i] = x[i]
        data[2, i] = y[i]
    end

    kdtree = KDTree(data; leafsize=10)
    idxs, _ = knn(kdtree, data, k + 1, true)  # k+1 because first is self

    @inbounds for i in 1:N
        state.neighbor_indices[i] = idxs[i][2:end]  # exclude self
    end

    state.rebuild_count += 1
end

"""
    build_neighbors!(state, x, y, z)

Build neighbor indices from 3D coordinates using KDTree.
"""
function build_neighbors!(state::NeighborState{T}, x::Vector{T}, y::Vector{T}, z::Vector{T}) where {T<:Real}
    N = length(x)
    k = min(state.k, N - 1)
    if k < 1
        return
    end

    data = Matrix{T}(undef, 3, N)
    @inbounds for i in 1:N
        data[1, i] = x[i]
        data[2, i] = y[i]
        data[3, i] = z[i]
    end

    kdtree = KDTree(data; leafsize=10)
    idxs, _ = knn(kdtree, data, k + 1, true)

    @inbounds for i in 1:N
        state.neighbor_indices[i] = idxs[i][2:end]
    end

    state.rebuild_count += 1
end

"""
    max_drift_magnitude(intra::AbstractIntraDrift, nframes::Int)

Compute maximum drift magnitude across all frames for current polynomial.
Used to detect when drift has changed enough to warrant neighbor rebuild.
"""
function max_drift_magnitude(intra::AbstractIntraDrift, nframes::Int)
    max_drift = 0.0
    # Sample at endpoints and midpoint (polynomial extrema are at boundaries or interior)
    for frame in [1, nframes ÷ 2, nframes]
        drift_sq = 0.0
        for dim in 1:intra.ndims
            d = evaluate_at_frame(intra.dm[dim], frame)
            drift_sq += d * d
        end
        max_drift = max(max_drift, sqrt(drift_sq))
    end
    return max_drift
end

"""
    maybe_rebuild_neighbors!(state, x_work, y_work, intra, nframes)

Check if neighbors need rebuilding based on drift magnitude change.
Rebuilds if |current_drift - last_rebuild_drift| > threshold.
"""
function maybe_rebuild_neighbors!(state::NeighborState{T},
                                   x_work::Vector{T}, y_work::Vector{T},
                                   intra::AbstractIntraDrift, nframes::Int) where {T<:Real}
    current_drift = max_drift_magnitude(intra, nframes)

    if abs(current_drift - state.last_rebuild_drift) > state.rebuild_threshold
        build_neighbors!(state, x_work, y_work)
        state.last_rebuild_drift = current_drift
    end
end

"""
    maybe_rebuild_neighbors!(state, x_work, y_work, z_work, intra, nframes)

3D version of maybe_rebuild_neighbors!
"""
function maybe_rebuild_neighbors!(state::NeighborState{T},
                                   x_work::Vector{T}, y_work::Vector{T}, z_work::Vector{T},
                                   intra::AbstractIntraDrift, nframes::Int) where {T<:Real}
    current_drift = max_drift_magnitude(intra, nframes)

    if abs(current_drift - state.last_rebuild_drift) > state.rebuild_threshold
        build_neighbors!(state, x_work, y_work, z_work)
        state.last_rebuild_drift = current_drift
    end
end

"""
    cost_from_neighbors_2D(x, y, neighbor_indices, d_cutoff)

Compute KDTree-style cost using precomputed neighbor indices.
O(N × k) instead of O(N log N) for tree building.
"""
function cost_from_neighbors_2D(x::Vector{T}, y::Vector{T},
                                 neighbor_indices::Vector{Vector{Int}},
                                 d_cutoff::T) where {T<:Real}
    cost = T(0)
    @inbounds for i in 1:length(x)
        for j in neighbor_indices[i]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = sqrt(dx * dx + dy * dy)
            cost -= exp(-d / d_cutoff)
        end
    end
    return cost
end

"""
    cost_from_neighbors_3D(x, y, z, neighbor_indices, d_cutoff)

3D version of cost_from_neighbors.
"""
function cost_from_neighbors_3D(x::Vector{T}, y::Vector{T}, z::Vector{T},
                                 neighbor_indices::Vector{Vector{Int}},
                                 d_cutoff::T) where {T<:Real}
    cost = T(0)
    @inbounds for i in 1:length(x)
        for j in neighbor_indices[i]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dz = z[i] - z[j]
            d = sqrt(dx * dx + dy * dy + dz * dz)
            cost -= exp(-d / d_cutoff)
        end
    end
    return cost
end

# ============================================================================
# Adaptive neighbor cost functions for INTRA-dataset optimization
# ============================================================================

"""
INTRA-KNN with adaptive neighbor rebuilding (2D)

Uses fixed neighbor indices, only rebuilding when drift magnitude changes
significantly. Much faster than rebuilding KDTree every iteration.

# Arguments
- `θ`: parameters for drift correction proposal
- `x, y`: uncorrected coordinates
- `framenum`: frame number for each localization
- `d_cutoff`: cutoff distance
- `intra`: intra-dataset drift model
- `state`: NeighborState for tracking neighbors and rebuilds
- `nframes`: number of frames (for drift magnitude calculation)
- `x_work, y_work`: pre-allocated work arrays
"""
function costfun_kdtree_intra_2D_adaptive(θ, x::Vector{T}, y::Vector{T},
                                           framenum::Vector{Int}, d_cutoff::T,
                                           intra::AbstractIntraDrift,
                                           state::NeighborState{T}, nframes::Int;
                                           x_work::Vector{T}=similar(x),
                                           y_work::Vector{T}=similar(y)) where {T<:Real}
    theta2intra!(intra, θ)
    N = length(x)

    # Apply drift correction to work arrays
    @inbounds for i in 1:N
        x_work[i] = correctdrift(x[i], framenum[i], intra.dm[1])
        y_work[i] = correctdrift(y[i], framenum[i], intra.dm[2])
    end

    # Check if we need to rebuild neighbors
    maybe_rebuild_neighbors!(state, x_work, y_work, intra, nframes)

    # Compute cost using fixed neighbors
    return cost_from_neighbors_2D(x_work, y_work, state.neighbor_indices, d_cutoff)
end

"""
INTRA-KNN with adaptive neighbor rebuilding (3D)
"""
function costfun_kdtree_intra_3D_adaptive(θ, x::Vector{T}, y::Vector{T}, z::Vector{T},
                                           framenum::Vector{Int}, d_cutoff::T,
                                           intra::AbstractIntraDrift,
                                           state::NeighborState{T}, nframes::Int;
                                           x_work::Vector{T}=similar(x),
                                           y_work::Vector{T}=similar(y),
                                           z_work::Vector{T}=similar(z)) where {T<:Real}
    theta2intra!(intra, θ)
    N = length(x)

    @inbounds for i in 1:N
        x_work[i] = correctdrift(x[i], framenum[i], intra.dm[1])
        y_work[i] = correctdrift(y[i], framenum[i], intra.dm[2])
        z_work[i] = correctdrift(z[i], framenum[i], intra.dm[3])
    end

    maybe_rebuild_neighbors!(state, x_work, y_work, z_work, intra, nframes)

    return cost_from_neighbors_3D(x_work, y_work, z_work, state.neighbor_indices, d_cutoff)
end

# ============================================================================
# KDTree-based cost functions (KNN approach) - Original implementations
# ============================================================================

"""
INTRA-KNN
Cost function computes the cost of an intra-dataset drift correction proposal.
Here, the cost is the sum of the scaled negative exponentials of the
nearest neighbor distances computed frame-by-frame within a dataset.

Optimized: applies drift directly without deepcopy.

# Fields:
- θ:          parameters for a drift correction proposal
- x, y:       uncorrected coordinates (vectors)
- framenum:   frame number for each localization
- d_cutoff:   cutoff distance
- intra:      intra-dataset data structure
- x_work, y_work: pre-allocated work arrays (optional)
"""
function costfun_kdtree_intra_2D(θ, x::Vector{T}, y::Vector{T},
                                  framenum::Vector{Int}, d_cutoff::T,
                                  intra::AbstractIntraDrift;
                                  x_work::Vector{T}=similar(x),
                                  y_work::Vector{T}=similar(y)) where {T<:Real}
    theta2intra!(intra, θ)
    N = length(x)

    # Apply drift correction to work arrays
    @inbounds for i in 1:N
        x_work[i] = correctdrift(x[i], framenum[i], intra.dm[1])
        y_work[i] = correctdrift(y[i], framenum[i], intra.dm[2])
    end

    # Build KDTree from corrected data
    data = Matrix{T}(undef, 2, N)
    @inbounds for i in 1:N
        data[1, i] = x_work[i]
        data[2, i] = y_work[i]
    end

    k = min(4, N - 1)   # number of nearest neighbors (bounded by available points)
    if k < 1
        return T(0)  # Not enough points for meaningful cost
    end
    kdtree = KDTree(data; leafsize=10)
    idxs, dists = knn(kdtree, data, k, true)

    cost = T(0)
    @inbounds for nn in 2:length(dists)
        for d in dists[nn]
            cost -= exp(-d / d_cutoff)
        end
    end

    return cost
end

"""
INTRA-KNN for 3D
"""
function costfun_kdtree_intra_3D(θ, x::Vector{T}, y::Vector{T}, z::Vector{T},
                                  framenum::Vector{Int}, d_cutoff::T,
                                  intra::AbstractIntraDrift;
                                  x_work::Vector{T}=similar(x),
                                  y_work::Vector{T}=similar(y),
                                  z_work::Vector{T}=similar(z)) where {T<:Real}
    theta2intra!(intra, θ)
    N = length(x)

    @inbounds for i in 1:N
        x_work[i] = correctdrift(x[i], framenum[i], intra.dm[1])
        y_work[i] = correctdrift(y[i], framenum[i], intra.dm[2])
        z_work[i] = correctdrift(z[i], framenum[i], intra.dm[3])
    end

    data = Matrix{T}(undef, 3, N)
    @inbounds for i in 1:N
        data[1, i] = x_work[i]
        data[2, i] = y_work[i]
        data[3, i] = z_work[i]
    end

    k = min(4, N - 1)
    if k < 1
        return T(0)
    end
    kdtree = KDTree(data; leafsize=10)
    idxs, dists = knn(kdtree, data, k, true)

    cost = T(0)
    @inbounds for nn in 2:length(dists)
        for d in dists[nn]
            cost -= exp(-d / d_cutoff)
        end
    end

    return cost
end

"""
INTER-KNN with prebuilt KDTree
Cost function for inter-dataset drift correction using a static reference KDTree.

# Fields:
- θ:          parameters for a drift correction proposal
- x, y:       uncorrected coordinates (vectors)
- kdtree:     KDTree from reference dataset
- d_cutoff:   cutoff distance
- inter:      inter-dataset data structure
"""
function costfun_kdtree_inter_2D(θ, x::Vector{T}, y::Vector{T},
                                  kdtree::KDTree, d_cutoff::T,
                                  inter::InterShift;
                                  x_work::Vector{T}=similar(x),
                                  y_work::Vector{T}=similar(y)) where {T<:Real}
    theta2inter!(inter, θ)
    N = length(x)

    @inbounds for i in 1:N
        x_work[i] = correctdrift(x[i], inter, 1)
        y_work[i] = correctdrift(y[i], inter, 2)
    end

    data = Matrix{T}(undef, 2, N)
    @inbounds for i in 1:N
        data[1, i] = x_work[i]
        data[2, i] = y_work[i]
    end

    k = min(4, N - 1)
    if k < 1
        return T(0)
    end
    idxs, dists = knn(kdtree, data, k, true)

    cost = T(0)
    @inbounds for nn in 1:length(dists)
        for d in dists[nn]
            cost -= exp(-d / d_cutoff)
        end
    end

    return cost
end

"""
INTER-KNN for 3D with prebuilt KDTree
"""
function costfun_kdtree_inter_3D(θ, x::Vector{T}, y::Vector{T}, z::Vector{T},
                                  kdtree::KDTree, d_cutoff::T,
                                  inter::InterShift;
                                  x_work::Vector{T}=similar(x),
                                  y_work::Vector{T}=similar(y),
                                  z_work::Vector{T}=similar(z)) where {T<:Real}
    theta2inter!(inter, θ)
    N = length(x)

    @inbounds for i in 1:N
        x_work[i] = correctdrift(x[i], inter, 1)
        y_work[i] = correctdrift(y[i], inter, 2)
        z_work[i] = correctdrift(z[i], inter, 3)
    end

    data = Matrix{T}(undef, 3, N)
    @inbounds for i in 1:N
        data[1, i] = x_work[i]
        data[2, i] = y_work[i]
        data[3, i] = z_work[i]
    end

    k = min(4, N - 1)
    if k < 1
        return T(0)
    end
    idxs, dists = knn(kdtree, data, k, true)

    cost = T(0)
    @inbounds for nn in 1:length(dists)
        for d in dists[nn]
            cost -= exp(-d / d_cutoff)
        end
    end

    return cost
end

# ============================================================================
# Entropy-based cost functions
# ============================================================================

"""
INTRA-ENTROPY for 2D
Cost function using entropy upper bound. Optimized to avoid deepcopy.

# Fields:
- θ:          parameters for a drift correction proposal
- x, y:       uncorrected coordinates (vectors)
- σ_x, σ_y:   localization uncertainties
- framenum:   frame number for each localization
- maxn:       maximum number of neighbors
- intra:      intra-dataset data structure
- divmethod:  divergence method
- x_work, y_work: pre-allocated work arrays
"""
function costfun_entropy_intra_2D(θ, x::Vector{T}, y::Vector{T},
                                   σ_x::Vector{T}, σ_y::Vector{T},
                                   framenum::Vector{Int}, maxn::Int,
                                   intra::AbstractIntraDrift;
                                   divmethod::String="KL",
                                   x_work::Vector{T}=similar(x),
                                   y_work::Vector{T}=similar(y)) where {T<:Real}
    theta2intra!(intra, θ)
    N = length(x)

    @inbounds for i in 1:N
        x_work[i] = correctdrift(x[i], framenum[i], intra.dm[1])
        y_work[i] = correctdrift(y[i], framenum[i], intra.dm[2])
    end

    return ub_entropy(x_work, y_work, σ_x, σ_y; maxn=maxn, divmethod=divmethod)
end

"""
INTRA-ENTROPY for 3D
"""
function costfun_entropy_intra_3D(θ, x::Vector{T}, y::Vector{T}, z::Vector{T},
                                   σ_x::Vector{T}, σ_y::Vector{T}, σ_z::Vector{T},
                                   framenum::Vector{Int}, maxn::Int,
                                   intra::AbstractIntraDrift;
                                   divmethod::String="KL",
                                   x_work::Vector{T}=similar(x),
                                   y_work::Vector{T}=similar(y),
                                   z_work::Vector{T}=similar(z)) where {T<:Real}
    theta2intra!(intra, θ)
    N = length(x)

    @inbounds for i in 1:N
        x_work[i] = correctdrift(x[i], framenum[i], intra.dm[1])
        y_work[i] = correctdrift(y[i], framenum[i], intra.dm[2])
        z_work[i] = correctdrift(z[i], framenum[i], intra.dm[3])
    end

    return ub_entropy(x_work, y_work, z_work, σ_x, σ_y, σ_z; maxn=maxn, divmethod=divmethod)
end

"""
INTER-ENTROPY for 2D
"""
function costfun_entropy_inter_2D(θ, x::Vector{T}, y::Vector{T},
                                   σ_x::Vector{T}, σ_y::Vector{T},
                                   maxn::Int, inter::InterShift;
                                   divmethod::String="KL",
                                   x_work::Vector{T}=similar(x),
                                   y_work::Vector{T}=similar(y)) where {T<:Real}
    theta2inter!(inter, θ)
    N = length(x)

    @inbounds for i in 1:N
        x_work[i] = correctdrift(x[i], inter, 1)
        y_work[i] = correctdrift(y[i], inter, 2)
    end

    return ub_entropy(x_work, y_work, σ_x, σ_y; maxn=maxn, divmethod=divmethod)
end

"""
INTER-ENTROPY for 3D
"""
function costfun_entropy_inter_3D(θ, x::Vector{T}, y::Vector{T}, z::Vector{T},
                                   σ_x::Vector{T}, σ_y::Vector{T}, σ_z::Vector{T},
                                   maxn::Int, inter::InterShift;
                                   divmethod::String="KL",
                                   x_work::Vector{T}=similar(x),
                                   y_work::Vector{T}=similar(y),
                                   z_work::Vector{T}=similar(z)) where {T<:Real}
    theta2inter!(inter, θ)
    N = length(x)

    @inbounds for i in 1:N
        x_work[i] = correctdrift(x[i], inter, 1)
        y_work[i] = correctdrift(y[i], inter, 2)
        z_work[i] = correctdrift(z[i], inter, 3)
    end

    return ub_entropy(x_work, y_work, z_work, σ_x, σ_y, σ_z; maxn=maxn, divmethod=divmethod)
end
