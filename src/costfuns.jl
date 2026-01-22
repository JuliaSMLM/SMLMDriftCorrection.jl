"""
Cost functions for drift correction optimization.
Uses entropy-based cost with adaptive KDTree neighbor rebuilding.
"""

# ============================================================================
# Adaptive Neighbor State for efficient optimization
# ============================================================================

"""
    NeighborState

Mutable state for adaptive neighbor-based cost functions.
Tracks neighbor indices and triggers rebuilds when drift changes significantly.

Instead of rebuilding KDTree every iteration (O(N log N)), we:
1. Build neighbors once from initial coordinates
2. Only rebuild when total drift magnitude changes by more than threshold
3. Compute divergences only between fixed neighbor pairs (O(N × k))
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
    max_drift_magnitude(intra, nframes)

Compute maximum drift magnitude across all frames for current polynomial.
Used to detect when drift has changed enough to warrant neighbor rebuild.
"""
function max_drift_magnitude(intra::AbstractIntraDrift, nframes::Int)
    max_drift = 0.0
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

# ============================================================================
# Adaptive Entropy cost functions (self-excluded neighbor indices)
# ============================================================================

"""
    entropy1_2D_adaptive(neighbor_indices, x, y, σ_x, σ_y; divmethod)

Compute entropy using pre-computed neighbor indices (self excluded).
"""
function entropy1_2D_adaptive(neighbor_indices::Vector{Vector{Int}},
                               x::Vector{T}, y::Vector{T},
                               σ_x::Vector{T}, σ_y::Vector{T};
                               divmethod::String="KL") where {T<:Real}
    divfunc = select_divfunc_2D(divmethod)
    N = length(x)

    if isempty(neighbor_indices) || isempty(neighbor_indices[1])
        return entropy_HD(σ_x, σ_y)
    end

    maxn = length(neighbor_indices[1])
    log_maxn = log(T(maxn))

    kldiv = Vector{T}(undef, maxn)
    out = T(0)

    @inbounds for i in 1:N
        idx = neighbor_indices[i]
        xi, yi = x[i], y[i]
        sxi, syi = σ_x[i], σ_y[i]

        for j in 1:maxn
            jj = idx[j]
            kldiv[j] = divfunc(xi, yi, sxi, syi, x[jj], y[jj], σ_x[jj], σ_y[jj])
        end

        out += logsumexp(-kldiv) - log_maxn
    end

    return entropy_HD(σ_x, σ_y) - out / N
end

"""
    entropy1_3D_adaptive(neighbor_indices, x, y, z, σ_x, σ_y, σ_z; divmethod)

3D version of entropy1_2D_adaptive.
"""
function entropy1_3D_adaptive(neighbor_indices::Vector{Vector{Int}},
                               x::Vector{T}, y::Vector{T}, z::Vector{T},
                               σ_x::Vector{T}, σ_y::Vector{T}, σ_z::Vector{T};
                               divmethod::String="KL") where {T<:Real}
    divfunc = select_divfunc_3D(divmethod)
    N = length(x)

    if isempty(neighbor_indices) || isempty(neighbor_indices[1])
        return entropy_HD(σ_x, σ_y, σ_z)
    end

    maxn = length(neighbor_indices[1])
    log_maxn = log(T(maxn))

    kldiv = Vector{T}(undef, maxn)
    out = T(0)

    @inbounds for i in 1:N
        idx = neighbor_indices[i]
        xi, yi, zi = x[i], y[i], z[i]
        sxi, syi, szi = σ_x[i], σ_y[i], σ_z[i]

        for j in 1:maxn
            jj = idx[j]
            kldiv[j] = divfunc(xi, yi, zi, sxi, syi, szi,
                               x[jj], y[jj], z[jj], σ_x[jj], σ_y[jj], σ_z[jj])
        end

        out += logsumexp(-kldiv) - log_maxn
    end

    return entropy_HD(σ_x, σ_y, σ_z) - out / N
end

"""
INTRA-ENTROPY with adaptive neighbor rebuilding (2D)
"""
function costfun_entropy_intra_2D_adaptive(θ, x::Vector{T}, y::Vector{T},
                                            σ_x::Vector{T}, σ_y::Vector{T},
                                            framenum::Vector{Int}, maxn::Int,
                                            intra::AbstractIntraDrift,
                                            state::NeighborState{T}, nframes::Int;
                                            divmethod::String="KL",
                                            x_work::Vector{T}=similar(x),
                                            y_work::Vector{T}=similar(y)) where {T<:Real}
    theta2intra!(intra, θ)
    N = length(x)

    @inbounds for i in 1:N
        x_work[i] = correctdrift(x[i], framenum[i], intra.dm[1])
        y_work[i] = correctdrift(y[i], framenum[i], intra.dm[2])
    end

    maybe_rebuild_neighbors!(state, x_work, y_work, intra, nframes)

    return entropy1_2D_adaptive(state.neighbor_indices, x_work, y_work, σ_x, σ_y;
                                 divmethod=divmethod)
end

"""
INTRA-ENTROPY with adaptive neighbor rebuilding (3D)
"""
function costfun_entropy_intra_3D_adaptive(θ, x::Vector{T}, y::Vector{T}, z::Vector{T},
                                            σ_x::Vector{T}, σ_y::Vector{T}, σ_z::Vector{T},
                                            framenum::Vector{Int}, maxn::Int,
                                            intra::AbstractIntraDrift,
                                            state::NeighborState{T}, nframes::Int;
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

    maybe_rebuild_neighbors!(state, x_work, y_work, z_work, intra, nframes)

    return entropy1_3D_adaptive(state.neighbor_indices, x_work, y_work, z_work,
                                 σ_x, σ_y, σ_z; divmethod=divmethod)
end

# ============================================================================
# Inter-dataset entropy cost functions (non-adaptive, tree built once)
# ============================================================================

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
