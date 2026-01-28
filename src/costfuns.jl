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
# Inter-dataset entropy cost functions (merged cloud approach)
# ============================================================================

"""
    costfun_entropy_inter_2D_merged(θ, x_n, y_n, σ_x_n, σ_y_n,
                                     x_ref, y_ref, σ_x_ref, σ_y_ref,
                                     maxn, inter; kwargs...)

Inter-dataset entropy cost: compute entropy of combined cloud
(shifted dataset_n + reference datasets).

When properly aligned, the combined cloud is tighter (lower entropy).

# Arguments
- `θ`: shift parameters [dx, dy]
- `x_n, y_n`: coordinates of dataset to be shifted (uncorrected)
- `σ_x_n, σ_y_n`: uncertainties of dataset to be shifted
- `x_ref, y_ref`: coordinates of reference datasets (already corrected)
- `σ_x_ref, σ_y_ref`: uncertainties of reference datasets
- `maxn`: maximum neighbors for entropy calculation
- `inter`: InterShift struct to update

# Keyword Arguments
- `divmethod`: divergence method ("KL" or "KL2")
- `x_work, y_work`: pre-allocated work arrays for shifted coords
"""
function costfun_entropy_inter_2D_merged(θ,
    x_n::Vector{T}, y_n::Vector{T}, σ_x_n::Vector{T}, σ_y_n::Vector{T},
    x_ref::Vector{T}, y_ref::Vector{T}, σ_x_ref::Vector{T}, σ_y_ref::Vector{T},
    maxn::Int, inter::InterShift;
    divmethod::String="KL",
    x_work::Vector{T}=similar(x_n),
    y_work::Vector{T}=similar(y_n)) where {T<:Real}

    # Apply shift to dataset_n
    theta2inter!(inter, θ)
    N_n = length(x_n)
    @inbounds for i in 1:N_n
        x_work[i] = correctdrift(x_n[i], inter, 1)
        y_work[i] = correctdrift(y_n[i], inter, 2)
    end

    # Combine shifted dataset with reference (reference is already corrected)
    x_combined = vcat(x_work, x_ref)
    y_combined = vcat(y_work, y_ref)
    σ_x_combined = vcat(σ_x_n, σ_x_ref)
    σ_y_combined = vcat(σ_y_n, σ_y_ref)

    return ub_entropy(x_combined, y_combined, σ_x_combined, σ_y_combined;
                      maxn=maxn, divmethod=divmethod)
end

"""
    costfun_entropy_inter_3D_merged(θ, x_n, y_n, z_n, σ_x_n, σ_y_n, σ_z_n,
                                     x_ref, y_ref, z_ref, σ_x_ref, σ_y_ref, σ_z_ref,
                                     maxn, inter; kwargs...)

3D version of `costfun_entropy_inter_2D_merged`.
"""
function costfun_entropy_inter_3D_merged(θ,
    x_n::Vector{T}, y_n::Vector{T}, z_n::Vector{T},
    σ_x_n::Vector{T}, σ_y_n::Vector{T}, σ_z_n::Vector{T},
    x_ref::Vector{T}, y_ref::Vector{T}, z_ref::Vector{T},
    σ_x_ref::Vector{T}, σ_y_ref::Vector{T}, σ_z_ref::Vector{T},
    maxn::Int, inter::InterShift;
    divmethod::String="KL",
    x_work::Vector{T}=similar(x_n),
    y_work::Vector{T}=similar(y_n),
    z_work::Vector{T}=similar(z_n)) where {T<:Real}

    # Apply shift to dataset_n
    theta2inter!(inter, θ)
    N_n = length(x_n)
    @inbounds for i in 1:N_n
        x_work[i] = correctdrift(x_n[i], inter, 1)
        y_work[i] = correctdrift(y_n[i], inter, 2)
        z_work[i] = correctdrift(z_n[i], inter, 3)
    end

    # Combine shifted dataset with reference (reference is already corrected)
    x_combined = vcat(x_work, x_ref)
    y_combined = vcat(y_work, y_ref)
    z_combined = vcat(z_work, z_ref)
    σ_x_combined = vcat(σ_x_n, σ_x_ref)
    σ_y_combined = vcat(σ_y_n, σ_y_ref)
    σ_z_combined = vcat(σ_z_n, σ_z_ref)

    return ub_entropy(x_combined, y_combined, z_combined,
                      σ_x_combined, σ_y_combined, σ_z_combined;
                      maxn=maxn, divmethod=divmethod)
end
