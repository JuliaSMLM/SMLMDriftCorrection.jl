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
# Inter-dataset entropy cost functions (optimized merged cloud approach)
# ============================================================================

"""
    InterNeighborState

State for inter-dataset entropy optimization with adaptive neighbor rebuilding.
Only rebuilds KDTree when shift magnitude changes significantly.
"""
mutable struct InterNeighborState{T<:Real}
    neighbor_indices::Vector{Vector{Int}}  # neighbor_indices[i] = indices of neighbors of point i
    last_shift::Vector{T}                   # shift at last rebuild
    rebuild_threshold::T                    # shift change that triggers rebuild
    rebuild_count::Int                      # for diagnostics
    k::Int                                  # number of neighbors
end

function InterNeighborState(N_n::Int, k::Int, rebuild_threshold::T) where {T<:Real}
    neighbor_indices = [Int[] for _ in 1:N_n]
    return InterNeighborState{T}(neighbor_indices, T[Inf, Inf], rebuild_threshold, 0, k)
end

function InterNeighborState3D(N_n::Int, k::Int, rebuild_threshold::T) where {T<:Real}
    neighbor_indices = [Int[] for _ in 1:N_n]
    return InterNeighborState{T}(neighbor_indices, T[Inf, Inf, Inf], rebuild_threshold, 0, k)
end

"""
    costfun_entropy_inter_2D_merged(θ, x_n, y_n, σ_x_n, σ_y_n,
                                     x_ref, y_ref, σ_x_ref, σ_y_ref,
                                     maxn, inter; kwargs...)

Inter-dataset entropy cost: compute entropy contribution from dataset_n points
when merged with reference datasets.

OPTIMIZATION: Uses adaptive neighbor rebuilding - only rebuilds KDTree when
shift changes by more than 0.5 μm. Since typical inter-dataset shifts are
small, neighbors are stable across most optimizer iterations.

When properly aligned, dataset_n points have lower entropy (neighbors are closer).

# Arguments
- `θ`: shift parameters [dx, dy]
- `x_n, y_n`: coordinates of dataset to be shifted (uncorrected)
- `σ_x_n, σ_y_n`: uncertainties of dataset to be shifted
- `x_ref, y_ref`: coordinates of reference datasets (already corrected)
- `σ_x_ref, σ_y_ref`: uncertainties of reference datasets
- `maxn`: maximum neighbors for entropy calculation
- `inter`: InterShift struct to update

# Keyword Arguments
- `divmethod`: divergence method ("KL")
- `x_work, y_work`: pre-allocated work arrays for shifted coords
- `data_combined`: pre-allocated 2×(N_n+N_ref) matrix for KDTree
- `state`: InterNeighborState for adaptive rebuilding
"""
function costfun_entropy_inter_2D_merged(θ,
    x_n::Vector{T}, y_n::Vector{T}, σ_x_n::Vector{T}, σ_y_n::Vector{T},
    x_ref::Vector{T}, y_ref::Vector{T}, σ_x_ref::Vector{T}, σ_y_ref::Vector{T},
    maxn::Int, inter::InterShift;
    divmethod::String="KL",
    x_work::Vector{T}=similar(x_n),
    y_work::Vector{T}=similar(y_n),
    data_combined::Matrix{T}=Matrix{T}(undef, 2, length(x_n)+length(x_ref)),
    state::Union{InterNeighborState{T}, Nothing}=nothing) where {T<:Real}

    divfunc = select_divfunc_2D(divmethod)

    # Apply shift to dataset_n
    theta2inter!(inter, θ)
    N_n = length(x_n)
    N_ref = length(x_ref)
    N_combined = N_n + N_ref

    @inbounds for i in 1:N_n
        x_work[i] = correctdrift(x_n[i], inter, 1)
        y_work[i] = correctdrift(y_n[i], inter, 2)
    end

    # Build combined coordinate matrix for KDTree
    # First N_n entries are shifted dataset_n, rest are reference
    @inbounds for i in 1:N_n
        data_combined[1, i] = x_work[i]
        data_combined[2, i] = y_work[i]
    end
    @inbounds for i in 1:N_ref
        data_combined[1, N_n + i] = x_ref[i]
        data_combined[2, N_n + i] = y_ref[i]
    end

    k = min(maxn, N_combined - 1)

    # Check if we need to rebuild neighbors
    need_rebuild = state === nothing ||
                   isempty(state.neighbor_indices[1]) ||
                   sqrt((θ[1] - state.last_shift[1])^2 + (θ[2] - state.last_shift[2])^2) > state.rebuild_threshold

    if need_rebuild
        kdtree = KDTree(data_combined; leafsize=10)
        query_points = view(data_combined, :, 1:N_n)
        idxs, _ = knn(kdtree, query_points, k + 1, true)

        if state !== nothing
            @inbounds for i in 1:N_n
                state.neighbor_indices[i] = idxs[i][2:end]  # exclude self
            end
            state.last_shift[1] = θ[1]
            state.last_shift[2] = θ[2]
            state.rebuild_count += 1
        end
    end

    # Get neighbor indices
    neighbor_indices = state !== nothing ? state.neighbor_indices : [idxs[i][2:end] for i in 1:N_n]

    # Compute entropy contribution from dataset_n points only
    log_k = log(T(k))
    kldiv = Vector{T}(undef, k)
    out = T(0)

    @inbounds for i in 1:N_n
        idx = neighbor_indices[i]
        xi, yi = x_work[i], y_work[i]
        sxi, syi = σ_x_n[i], σ_y_n[i]

        for j in 1:k
            jj = idx[j]
            # Get coordinates from combined cloud
            if jj <= N_n
                xj, yj = x_work[jj], y_work[jj]
                sxj, syj = σ_x_n[jj], σ_y_n[jj]
            else
                ref_idx = jj - N_n
                xj, yj = x_ref[ref_idx], y_ref[ref_idx]
                sxj, syj = σ_x_ref[ref_idx], σ_y_ref[ref_idx]
            end
            kldiv[j] = divfunc(xi, yi, sxi, syi, xj, yj, sxj, syj)
        end

        out += logsumexp(-kldiv) - log_k
    end

    return entropy_HD(σ_x_n, σ_y_n) - out / N_n
end

"""
    costfun_entropy_inter_3D_merged(θ, x_n, y_n, z_n, σ_x_n, σ_y_n, σ_z_n,
                                     x_ref, y_ref, z_ref, σ_x_ref, σ_y_ref, σ_z_ref,
                                     maxn, inter; kwargs...)

3D version of `costfun_entropy_inter_2D_merged`. Same adaptive rebuilding optimization.
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
    z_work::Vector{T}=similar(z_n),
    data_combined::Matrix{T}=Matrix{T}(undef, 3, length(x_n)+length(x_ref)),
    state::Union{InterNeighborState{T}, Nothing}=nothing) where {T<:Real}

    divfunc = select_divfunc_3D(divmethod)

    # Apply shift to dataset_n
    theta2inter!(inter, θ)
    N_n = length(x_n)
    N_ref = length(x_ref)
    N_combined = N_n + N_ref

    @inbounds for i in 1:N_n
        x_work[i] = correctdrift(x_n[i], inter, 1)
        y_work[i] = correctdrift(y_n[i], inter, 2)
        z_work[i] = correctdrift(z_n[i], inter, 3)
    end

    # Build combined coordinate matrix for KDTree
    @inbounds for i in 1:N_n
        data_combined[1, i] = x_work[i]
        data_combined[2, i] = y_work[i]
        data_combined[3, i] = z_work[i]
    end
    @inbounds for i in 1:N_ref
        data_combined[1, N_n + i] = x_ref[i]
        data_combined[2, N_n + i] = y_ref[i]
        data_combined[3, N_n + i] = z_ref[i]
    end

    k = min(maxn, N_combined - 1)

    # Check if we need to rebuild neighbors
    need_rebuild = state === nothing ||
                   isempty(state.neighbor_indices[1]) ||
                   sqrt((θ[1] - state.last_shift[1])^2 + (θ[2] - state.last_shift[2])^2 + (θ[3] - state.last_shift[3])^2) > state.rebuild_threshold

    if need_rebuild
        kdtree = KDTree(data_combined; leafsize=10)
        query_points = view(data_combined, :, 1:N_n)
        idxs, _ = knn(kdtree, query_points, k + 1, true)

        if state !== nothing
            @inbounds for i in 1:N_n
                state.neighbor_indices[i] = idxs[i][2:end]
            end
            state.last_shift[1] = θ[1]
            state.last_shift[2] = θ[2]
            state.last_shift[3] = θ[3]
            state.rebuild_count += 1
        end
    end

    neighbor_indices = state !== nothing ? state.neighbor_indices : [idxs[i][2:end] for i in 1:N_n]

    # Compute entropy contribution from dataset_n points only
    log_k = log(T(k))
    kldiv = Vector{T}(undef, k)
    out = T(0)

    @inbounds for i in 1:N_n
        idx = neighbor_indices[i]
        xi, yi, zi = x_work[i], y_work[i], z_work[i]
        sxi, syi, szi = σ_x_n[i], σ_y_n[i], σ_z_n[i]

        for j in 1:k
            jj = idx[j]
            if jj <= N_n
                xj, yj, zj = x_work[jj], y_work[jj], z_work[jj]
                sxj, syj, szj = σ_x_n[jj], σ_y_n[jj], σ_z_n[jj]
            else
                ref_idx = jj - N_n
                xj, yj, zj = x_ref[ref_idx], y_ref[ref_idx], z_ref[ref_idx]
                sxj, syj, szj = σ_x_ref[ref_idx], σ_y_ref[ref_idx], σ_z_ref[ref_idx]
            end
            kldiv[j] = divfunc(xi, yi, zi, sxi, syi, szi, xj, yj, zj, sxj, syj, szj)
        end

        out += logsumexp(-kldiv) - log_k
    end

    return entropy_HD(σ_x_n, σ_y_n, σ_z_n) - out / N_n
end
