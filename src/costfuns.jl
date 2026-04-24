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
    neighbor_indices::Matrix{Int}           # k × N matrix of neighbor indices
    last_drift_vecs::Matrix{T}              # ndims × n_test_frames drift vectors at last rebuild
    rebuild_threshold::T                    # |drift-vector delta| that triggers rebuild (μm)
    rebuild_count::Int                      # number of rebuilds (for diagnostics)
    k::Int                                  # number of neighbors
    kldiv::Vector{T}                        # pre-allocated divergence buffer (length k)
    allow_rebuild::Bool                     # when false, cost fun sees a frozen objective
    current_drift_vecs::Matrix{T}           # scratch buffer for current drift (avoid alloc)
end

# Test frames used for drift-vector comparison: start, middle, end.
const _DRIFT_TEST_FRAMES_N = 3

function NeighborState(N::Int, k::Int, rebuild_threshold::T, ndims::Int = 2) where {T<:Real}
    neighbor_indices = Matrix{Int}(undef, k, N)
    kldiv = Vector{T}(undef, k)
    # last_drift_vecs initialised to +Inf so the first rebuild check always fires
    # (delta norm = +Inf > threshold).
    last = fill(typemax(T), ndims, _DRIFT_TEST_FRAMES_N)
    current = Matrix{T}(undef, ndims, _DRIFT_TEST_FRAMES_N)
    return NeighborState{T}(neighbor_indices, last, rebuild_threshold, 0, k, kldiv, true, current)
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
        idx_i = idxs[i]
        for j in 1:k
            state.neighbor_indices[j, i] = idx_i[j+1]  # skip self at index 1
        end
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
        idx_i = idxs[i]
        for j in 1:k
            state.neighbor_indices[j, i] = idx_i[j+1]
        end
    end

    state.rebuild_count += 1
end

"""
    max_drift_magnitude(intra, nframes)

Compute maximum drift magnitude across test frames for current polynomial.
Kept for callers outside the cost-function hot path (e.g. findintra! outer
loop convergence checks) — rebuild gating itself uses vector deltas now.
"""
function max_drift_magnitude(intra::AbstractIntraDrift, nframes::Int)
    max_drift = 0.0
    for frame in (1, max(1, nframes ÷ 2), max(1, nframes))
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
    drift_vecs!(buf, intra, nframes)

Fill `buf` (ndims × _DRIFT_TEST_FRAMES_N) with drift vectors at test frames
[1, ⌊nframes/2⌋, nframes]. Using vectors (not just magnitudes) means a
direction/sign change at the same magnitude is still picked up as a change.
"""
@inline function drift_vecs!(buf::AbstractMatrix, intra::AbstractIntraDrift, nframes::Int)
    f1 = 1
    f2 = max(1, nframes ÷ 2)
    f3 = max(1, nframes)
    @inbounds for dim in 1:intra.ndims
        buf[dim, 1] = evaluate_at_frame(intra.dm[dim], f1)
        buf[dim, 2] = evaluate_at_frame(intra.dm[dim], f2)
        buf[dim, 3] = evaluate_at_frame(intra.dm[dim], f3)
    end
    return buf
end

"""
    max_drift_vec_delta(current, last)

Max Euclidean norm of column-wise differences between two ndims × n_test_frames
drift-vector matrices. Triggers a rebuild if this exceeds rebuild_threshold
even when the scalar magnitude max_drift_magnitude is unchanged (e.g. rotation
of the drift vector at the midpoint frame).
"""
@inline function max_drift_vec_delta(current::AbstractMatrix{T}, last::AbstractMatrix{T}) where {T<:Real}
    max_delta = zero(T)
    n_dim, n_test = size(current)
    @inbounds for col in 1:n_test
        d2 = zero(T)
        for row in 1:n_dim
            d = current[row, col] - last[row, col]
            d2 += d * d
        end
        s = sqrt(d2)
        max_delta = s > max_delta ? s : max_delta
    end
    return max_delta
end

"""
    maybe_rebuild_neighbors!(state, x_work, y_work, intra, nframes)

Check if neighbors need rebuilding based on drift-vector change. Uses the
ndims × n_test_frames vector representation so direction changes at the same
magnitude still trigger a rebuild.
"""
function maybe_rebuild_neighbors!(state::NeighborState{T},
                                   x_work::Vector{T}, y_work::Vector{T},
                                   intra::AbstractIntraDrift, nframes::Int) where {T<:Real}
    state.allow_rebuild || return
    drift_vecs!(state.current_drift_vecs, intra, nframes)

    if max_drift_vec_delta(state.current_drift_vecs, state.last_drift_vecs) > state.rebuild_threshold
        build_neighbors!(state, x_work, y_work)
        state.last_drift_vecs .= state.current_drift_vecs
    end
end

"""
    maybe_rebuild_neighbors!(state, x_work, y_work, z_work, intra, nframes)

3D version of maybe_rebuild_neighbors!
"""
function maybe_rebuild_neighbors!(state::NeighborState{T},
                                   x_work::Vector{T}, y_work::Vector{T}, z_work::Vector{T},
                                   intra::AbstractIntraDrift, nframes::Int) where {T<:Real}
    state.allow_rebuild || return
    drift_vecs!(state.current_drift_vecs, intra, nframes)

    if max_drift_vec_delta(state.current_drift_vecs, state.last_drift_vecs) > state.rebuild_threshold
        build_neighbors!(state, x_work, y_work, z_work)
        state.last_drift_vecs .= state.current_drift_vecs
    end
end

# ============================================================================
# Adaptive Entropy cost functions (self-excluded neighbor indices)
# ============================================================================

"""
    entropy1_2D_adaptive(neighbor_indices, x, y, σ_x, σ_y, kldiv; divmethod)

Compute entropy using pre-computed neighbor indices (self excluded).
Uses matrix-stored neighbor indices (k × N) and pre-allocated kldiv buffer.
"""
function entropy1_2D_adaptive(neighbor_indices::Matrix{Int}, k::Int,
                               x::Vector{T}, y::Vector{T},
                               σ_x::Vector{T}, σ_y::Vector{T},
                               kldiv::Vector{T}) where {T<:Real}
    N = length(x)

    if k < 1
        return entropy_HD(σ_x, σ_y)
    end

    log_k = log(T(k))
    out = T(0)

    @inbounds for i in 1:N
        xi, yi = x[i], y[i]
        sxi, syi = σ_x[i], σ_y[i]

        for j in 1:k
            jj = neighbor_indices[j, i]
            kldiv[j] = -divKL_2D(xi, yi, sxi, syi, x[jj], y[jj], σ_x[jj], σ_y[jj])
        end

        out += _logsumexp(kldiv, k) - log_k
    end

    return entropy_HD(σ_x, σ_y) - out / N
end

"""
    entropy1_3D_adaptive(neighbor_indices, k, x, y, z, σ_x, σ_y, σ_z, kldiv; divmethod)

3D version of entropy1_2D_adaptive.
"""
function entropy1_3D_adaptive(neighbor_indices::Matrix{Int}, k::Int,
                               x::Vector{T}, y::Vector{T}, z::Vector{T},
                               σ_x::Vector{T}, σ_y::Vector{T}, σ_z::Vector{T},
                               kldiv::Vector{T}) where {T<:Real}
    N = length(x)

    if k < 1
        return entropy_HD(σ_x, σ_y, σ_z)
    end

    log_k = log(T(k))
    out = T(0)

    @inbounds for i in 1:N
        xi, yi, zi = x[i], y[i], z[i]
        sxi, syi, szi = σ_x[i], σ_y[i], σ_z[i]

        for j in 1:k
            jj = neighbor_indices[j, i]
            kldiv[j] = -divKL_3D(xi, yi, zi, sxi, syi, szi,
                               x[jj], y[jj], z[jj], σ_x[jj], σ_y[jj], σ_z[jj])
        end

        out += _logsumexp(kldiv, k) - log_k
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
                                            x_work::Vector{T}=similar(x),
                                            y_work::Vector{T}=similar(y)) where {T<:Real}
    theta2intra!(intra, θ)
    N = length(x)

    @inbounds for i in 1:N
        x_work[i] = correctdrift(x[i], framenum[i], intra.dm[1])
        y_work[i] = correctdrift(y[i], framenum[i], intra.dm[2])
    end

    maybe_rebuild_neighbors!(state, x_work, y_work, intra, nframes)

    return entropy1_2D_adaptive(state.neighbor_indices, state.k,
                                 x_work, y_work, σ_x, σ_y, state.kldiv)
end

"""
INTRA-ENTROPY with adaptive neighbor rebuilding (3D)
"""
function costfun_entropy_intra_3D_adaptive(θ, x::Vector{T}, y::Vector{T}, z::Vector{T},
                                            σ_x::Vector{T}, σ_y::Vector{T}, σ_z::Vector{T},
                                            framenum::Vector{Int}, maxn::Int,
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

    return entropy1_3D_adaptive(state.neighbor_indices, state.k,
                                 x_work, y_work, z_work,
                                 σ_x, σ_y, σ_z, state.kldiv)
end

# ============================================================================
# Intra-dataset entropy cost — merged-cloud variant for iteration 2+
# ============================================================================
# Probe: dataset_n points corrected by the intra polynomial under optimization.
# Reference: all other datasets, already fully drift-corrected (inter + intra
#            applied before findintra! is called, so they're a fixed scaffold
#            in the common frame).
# This mirrors costfun_entropy_inter_2D_merged — the KDTree is built over the
# combined (probe ∪ reference) cloud and the probe's nearest neighbors can be
# either other probe points (coords vary with θ) or reference points (fixed).
# Using the merged cloud stops intra ↔ inter from chasing each other in a
# limit cycle: intra now sees the same structural scaffold that findinter! sees.
#
# Rebuild is gated on state.allow_rebuild — findintra! freezes the state
# during each optimize() call and drives rebuilds from an outer loop.
# Caller must set state.last_rebuild_drift = Inf before the first call so
# the initial build fires.

"""
INTRA-ENTROPY merged-cloud (2D) — iteration 2+ intra fit against all datasets.

Ref points are **fixed** during optimize() (they're the other datasets' already
fully-corrected coords). The caller (findintra!) fills `data_combined[:, N_n+1:end]`
ONCE before the optimize loop; this function only rewrites the probe columns
(1:N_n) each evaluation. σ_x_ref / σ_y_ref must align with the ref slots of
data_combined.
"""
function costfun_entropy_intra_2D_merged(θ,
    x_n::Vector{T}, y_n::Vector{T}, σ_x_n::Vector{T}, σ_y_n::Vector{T},
    framenum_n::Vector{Int},
    σ_x_ref::Vector{T}, σ_y_ref::Vector{T},
    maxn::Int, intra::AbstractIntraDrift, nframes::Int,
    data_combined::Matrix{T},
    state::NeighborState{T};
    x_work::Vector{T} = similar(x_n),
    y_work::Vector{T} = similar(y_n)) where {T<:Real}

    theta2intra!(intra, θ)
    N_n = length(x_n)
    N_ref = length(σ_x_ref)
    N_combined = N_n + N_ref

    @inbounds for i in 1:N_n
        x_work[i] = correctdrift(x_n[i], framenum_n[i], intra.dm[1])
        y_work[i] = correctdrift(y_n[i], framenum_n[i], intra.dm[2])
        data_combined[1, i] = x_work[i]
        data_combined[2, i] = y_work[i]
        # ref columns (N_n+1 : end) were filled once by the caller and stay put
    end

    k = min(maxn, N_combined - 1)

    # Rebuild gating: vector delta of drift at [1, mid, end] frames. Catches
    # same-magnitude direction/sign changes that the scalar max_drift_magnitude
    # comparison would miss.
    need_rebuild = false
    if state.allow_rebuild
        drift_vecs!(state.current_drift_vecs, intra, nframes)
        need_rebuild = max_drift_vec_delta(state.current_drift_vecs, state.last_drift_vecs) > state.rebuild_threshold
    end

    if need_rebuild
        kdtree = KDTree(data_combined; leafsize = 10)
        query_points = view(data_combined, :, 1:N_n)
        idxs, _ = knn(kdtree, query_points, k + 1, true)
        @inbounds for i in 1:N_n
            idx_i = idxs[i]
            for j in 1:k
                state.neighbor_indices[j, i] = idx_i[j + 1]  # skip self
            end
        end
        state.last_drift_vecs .= state.current_drift_vecs
        state.rebuild_count += 1
    end

    log_k = log(T(k))
    out = T(0)

    @inbounds for i in 1:N_n
        xi, yi = x_work[i], y_work[i]
        sxi, syi = σ_x_n[i], σ_y_n[i]

        for j in 1:k
            jj = state.neighbor_indices[j, i]
            if jj <= N_n
                xj, yj = x_work[jj], y_work[jj]
                sxj, syj = σ_x_n[jj], σ_y_n[jj]
            else
                ref_idx = jj - N_n
                xj = data_combined[1, jj]
                yj = data_combined[2, jj]
                sxj, syj = σ_x_ref[ref_idx], σ_y_ref[ref_idx]
            end
            state.kldiv[j] = -divKL_2D(xi, yi, sxi, syi, xj, yj, sxj, syj)
        end

        out += _logsumexp(state.kldiv, k) - log_k
    end

    return entropy_HD(σ_x_n, σ_y_n) - out / N_n
end

"""
INTRA-ENTROPY merged-cloud (3D) — iteration 2+ intra fit against all datasets.

Ref columns of `data_combined` filled once by caller (see 2D variant's docstring).
"""
function costfun_entropy_intra_3D_merged(θ,
    x_n::Vector{T}, y_n::Vector{T}, z_n::Vector{T},
    σ_x_n::Vector{T}, σ_y_n::Vector{T}, σ_z_n::Vector{T},
    framenum_n::Vector{Int},
    σ_x_ref::Vector{T}, σ_y_ref::Vector{T}, σ_z_ref::Vector{T},
    maxn::Int, intra::AbstractIntraDrift, nframes::Int,
    data_combined::Matrix{T},
    state::NeighborState{T};
    x_work::Vector{T} = similar(x_n),
    y_work::Vector{T} = similar(y_n),
    z_work::Vector{T} = similar(z_n)) where {T<:Real}

    theta2intra!(intra, θ)
    N_n = length(x_n)
    N_ref = length(σ_x_ref)
    N_combined = N_n + N_ref

    @inbounds for i in 1:N_n
        x_work[i] = correctdrift(x_n[i], framenum_n[i], intra.dm[1])
        y_work[i] = correctdrift(y_n[i], framenum_n[i], intra.dm[2])
        z_work[i] = correctdrift(z_n[i], framenum_n[i], intra.dm[3])
        data_combined[1, i] = x_work[i]
        data_combined[2, i] = y_work[i]
        data_combined[3, i] = z_work[i]
        # ref columns (N_n+1 : end) were filled once by caller
    end

    k = min(maxn, N_combined - 1)

    need_rebuild = false
    if state.allow_rebuild
        drift_vecs!(state.current_drift_vecs, intra, nframes)
        need_rebuild = max_drift_vec_delta(state.current_drift_vecs, state.last_drift_vecs) > state.rebuild_threshold
    end

    if need_rebuild
        kdtree = KDTree(data_combined; leafsize = 10)
        query_points = view(data_combined, :, 1:N_n)
        idxs, _ = knn(kdtree, query_points, k + 1, true)
        @inbounds for i in 1:N_n
            idx_i = idxs[i]
            for j in 1:k
                state.neighbor_indices[j, i] = idx_i[j + 1]
            end
        end
        state.last_drift_vecs .= state.current_drift_vecs
        state.rebuild_count += 1
    end

    log_k = log(T(k))
    out = T(0)

    @inbounds for i in 1:N_n
        xi, yi, zi = x_work[i], y_work[i], z_work[i]
        sxi, syi, szi = σ_x_n[i], σ_y_n[i], σ_z_n[i]

        for j in 1:k
            jj = state.neighbor_indices[j, i]
            if jj <= N_n
                xj, yj, zj = x_work[jj], y_work[jj], z_work[jj]
                sxj, syj, szj = σ_x_n[jj], σ_y_n[jj], σ_z_n[jj]
            else
                ref_idx = jj - N_n
                xj = data_combined[1, jj]
                yj = data_combined[2, jj]
                zj = data_combined[3, jj]
                sxj, syj, szj = σ_x_ref[ref_idx], σ_y_ref[ref_idx], σ_z_ref[ref_idx]
            end
            state.kldiv[j] = -divKL_3D(xi, yi, zi, sxi, syi, szi,
                                       xj, yj, zj, sxj, syj, szj)
        end

        out += _logsumexp(state.kldiv, k) - log_k
    end

    return entropy_HD(σ_x_n, σ_y_n, σ_z_n) - out / N_n
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
    neighbor_indices::Matrix{Int}           # k × N_n matrix of neighbor indices
    last_shift::Vector{T}                   # shift at last rebuild
    rebuild_threshold::T                    # shift change that triggers rebuild
    rebuild_count::Int                      # for diagnostics
    k::Int                                  # number of neighbors
    kldiv::Vector{T}                        # pre-allocated divergence buffer
    needs_initial_build::Bool               # true until first build
    allow_rebuild::Bool                     # when false, cost fun sees a frozen objective
end

function InterNeighborState(N_n::Int, k::Int, rebuild_threshold::T) where {T<:Real}
    neighbor_indices = Matrix{Int}(undef, k, N_n)
    kldiv = Vector{T}(undef, k)
    return InterNeighborState{T}(neighbor_indices, T[Inf, Inf], rebuild_threshold, 0, k, kldiv, true, true)
end

function InterNeighborState3D(N_n::Int, k::Int, rebuild_threshold::T) where {T<:Real}
    neighbor_indices = Matrix{Int}(undef, k, N_n)
    kldiv = Vector{T}(undef, k)
    return InterNeighborState{T}(neighbor_indices, T[Inf, Inf, Inf], rebuild_threshold, 0, k, kldiv, true, true)
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
    x_work::Vector{T}=similar(x_n),
    y_work::Vector{T}=similar(y_n),
    data_combined::Matrix{T}=Matrix{T}(undef, 2, length(x_n)+length(x_ref)),
    state::Union{InterNeighborState{T}, Nothing}=nothing) where {T<:Real}

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
    @inbounds for i in 1:N_n
        data_combined[1, i] = x_work[i]
        data_combined[2, i] = y_work[i]
    end
    @inbounds for i in 1:N_ref
        data_combined[1, N_n + i] = x_ref[i]
        data_combined[2, N_n + i] = y_ref[i]
    end

    k = min(maxn, N_combined - 1)

    # Check if we need to rebuild neighbors.
    # Initial build is always allowed; subsequent rebuilds only when state.allow_rebuild.
    # findinter! freezes the state during optimize() and drives rebuilds from an outer loop
    # to keep BFGS's view of the objective deterministic.
    need_rebuild = state === nothing ||
                   state.needs_initial_build ||
                   (state.allow_rebuild &&
                    sqrt((θ[1] - state.last_shift[1])^2 + (θ[2] - state.last_shift[2])^2) > state.rebuild_threshold)

    local idxs  # for stateless fallback
    if need_rebuild
        kdtree = KDTree(data_combined; leafsize=10)
        query_points = view(data_combined, :, 1:N_n)
        idxs, _ = knn(kdtree, query_points, k + 1, true)

        if state !== nothing
            @inbounds for i in 1:N_n
                idx_i = idxs[i]
                for j in 1:k
                    state.neighbor_indices[j, i] = idx_i[j+1]  # skip self
                end
            end
            state.last_shift[1] = θ[1]
            state.last_shift[2] = θ[2]
            state.rebuild_count += 1
            state.needs_initial_build = false
        end
    end

    # Compute entropy contribution from dataset_n points only
    log_k = log(T(k))
    kldiv_buf = state !== nothing ? state.kldiv : Vector{T}(undef, k)
    out = T(0)

    @inbounds for i in 1:N_n
        xi, yi = x_work[i], y_work[i]
        sxi, syi = σ_x_n[i], σ_y_n[i]

        for j in 1:k
            jj = state !== nothing ? state.neighbor_indices[j, i] : idxs[i][j+1]
            if jj <= N_n
                xj, yj = x_work[jj], y_work[jj]
                sxj, syj = σ_x_n[jj], σ_y_n[jj]
            else
                ref_idx = jj - N_n
                xj, yj = x_ref[ref_idx], y_ref[ref_idx]
                sxj, syj = σ_x_ref[ref_idx], σ_y_ref[ref_idx]
            end
            kldiv_buf[j] = -divKL_2D(xi, yi, sxi, syi, xj, yj, sxj, syj)
        end

        out += _logsumexp(kldiv_buf, k) - log_k
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
    x_work::Vector{T}=similar(x_n),
    y_work::Vector{T}=similar(y_n),
    z_work::Vector{T}=similar(z_n),
    data_combined::Matrix{T}=Matrix{T}(undef, 3, length(x_n)+length(x_ref)),
    state::Union{InterNeighborState{T}, Nothing}=nothing) where {T<:Real}

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

    # Check if we need to rebuild neighbors.
    # Initial build is always allowed; subsequent rebuilds only when state.allow_rebuild.
    # findinter! freezes the state during optimize() and drives rebuilds from an outer loop
    # to keep BFGS's view of the objective deterministic.
    need_rebuild = state === nothing ||
                   state.needs_initial_build ||
                   (state.allow_rebuild &&
                    sqrt((θ[1] - state.last_shift[1])^2 + (θ[2] - state.last_shift[2])^2 + (θ[3] - state.last_shift[3])^2) > state.rebuild_threshold)

    local idxs
    if need_rebuild
        kdtree = KDTree(data_combined; leafsize=10)
        query_points = view(data_combined, :, 1:N_n)
        idxs, _ = knn(kdtree, query_points, k + 1, true)

        if state !== nothing
            @inbounds for i in 1:N_n
                idx_i = idxs[i]
                for j in 1:k
                    state.neighbor_indices[j, i] = idx_i[j+1]
                end
            end
            state.last_shift[1] = θ[1]
            state.last_shift[2] = θ[2]
            state.last_shift[3] = θ[3]
            state.rebuild_count += 1
            state.needs_initial_build = false
        end
    end

    # Compute entropy contribution from dataset_n points only
    log_k = log(T(k))
    kldiv_buf = state !== nothing ? state.kldiv : Vector{T}(undef, k)
    out = T(0)

    @inbounds for i in 1:N_n
        xi, yi, zi = x_work[i], y_work[i], z_work[i]
        sxi, syi, szi = σ_x_n[i], σ_y_n[i], σ_z_n[i]

        for j in 1:k
            jj = state !== nothing ? state.neighbor_indices[j, i] : idxs[i][j+1]
            if jj <= N_n
                xj, yj, zj = x_work[jj], y_work[jj], z_work[jj]
                sxj, syj, szj = σ_x_n[jj], σ_y_n[jj], σ_z_n[jj]
            else
                ref_idx = jj - N_n
                xj, yj, zj = x_ref[ref_idx], y_ref[ref_idx], z_ref[ref_idx]
                sxj, syj, szj = σ_x_ref[ref_idx], σ_y_ref[ref_idx], σ_z_ref[ref_idx]
            end
            kldiv_buf[j] = -divKL_3D(xi, yi, zi, sxi, syi, szi, xj, yj, zj, sxj, syj, szj)
        end

        out += _logsumexp(kldiv_buf, k) - log_k
    end

    return entropy_HD(σ_x_n, σ_y_n, σ_z_n) - out / N_n
end
