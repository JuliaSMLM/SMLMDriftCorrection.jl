"""
Cost functions for drift correction optimization.
Optimized to avoid unnecessary allocations.
"""

# ============================================================================
# KDTree-based cost functions (KNN approach)
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
