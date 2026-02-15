"""
Entropy based cost function.  See
"Drift correction in localization microscopy using entropy minimization",
Jelmer Cnossen, Tao Ju Cui, Chirlmin Joo, and Carlas Smith;
Optics Express, Volume 29, Issue 18, pp 27961-27974, 30 August 2021;
https://doi.org/10.1364/OE.620

Optimized for minimal allocations - uses scalar operations in inner loops.
"""

# Zero-allocation logsumexp for pre-allocated buffers
@inline function _logsumexp(x::AbstractVector{T}, n::Int) where T
    m = x[1]
    @inbounds for i in 2:n
        xi = x[i]
        m = ifelse(xi > m, xi, m)
    end
    s = zero(T)
    @inbounds for i in 1:n
        s += exp(x[i] - m)
    end
    return m + log(s)
end

# ============================================================================
# Entropy of localization uncertainties (H_i(D) base term)
# ============================================================================

"""
H_i(D) is the entropy of the distribution p(r), where
D = {d(1),...,d(L)} is the drift at all frames (1 to L).
The quantity below is Gaussian Mixture Model single components
summed over all localizations provided.

σ_x and σ_y are localization uncertainties.
"""
function entropy_HD(σ_x::Vector{T}, σ_y::Vector{T}) where {T<:Real}
    c = T(0.5) * log(T(2) * T(π) * T(ℯ))
    out = T(0)
    @inbounds for i in eachindex(σ_x)
        out += c + T(0.5) * log(σ_x[i] * σ_y[i])
    end
    return out
end

function entropy_HD(σ_x::Vector{T}, σ_y::Vector{T}, σ_z::Vector{T}) where {T<:Real}
    c = T(0.5) * log(T(2) * T(π) * T(ℯ))
    out = T(0)
    @inbounds for i in eachindex(σ_x)
        out += c + T(0.5) * log(σ_x[i] * σ_y[i] * σ_z[i])
    end
    return out
end

# ============================================================================
# Divergence functions - optimized scalar versions (no allocations)
# ============================================================================

"""
Kullback-Leibler divergence for 2D - scalar operations, zero allocations.
"""
@inline function divKL_2D(x1::T, y1::T, sx1::T, sy1::T,
                          x2::T, y2::T, sx2::T, sy2::T) where {T<:Real}
    si2_x = sx1^2
    si2_y = sy1^2
    sj2_x = sx2^2
    sj2_y = sy2^2

    out = log(sj2_x / si2_x) + si2_x / sj2_x + (x1 - x2)^2 / sj2_x
    out += log(sj2_y / si2_y) + si2_y / sj2_y + (y1 - y2)^2 / sj2_y
    out -= T(2)
    out /= T(2)
    return out
end

"""
Kullback-Leibler divergence for 3D - scalar operations, zero allocations.
"""
@inline function divKL_3D(x1::T, y1::T, z1::T, sx1::T, sy1::T, sz1::T,
                          x2::T, y2::T, z2::T, sx2::T, sy2::T, sz2::T) where {T<:Real}
    si2_x = sx1^2
    si2_y = sy1^2
    si2_z = sz1^2
    sj2_x = sx2^2
    sj2_y = sy2^2
    sj2_z = sz2^2

    out = log(sj2_x / si2_x) + si2_x / sj2_x + (x1 - x2)^2 / sj2_x
    out += log(sj2_y / si2_y) + si2_y / sj2_y + (y1 - y2)^2 / sj2_y
    out += log(sj2_z / si2_z) + si2_z / sj2_z + (z1 - z2)^2 / sj2_z
    out -= T(3)
    out /= T(2)
    return out
end

"""
Symmetric divergence for 2D using combined variance.
"""
@inline function divSymmetric_2D(x1::T, y1::T, sx1::T, sy1::T,
                                  x2::T, y2::T, sx2::T, sy2::T) where {T<:Real}
    si2_x = sx1^2
    si2_y = sy1^2
    sj2_x = sx2^2
    sj2_y = sy2^2

    out = log(sj2_x / si2_x) + si2_x / sj2_x + (x1 - x2)^2 / (si2_x + sj2_x)
    out += log(sj2_y / si2_y) + si2_y / sj2_y + (y1 - y2)^2 / (si2_y + sj2_y)
    out -= T(2)
    out /= T(2)
    return out
end

"""
Symmetric divergence for 3D using combined variance.
"""
@inline function divSymmetric_3D(x1::T, y1::T, z1::T, sx1::T, sy1::T, sz1::T,
                                  x2::T, y2::T, z2::T, sx2::T, sy2::T, sz2::T) where {T<:Real}
    si2_x = sx1^2
    si2_y = sy1^2
    si2_z = sz1^2
    sj2_x = sx2^2
    sj2_y = sy2^2
    sj2_z = sz2^2

    out = log(sj2_x / si2_x) + si2_x / sj2_x + (x1 - x2)^2 / (si2_x + sj2_x)
    out += log(sj2_y / si2_y) + si2_y / sj2_y + (y1 - y2)^2 / (si2_y + sj2_y)
    out += log(sj2_z / si2_z) + si2_z / sj2_z + (z1 - z2)^2 / (si2_z + sj2_z)
    out -= T(3)
    out /= T(2)
    return out
end

"""
Bhattacharyya distance for 2D.
"""
@inline function divBhattacharyya_2D(x1::T, y1::T, sx1::T, sy1::T,
                                      x2::T, y2::T, sx2::T, sy2::T) where {T<:Real}
    si2_x = sx1^2
    si2_y = sy1^2
    sj2_x = sx2^2
    sj2_y = sy2^2

    out = T(0)
    # X dimension
    var_ratio_sum = si2_x / sj2_x + sj2_x / si2_x + T(2)
    out += T(0.25) * log(T(0.25) * var_ratio_sum)
    out += T(0.25) * (x1 - x2)^2 / (si2_x + sj2_x)
    # Y dimension
    var_ratio_sum = si2_y / sj2_y + sj2_y / si2_y + T(2)
    out += T(0.25) * log(T(0.25) * var_ratio_sum)
    out += T(0.25) * (y1 - y2)^2 / (si2_y + sj2_y)

    return out
end

"""
Bhattacharyya distance for 3D.
"""
@inline function divBhattacharyya_3D(x1::T, y1::T, z1::T, sx1::T, sy1::T, sz1::T,
                                      x2::T, y2::T, z2::T, sx2::T, sy2::T, sz2::T) where {T<:Real}
    si2_x = sx1^2
    si2_y = sy1^2
    si2_z = sz1^2
    sj2_x = sx2^2
    sj2_y = sy2^2
    sj2_z = sz2^2

    out = T(0)
    for (si2, sj2, d) in ((si2_x, sj2_x, x1-x2), (si2_y, sj2_y, y1-y2), (si2_z, sj2_z, z1-z2))
        var_ratio_sum = si2 / sj2 + sj2 / si2 + T(2)
        out += T(0.25) * log(T(0.25) * var_ratio_sum)
        out += T(0.25) * d^2 / (si2 + sj2)
    end
    return out
end

"""
Mahalanobis-like distance for 2D.
"""
@inline function divMahalanobis_2D(x1::T, y1::T, sx1::T, sy1::T,
                                    x2::T, y2::T, sx2::T, sy2::T) where {T<:Real}
    return (x1 - x2)^2 / (sx1^2 + sx2^2) + (y1 - y2)^2 / (sy1^2 + sy2^2)
end

"""
Mahalanobis-like distance for 3D.
"""
@inline function divMahalanobis_3D(x1::T, y1::T, z1::T, sx1::T, sy1::T, sz1::T,
                                    x2::T, y2::T, z2::T, sx2::T, sy2::T, sz2::T) where {T<:Real}
    return (x1 - x2)^2 / (sx1^2 + sx2^2) +
           (y1 - y2)^2 / (sy1^2 + sy2^2) +
           (z1 - z2)^2 / (sz1^2 + sz2^2)
end

# ============================================================================
# Entropy computation - threaded with per-thread buffers
# ============================================================================

"""
Select divergence function by name (2D).
"""
@inline function select_divfunc_2D(divmethod::String)
    if divmethod == "KL"
        return divKL_2D
    elseif divmethod == "Symmetric"
        return divSymmetric_2D
    elseif divmethod == "Bhattacharyya"
        return divBhattacharyya_2D
    elseif divmethod == "Mahalanobis"
        return divMahalanobis_2D
    else
        error("Unknown divmethod: $divmethod")
    end
end

"""
Select divergence function by name (3D).
"""
@inline function select_divfunc_3D(divmethod::String)
    if divmethod == "KL"
        return divKL_3D
    elseif divmethod == "Symmetric"
        return divSymmetric_3D
    elseif divmethod == "Bhattacharyya"
        return divBhattacharyya_3D
    elseif divmethod == "Mahalanobis"
        return divMahalanobis_3D
    else
        error("Unknown divmethod: $divmethod")
    end
end

"""
Entropy1 for 2D - sequential with minimal allocations.

# Arguments
- idxs: Vector of neighbor indices from KNN
- x, y: Localization positions
- σ_x, σ_y: Localization uncertainties
- divmethod: Divergence method ("KL", "Symmetric", "Bhattacharyya", "Mahalanobis")
- kldiv: Pre-allocated work buffer of length >= maxn (optional)
"""
function entropy1_2D(idxs::Vector{Vector{Int}},
                     x::Vector{T}, y::Vector{T},
                     σ_x::Vector{T}, σ_y::Vector{T};
                     divmethod::String="KL",
                     kldiv::Vector{T}=Vector{T}(undef, length(idxs[1]) - 1)) where {T<:Real}

    divfunc = select_divfunc_2D(divmethod)
    maxn = length(idxs[1]) - 1
    N = length(x)
    log_maxn = log(T(maxn))

    out = T(0)

    @inbounds for i in 1:N
        idx = idxs[i]
        xi, yi = x[i], y[i]
        sxi, syi = σ_x[i], σ_y[i]

        for j in 1:maxn
            jj = idx[j+1]  # +1 because first neighbor is self
            kldiv[j] = -divfunc(xi, yi, sxi, syi, x[jj], y[jj], σ_x[jj], σ_y[jj])
        end

        out += _logsumexp(kldiv, maxn) - log_maxn
    end

    return entropy_HD(σ_x, σ_y) - out / N
end

"""
Entropy1 for 3D - sequential with minimal allocations.
"""
function entropy1_3D(idxs::Vector{Vector{Int}},
                     x::Vector{T}, y::Vector{T}, z::Vector{T},
                     σ_x::Vector{T}, σ_y::Vector{T}, σ_z::Vector{T};
                     divmethod::String="KL",
                     kldiv::Vector{T}=Vector{T}(undef, length(idxs[1]) - 1)) where {T<:Real}

    divfunc = select_divfunc_3D(divmethod)
    maxn = length(idxs[1]) - 1
    N = length(x)
    log_maxn = log(T(maxn))

    out = T(0)

    @inbounds for i in 1:N
        idx = idxs[i]
        xi, yi, zi = x[i], y[i], z[i]
        sxi, syi, szi = σ_x[i], σ_y[i], σ_z[i]

        for j in 1:maxn
            jj = idx[j+1]
            kldiv[j] = -divfunc(xi, yi, zi, sxi, syi, szi,
                               x[jj], y[jj], z[jj], σ_x[jj], σ_y[jj], σ_z[jj])
        end

        out += _logsumexp(kldiv, maxn) - log_maxn
    end

    return entropy_HD(σ_x, σ_y, σ_z) - out / N
end

# ============================================================================
# Upper bound entropy - main entry points
# ============================================================================

"""
Entropy upper bound based on maxn nearest neighbors of each localization (2D).

# Arguments
- x, y: Vectors of localization positions
- σ_x, σ_y: Vectors of localization uncertainties
- maxn: Maximum number of neighbors considered (default 200)
- divmethod: Divergence method ("KL", "Symmetric", "Bhattacharyya", "Mahalanobis")
"""
function ub_entropy(x::Vector{T}, y::Vector{T},
                    σ_x::Vector{T}, σ_y::Vector{T};
                    maxn::Int=200,
                    symmetric::Bool=false,
                    divmethod::String="") where {T<:Real}

    # Handle legacy symmetric parameter
    if isempty(divmethod)
        divmethod = symmetric ? "Symmetric" : "KL"
    end

    N = length(x)
    maxn = min(maxn, N - 1)

    # Build KDTree from 2×N matrix (dimensions as rows)
    data = Matrix{T}(undef, 2, N)
    @inbounds for i in 1:N
        data[1, i] = x[i]
        data[2, i] = y[i]
    end

    kdtree = KDTree(data; leafsize=10)
    idxs, _ = knn(kdtree, data, maxn + 1, true)

    kldiv = Vector{T}(undef, maxn)
    entropy1_2D(idxs, x, y, σ_x, σ_y; divmethod=divmethod, kldiv=kldiv)
end

"""
Entropy upper bound based on maxn nearest neighbors of each localization (3D).
"""
function ub_entropy(x::Vector{T}, y::Vector{T}, z::Vector{T},
                    σ_x::Vector{T}, σ_y::Vector{T}, σ_z::Vector{T};
                    maxn::Int=200,
                    symmetric::Bool=false,
                    divmethod::String="") where {T<:Real}

    if isempty(divmethod)
        divmethod = symmetric ? "Symmetric" : "KL"
    end

    N = length(x)
    maxn = min(maxn, N - 1)

    # Build KDTree from 3×N matrix
    data = Matrix{T}(undef, 3, N)
    @inbounds for i in 1:N
        data[1, i] = x[i]
        data[2, i] = y[i]
        data[3, i] = z[i]
    end

    kdtree = KDTree(data; leafsize=10)
    idxs, _ = knn(kdtree, data, maxn + 1, true)

    kldiv = Vector{T}(undef, maxn)
    entropy1_3D(idxs, x, y, z, σ_x, σ_y, σ_z; divmethod=divmethod, kldiv=kldiv)
end

"""
Matrix interface for ub_entropy - extracts vectors and calls optimized version.
Expects r and σ to be N×K matrices (points as rows, dimensions as columns).
"""
function ub_entropy(r::Matrix{T}, σ::Matrix{T};
                    maxn::Int=200,
                    symmetric::Bool=false,
                    divmethod::String="") where {T<:Real}
    K = size(r, 2)
    if K == 2
        x = r[:, 1]
        y = r[:, 2]
        σ_x = σ[:, 1]
        σ_y = σ[:, 2]
        ub_entropy(x, y, σ_x, σ_y; maxn=maxn, symmetric=symmetric, divmethod=divmethod)
    elseif K == 3
        x = r[:, 1]
        y = r[:, 2]
        z = r[:, 3]
        σ_x = σ[:, 1]
        σ_y = σ[:, 2]
        σ_z = σ[:, 3]
        ub_entropy(x, y, z, σ_x, σ_y, σ_z; maxn=maxn, symmetric=symmetric, divmethod=divmethod)
    else
        error("Only 2D and 3D supported, got K=$K")
    end
end
