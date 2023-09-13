using NearestNeighbors
using StatsFuns

"""
H(D) is the entropy of the distribution p(r), where
D = {d(1),...,d(L)} is the drift at all frames (1 to L).
"""
function entropy_HD(σ_x::Vector{T}, σ_y::Vector{T}) where {T<:Real}
    return sum(1 / 2 * log.(2 * pi * exp(1) * σ_y .* σ_x))
end

"""
Kullback-Keibler divergence between the probability distributions for
localization uncertainties 1 and 2, where si = s1^2 and sj = s2^2.
x1 and x2 are the differences (1 and 2) between the estimated positions
of each localization and the corresponding drift in frame at time t_1 / t_2.
"""
function divKL(x1::Vector{T}, s1::Vector{T},
    x2::Vector{T}, s2::Vector{T}) where {T<:Real}

    si = s1 .^ 2
    sj = s2 .^ 2
    out = 1 / 2 * sum(log.(sj ./ si) + si ./ sj + (x1 - x2) .^ 2 ./ sj) - length(x1) / 2
    return out
end

"""
Entropy.

# Fields
- idxs       matrix of indices
- x, y       vector of localization positions
- σ_x, σ_y   vector of localization uncertainties
"""
function entropy1(idxs::Vector{Vector{Int}}, x::Vector{T}, y::Vector{T},
    σ_x::Vector{T}, σ_y::Vector{T}) where {T<:Real}

    out = 0.0
    maxn = length(idxs[1])

    kldiv = zeros(Float32, maxn - 1)
    #Threads.@threads 
    for i = 1:length(x)
        for j = 2:maxn
            idx = idxs[i]
            r1 = [x[i], y[i]]
            r2 = [x[idx[j]], y[idx[j]]]
            σ1 = [σ_x[i], σ_y[i]]
            σ2 = [σ_x[idx[j]], σ_y[idx[j]]]
            kldiv[j-1] = divKL(r1, σ1, r2, σ2)
        end

        out += logsumexp(kldiv) - log(length(kldiv))

    end
    return out / length(x) - entropy_HD(σ_x, σ_y)
end

function ub_entropy(x::Vector{T}, y::Vector{T},
    σ_x::Vector{T}, σ_y::Vector{T},
) where {T<:Real}

    #println("Calculating KDTree...")
    coords = cat(dims=2, x, y)

    data = transpose(coords)

    maxn = 200
    kdtree = KDTree(data; leafsize=10)
    idxs, dists = knn(kdtree, data, maxn + 1, true)
    #println("Calculating Entropy...")
    return entropy1(idxs, x, y, σ_x, σ_y)

end
