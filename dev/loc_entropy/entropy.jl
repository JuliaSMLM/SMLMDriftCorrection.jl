using NearestNeighbors
using StatsFuns

"""
Entropy based cost function.  See 
"Drift correction in localization microscopy using entropy minimization",
Jelmer Cnossen, Tao Ju Cui, Chirlmin Joo, and Carlas Smith;
Optics Express, Volume 29, Issue 18, pp 27961-27974, 30 August 2021;
https://doi.org/10.1364/OE.426620

"""

"""
H_i(D) is the entropy of the distribution p(r), where
D = {d(1),...,d(L)} is the drift at all frames (1 to L).
The quantity below is Gaussian Mixture Model single components
summed over all localizations provided (may not be the total set of
localizations if limited through nearest neighbors).

σ_x and σ_y are localization uncertainties.
"""
function entropy_HD(σ_x::Vector{T}, σ_y::Vector{T}) where {T<:Real}
    return sum(1 / 2 * log.(2 * pi * exp(1) * σ_y .* σ_x))
end

"""
Kullback-Keibler divergence between the probability distributions for
localization uncertainties 1 and 2, where si2 = s1^2 and sj2 = s2^2.
x1 and x2 are the differences (1 and 2) between the estimated positions
of each localization and the corresponding drift in the frame at time
t_1, t_2, respectively.
"""
function divKL(x1::Vector{T}, s1::Vector{T},
    x2::Vector{T}, s2::Vector{T}) where {T<:Real}

    si2 = s1 .^ 2
    sj2 = s2 .^ 2
    # K is the dimension of the space, typically 2.
    K = length(x1)
    out = 1 / 2 * sum(log.(sj2 ./ si2) + si2 ./ sj2 + (x1 - x2) .^ 2 ./ sj2) - K / 2
    return out
end

"""
Entropy computation using maxn nearest neighbors for each
    localization.

# Fields
- idxs       matrix of indices (maxn x K)
- x, y       vectors of localization positions
- σ_x, σ_y   vectors of localization uncertainties
"""
function entropy1(idxs::Vector{Vector{Int}}, x::Vector{T}, y::Vector{T},
    σ_x::Vector{T}, σ_y::Vector{T}) where {T<:Real}

    out = 0.0
    # maxn is the maxinum number of neighbors allowed.
    maxn = length(idxs[1]) - 1

    kldiv = zeros(Float32, maxn)
    #Threads.@threads 
    # length(x) number of localizations
    # maxn      number of neighbors per localizations
    # NOTE: kldiv is reused for each i
    for i = 1:length(x)
        for j = 2:maxn + 1
            idx = idxs[i]
            r1 = [x[i], y[i]]
            r2 = [x[idx[j]], y[idx[j]]]
            σ1 = [σ_x[i], σ_y[i]]
            σ2 = [σ_x[idx[j]], σ_y[idx[j]]]
            kldiv[j-1] = divKL(r1, σ1, r2, σ2)
        end

        out += logsumexp(- kldiv) - log(length(kldiv))

    end

    
    N = length(x)   # total number of localizations
    return entropy_HD(σ_x, σ_y) - out / N
end

"""
Entropy upper bound based on maxn nearest neighbors of each localization.

# Fields
- x, y       vector of localization positions
- σ_x, σ_y   vector of localization uncertainties
"""
function ub_entropy(x::Vector{T}, y::Vector{T},
    σ_x::Vector{T}, σ_y::Vector{T},
) where {T<:Real}

    #println("Calculating KDTree...")
    coords = cat(dims=2, x, y)

    data = transpose(coords)

    maxn = 200
    kdtree = KDTree(data; leafsize=10)
    # true below so that results are sorted into increasing order of distance
    idxs, dists = knn(kdtree, data, maxn + 1, true)
    #idxs = inrange(kdtree, data, maxn + 1)
    #println("Calculating Entropy...")
    return entropy1(idxs, x, y, σ_x, σ_y)

end
