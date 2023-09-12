using NearestNeighbors
using StatsFuns

function entropy_HD(σ_x::Vector{T}, σ_y::Vector{T}) where {T<:Real}
    return sum(1 / 2 * log.(2 * pi * exp(1) * σ_y .* σ_x))*2
end


function divKL(x1::Vector{T}, s1::Vector{T},
    x2::Vector{T}, s2::Vector{T}) where {T<:Real}

    si = s1 .^ 2
    sj = s2 .^ 2
    out = 1 / 2 * sum(log.(sj ./ si) + si ./ sj + (x1 - x2) .^ 2 ./ sj) - length(x1) / 2
    return out
end

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






