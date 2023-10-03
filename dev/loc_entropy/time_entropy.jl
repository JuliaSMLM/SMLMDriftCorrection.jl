using Revise
using CairoMakie
using CUDA
using NearestNeighbors
using StatsFuns

includet("cost_entropy.jl")
includet("gen_data.jl")

# Generated blinking data: 2D positions and uncertainties.
x, y, σ_x, σ_y = gendata(;n_blink = 10, n_clusters = 1000)

N = length(x)
println("N = ", N)
@time println(ub_entropy(x, y, σ_x, σ_y))

#@profview ub_entropy(x, y, σ_x, σ_y)


#ub_entropy(x, y, σ_x, σ_y)

# divKL(x1::Vector{T}, s1::Vector{T},
#     x2::Vector{T}, s2::Vector{T}) where {T<:Real}

#x1 = [2.0, 3.0]
#x2 = [1.0, 1.0]
#s1 = [1.0, 1.0]
#s2 = [1.0, 1.0]

#x1gpu = CuArray([2.0, 3.0])
#x2gpu = CuArray([1.0, 1.0])
#s1gpu = CuArray([1.0, 1.0])
#s2gpu = CuArray([1.0, 1.0])

#@time begin for i in 1:100000 divKL(x1, s1, x2, s2) end end

#@time begin for i in 1:1000 divKL(x1gpu, s1gpu, x2gpu, s2gpu) end end
