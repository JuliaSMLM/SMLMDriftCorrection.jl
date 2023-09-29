using Revise
using CairoMakie

includet("entropy_play.jl")
includet("gen_data.jl")

# Generated blinking data: 2D positions and uncertainties.
x, y, σ_x, σ_y = gendata(;n_blink = 10)

N = length(x)
println("N = ", N)
@time println(ub_entropy(x, y, σ_x, σ_y))
