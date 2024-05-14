using Revise
using CairoMakie
using NearestNeighbors
using Statistics
using StatsFuns
using SMLMDriftCorrection

#includet("cost_entropy.jl")
includet("gen_data.jl")

# Generated blinking data: 2D positions and uncertainties.
x, y, σ_x, σ_y = gendata(;n_blink = 10)
#plot(x,y)

N = length(x)
println("N = ", N)
println("ub_entropy = ", ub_entropy(x, y, σ_x, σ_y))

## 
# s in [-1, 1] in 201 steps of size .01
s = Float32.(range(-1,1, step = .01))
sigma_scan = zeros(length(s))
hd = zeros(length(s))

for i in 1:length(s)
    sx = 10f0^s[i]*σ_x
    sy = sx
    # ub_entropy is an upper bound on the entropy based on NN
    sigma_scan[i] = ub_entropy(x, y, sx, sy)
    # entropy_HD is the entropy summed over all/NN localizations
    hd[i] = entropy_HD(sx, sy) 
end
## SE_Adjust-like plot
# xs = [10^(-1), 10^1]
xs = 10.0.^s
f,ax = plot(xs, sigma_scan; yscale = :identity, xlabel="x", color = :black, label = "ub")
plot!(ax, xs, hd, color = :green, label = "hd")
plot!(ax, xs, sigma_scan + 1 .* hd, color = :red, label = "ub + hd")
ax.xscale = log10
ax.xlabel = "sigma x"
ax.ylabel = "entropy"
axislegend()
#ylims!(ax,-100,100)
display(f)

## Plot on a linear scale
g,bx = plot(xs, hd, color = :green)
bx.ylabel = "hd"
display(g)

h,cx = plot(xs, sigma_scan + 1 .* hd, color = :red)
cx.ylabel = "ub + hd"
display(h)

## 

println(ub_entropy(x, y, 10f0*σ_x, σ_y))

println(ub_entropy(x+10f0*randn(Float32,length(x)), y, σ_x, σ_y))
