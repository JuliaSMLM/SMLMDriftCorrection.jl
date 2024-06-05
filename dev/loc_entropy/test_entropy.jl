using Revise
using CairoMakie
#using GLMakie
using FileIO
using NearestNeighbors
using Statistics
using StatsFuns
using SMLMData
using SMLMDriftCorrection

includet("cost_entropy.jl")
includet("gen_data.jl")

# Generated blinking data: 2D positions and uncertainties.
x, y, σ_x, σ_y = gendata(;n_blink = 10)
#plot(x,y)

realdata = false
if realdata
    # Real data
    dir = "Y:\\Projects\\Super Critical Angle Localization Microscopy\\Data\\10-06-2023\\Data2\\old insitu psf and stg pos"
    file = "Data2-2023-10-6-17-11-54deepfit1.jld2"
    filepath = joinpath(dir, file)
    # Load the file if not done previously
    if !isdefined(Main, :data)
        println("Loading file: $file")
        data = load(filepath) #To check keys use, varnames = keys(data)
        println("Loaded file: $file")
    end
    # Get smld
    smld3 = data["smld"]

    smld2 = smld3
    println("N_smld2 = $(length(smld2.x))")
    subind = (smld2.x .> 10.0) .& (smld2.x .< 15.0) .& (smld2.y .> 10.0) .& (smld2.y .< 15.0)
    smld2roi = SMLMData.isolatesmld(smld2, subind)
    println("N_smld2roi = $(length(smld2roi.x))")

    x, y, σ_x, σ_y = smld2roi.x, smld2roi.y, smld2roi.σ_x, smld2roi.σ_y
end

N = length(x)
println("N = ", N)
println("ub_entropy = ", ub_entropy(x, y, σ_x, σ_y))
println("entropy_HD = ", entropy_HD(σ_x, σ_y))
z = zeros(Float32, size(σ_x))
println("ub_entropy (σ = 0) = ", ub_entropy(x, y, z, z))
println("entropy_HD (σ = 0) = ", entropy_HD(z, z))

## 
# s in [-1, 1] in 201 steps of size .01
s = Float32.(range(-1,1, step = .01))
sigma_scan = zeros(length(s))
hd = zeros(length(s))

@time for i in 1:length(s)
    sx = 10f0^s[i]*σ_x
    sy = sx
    # ub_entropy is an upper bound on the entropy based on NN
    sigma_scan[i] = ub_entropy(x, y, sx, sy)
    # entropy_HD is the entropy summed over all/NN localizations
    hd[i] = entropy_HD(sx, sy)
end

s1 = Float32.(range(-1,1, step = .01))
sigma_scan1 = zeros(length(s1))
hd1 = zeros(length(s1))

@time for i in 1:length(s1)
    sx = σ_x .+ 100f0*s[i]
    sy = sx
    sigma_scan1[i] = ub_entropy(x, y, sx, sy)
    hd1[i] = entropy_HD(sx, sy)
end

## SE_Adjust-like plot
# xs = [10^(-1), 10^1]
xs = 10.0.^s
f,ax = plot(xs, sigma_scan; yscale = :identity, xlabel="x", color = :red, label = "ub")
plot!(ax, xs, hd, color = :green, label = "HD")
plot!(ax, xs, sigma_scan + 1 .* hd, color = :black, label = "ub + hd")
ax.xscale = log10
ax.xlabel = "sigma x"
ax.ylabel = "entropy"
axislegend()
#ylims!(ax,-100,100)
display(f)

## Plot on a linear scale
g,bx = plot(xs, hd, color = :green)
bx.xlabel = "b"
bx.ylabel = "entropy_HD"
display(g)

h,cx = plot(xs, sigma_scan, color = :red)
cx.xlabel = "b"
cx.ylabel = "ub"
display(h)

# xs1 = [-100, 100]
xs1 = 100.0 .* s
g1,bx1 = plot(xs1, hd1, color = :green)
bx1.xlabel = "a"
bx1.ylabel = "HD"
display(g1)

h1,cx1 = plot(xs1, sigma_scan1, color = :red)
cx1.xlabel = "a"
cx1.ylabel = "ub"
display(h1)

## 

#println(ub_entropy(x, y, 10f0*σ_x, σ_y))

#println(ub_entropy(x+10f0*randn(Float32,length(x)), y, σ_x, σ_y))
