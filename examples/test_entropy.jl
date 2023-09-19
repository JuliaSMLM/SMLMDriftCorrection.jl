using Revise
using SMLMDriftCorrection
DC = SMLMDriftCorrection
using CairoMakie

include("gen_data.jl")

# Generated blinking data: 2D positions and uncertainties.
x, y, σ_x, σ_y = gendata(;n_blink = 10)
#plot(x,y)

N = length(x)
println("N = ", N)
println(DC.ub_entropy(x, y, σ_x, σ_y))

## 
s = Float32.(range(-1,1, step = .01))
sigma_scan = zeros(length(s))
hd = zeros(length(s))

for i in 1:length(s)
    sx = 10f0^s[i]*σ_x
    sigma_scan[i]=DC.ub_entropy(x, y, sx, sx)
    hd[i] = DC.entropy_HD(sx,sx) 
end
## SE_Adjust-like plot
xs = 10.0.^s
f,ax = CairoMakie.plot(xs,sigma_scan; yscale = :identity, xlabel="x")
CairoMakie.plot!(ax,xs,hd)
CairoMakie.plot!(ax,xs,sigma_scan + 1 .*hd)
ax.xscale = log10
ax.xlabel = "sigma X"
ax.ylabel = "entropy"
CairoMakie.ylims!(ax,-100,100)
display(f)

##
g = CairoMakie.plot(xs,hd)
display(g)

## 

DC.ub_entropy(x, y, 10f0*σ_x, σ_y)

DC.ub_entropy(x+10f0*randn(Float32,length(x)), y, σ_x, σ_y)
