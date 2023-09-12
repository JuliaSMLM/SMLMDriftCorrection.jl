using Revise
using CairoMakie

includet("entropy.jl")
includet("gen_data.jl")

x, y, σ_x, σ_y = gendata(;n_blink = 10)
plot(x,y)

length(x)
ub_entropy(x, y, σ_x, σ_y)

## 
s = Float32.(range(-1,1, step = .01))
sigma_scan = zeros(length(s))
hd = zeros(length(s))

for i in 1:length(s)
    sx = 10f0^s[i]*σ_x
    sigma_scan[i]=ub_entropy(x, y, sx, sx)
    hd[i] = entropy_HD(sx,sx) 
end
##
xs = 10.0.^s
f,ax = plot(xs,sigma_scan; yscale = :identity, xlabel="x")
plot!(ax,xs,hd)
plot!(ax,xs,sigma_scan + 1 .*hd)
ax.xscale = log10
ax.xlabel = "sigma X"
ax.ylabel = "entropy"
ylims!(ax,-100,100)
f

##
plot(xs,hd)


## 

ub_entropy(x, y, 10f0*σ_x, σ_y)

ub_entropy(x+10f0*randn(Float32,length(x)), y, σ_x, σ_y)



 















