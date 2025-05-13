using Revise
using CairoMakie
using SMLMData
using SMLMDriftCorrection
DC = SMLMDriftCorrection
using SMLMSim

# make an Nmer dataset
# Simulation parameters use physical units
# smld structures are in units of pixels and frames
smld_true, smld_model, smld_noisy = simulate(;
    ρ=0.1,                # emitters per μm²
    σ_psf=0.13,           # PSF width in μm (130nm)
    minphotons=50,        # minimum photons for detection
    ndatasets=10,         # number of independent datasets
    nframes=1000,         # frames per dataset
    framerate=50.0,       # frames per second
    pattern=Nmer2D(n=6, d=0.2),  # hexamer with 200nm diameter
    molecule=GenericFluor(; q=[0 50; 1e-2 0]),  # rates in 1/s
    camera=IdealCamera(1:256, 1:256, 0.1)  # pixelsize in μm
)
# Generated blinking data: 2D positions and uncertainties.
x = [e.x for e in smld_noisy.emitters]
y = [e.y for e in smld_noisy.emitters]
σ_x = [e.σ_x for e in smld_noisy.emitters]
σ_y = [e.σ_y for e in smld_noisy.emitters]

N = length(x)
println("N = ", N)
println("ub_entropy = ", DC.ub_entropy(x, y, σ_x, σ_y))
println("entropy_HD = ", DC.entropy_HD(σ_x, σ_y))
z = zeros(Float32, size(σ_x))
#println("ub_entropy (σ = 0) = ", DC.ub_entropy(x, y, z, z))
#println("entropy_HD (σ = 0) = ", DC.entropy_HD(z, z))

## 
# s in [-1, 1] in 201 steps of size .01
s = Float32.(range(-1,1, step = .01))
sigma_scan = zeros(length(s))
hd = zeros(length(s))

for i in 1:length(s)
    sx = 10f0^s[i]*σ_x
    sy = sx
    # ub_entropy is an upper bound on the entropy based on NN
    sigma_scan[i] = DC.ub_entropy(x, y, sx, sy)
    # entropy_HD is the entropy summed over all/NN localizations
    hd[i] = DC.entropy_HD(sx, sy)
end

s1 = Float32.(range(-1,1, step = .01))
sigma_scan1 = zeros(length(s1))
hd1 = zeros(length(s1))

@time for i in eachindex(s1)
    sx = σ_x .+ 100f0*s[i]
    sy = sx
    sigma_scan1[i] = DC.ub_entropy(x, y, sx, sy)
end

@time for i in eachindex(s1)
    sx = σ_x .+ 100f0*s[i]
    sy = sx
    hd1[i] = DC.entropy_HD(sx, sy)
end

## SE_Adjust-like plot
# sigma' = a + b * sigma
# xs = [10^(-1), 10^1]
xs = 10.0.^s
#f,ax = plot(xs, sigma_scan; yscale = :identity, xlabel="b", color = :red,
#                            label = "ub")
f,ax = CairoMakie.scatter(xs, sigma_scan; color = :red, label = "ub")
plot!(ax, xs, hd, color = :green, label = "HD")
plot!(ax, xs, sigma_scan - 1 .* hd, color = :black, label = "ub - HD")
ax.xscale = log10
#ax.lablel = "sigma x"
ax.xlabel = "b"
ax.ylabel = "entropy"
axislegend()
#ylims!(ax,-100,100)
display(f)

ff,aax = plot(xs, sigma_scan - 1 .* hd, color = :black)
aax.xlabel = "b"
aax.ylabel = "ub_entropy - entropy_HD"
display(ff)

## Plot on a linear scale
g,bx = plot(xs, hd, color = :green)
bx.xlabel = "b"
bx.ylabel = "entropy_HD"
display(g)

h,cx = plot(xs, sigma_scan, color = :red)
cx.xlabel = "b"
cx.ylabel = "entropy_ub"
display(h)

# xs1 = [-100, 100]
xs1 = 100.0 .* s
g1,bx1 = plot(xs1, hd1, color = :green)
bx1.xlabel = "a"
bx1.ylabel = "entropy_HD"
display(g1)

h1,cx1 = plot(xs1, sigma_scan1, color = :red)
cx1.xlabel = "a"
cx1.ylabel = "ub_entropy"
display(h1)

ff1,aax1 = plot(xs1, sigma_scan1 - 1 .* hd1, color = :black)
aax1.xlabel = "a"
aax1.ylabel = "ub_entropy - entropy_HD"
display(ff1)
