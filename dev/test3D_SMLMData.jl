using Revise
using CairoMakie
using SMLMData
using SMLMDriftCorrection

path = raw"Y:\Personal Folders\MJW\tmp"
file = raw"Cell_01_Label_01_Data_2023-10-12-12-51-39_Results.mat"
smd = SmiteSMD(path, file)
smld2 = load_smite_2d(smd)

# Load the file if not done previously
if !isdefined(Main, :smld2)
    println("Loading file: $file")
    @time smld2 = load_smite_2d(smd) #To check keys use, varnames = keys(smld2)
    println("Loaded file: $file")
end

# Generate a PSF stack
#sz = 20
#roi=[(x,y,k) for x=0:sz-1,y=0:sz-1,k=0:0]
#xe = sz/2
#ye = sz/2
#pos = [(x,y,k) for x=xe:xe,y=ye:ye,k=-0.675:0.05:0.675]
#for j=eachindex(pos)
#    im = PSF.pdf(p, roi, pos[j])
#    fig, ax, plt = heatmap(im[:, :, 1])
#    zpos = pos[j][3]
#    ax.title = "PSF, z: $zpos"
#    display(fig)
#    sleep(0.1)
#    println(sum(im))
#end

# --- 2D ---

println("N_smld2 = $(length(smld2.x))")
subind = (smld2.x .> 10.0) .& (smld2.x .< 15.0) .&
         (smld2.y .> 10.0) .& (smld2.y .< 15.0)
smld2roi = SMLMData.isolatesmld(smld2, subind)
println("N_smld2 = $(length(smld2roi.x))")

smld2_DC = SMLMDriftCorrection.driftcorrect(smld2roi; verbose = 1,
                                            cost_fun = "Kdtree")

# --- 3D ---

#println("N_smld3 = $(length(smld3.x))")
#zmin = minimum(smld3.z)
#zmax = maximum(smld3.z)
#subind = (smld3.x .> 10.0) .& (smld3.x .< 15.0) .&
#         (smld3.y .> 10.0) .& (smld3.y .< 15.0) .&
#         (smld3.z .> zmin) .& (smld3.z .< zmax)
#smld3roi = SMLMData.isolatesmld(smld3, subind)
#println("N_smld3 = $(length(smld3roi.x))")

smld3_DC = SMLMDriftCorrection.driftcorrect(smld3roi; verbose = 1,
                                            cost_fun = "Kdtree")

# ----------

f = Figure()
ax11 = Axis(f[1, 1], aspect=DataAspect(), title="smld2 roi")
scatter!(smld2roi.x, smld2roi.y, markersize = 5)
ax12 = Axis(f[1, 2], aspect=DataAspect(), title="DC2")
scatter!(smld2_DC.x, smld3_DC.y, markersize = 5)
#ax21 = Axis(f[2, 1], aspect=DataAspect(), title="smld3 roi")
#scatter!(smld3roi.x, smld3roi.y, markersize = 5)
#ax22 = Axis(f[2, 2], aspect=DataAspect(), title="DC3")
#scatter!(smld3_DC.x, smld3_DC.y, markersize = 5)
linkxaxes!(ax11, ax12)
#linkxaxes!(ax11, ax21)
#linkxaxes!(ax11, ax22)
linkyaxes!(ax11, ax12)
#linkyaxes!(ax11, ax21)
#linkyaxes!(ax11, ax22)
display(f)
