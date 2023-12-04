using Revise
using FileIO
using JLD2
using SMLMData
using SMLMSim

includet("crosscorr.jl")

# Simulation paramters use physical units
# smld structures are in units of pixels and frames 
smld_true, smld_model, smld_noisy = SMLMSim.sim(;
    ρ=1.0,
    σ_PSF=0.13, #micron 
    minphotons=50,
    ndatasets=10,
    nframes=1000,
    framerate=50.0, # 1/s
    pattern=SMLMSim.Nmer2D(),
    molecule=SMLMSim.GenericFluor(; q=[0 50; 1e-2 0]), #1/s 
    camera=SMLMSim.IdealCamera(; xpixels=256, ypixels=256, pixelsize=0.1) #pixelsize is microns
)
begin
println("N = $(size(smld_noisy.x, 1))")
println(findshift2D(smld_noisy, smld_noisy; histbinsize=0.25))
smldn = deepcopy(smld_noisy)
smldn.x .+= 4.3
smldn.y .+= -2.8
smldn.x .= max.(0, min.(smldn.x, 256))
smldn.y .= max.(0, min.(smldn.y, 256))
println(findshift2D(smld_noisy, smldn; histbinsize=0.25))
end

dirname = "Y:\\Projects\\Super Critical Angle Localization Microscopy\\Data\\10-06-2023\\Data2\\old insitu psf and stg pos"
file = "Data2-2023-10-6-17-11-54deepfit1.jld2"
filepath = joinpath(dirname, file)
# Load the file
data = load(filepath) #To check keys use, varnames = keys(data)

begin
# Get smld
smld = data["smld"]
subind = smld.datasetnum .== 1
smld = SMLMData.isolatesmld(smld, subind)
smld.datasize = [255, 255, 1] # making the 1st entry 256 causes weirdness

pixelsize = 0.128 # um / pixel
println(findshift3D(smld, smld; pixelsizeZunit=pixelsize))

subind1 = smld.framenum .<= 1000
subind2 = smld.framenum .> 1000
smld1 = SMLMData.isolatesmld(smld, subind1)
# smld2 = SMLMData.isolatesmld(smld, subind2)
smld2 = SMLMData.isolatesmld(smld, subind1) #to give identical datasets
println("N = $(size(smld1.x, 1))")
smld2.x .+= 4.3
smld2.y .+= -2.8
smld2.z .+= 0.2
smld2.x .= max.(0, min.(smld2.x, 256))
smld2.y .= max.(0, min.(smld2.y, 256))
println(findshift2D(smld1, smld2; histbinsize=1.0))
findshift3D(smld1, smld2; pixelsizeZunit=pixelsize, histbinsize=0.25)
end