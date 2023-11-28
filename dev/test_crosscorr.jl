using Revise
using FileIO
using JLD2
using SMLMData
using SMLMSim

dirname = "Y:\\Projects\\Super Critical Angle Localization Microscopy\\Data\\10-06-2023\\Data2\\old insitu psf and stg pos"
file = "Data2-2023-10-6-17-11-54deepfit1.jld2"
filepath = joinpath(dirname, file)
# Load the file
data = load(filepath) #To check keys use, varnames = keys(data)
# Get smld
smld = data["smld"]
subind = smld.datasetnum .== 1
smld = SMLMData.isolatesmld(smld, subind)

#findshift2D(smld, smld)

pixelsize = 0.128 # um / pixel
#findshift3D(smld, smld; pixelsizeZunit = pixelsize)

#subind1 = collect(1 : 2 : size(smld.x, 1))
#subind2 = collect(2 : 2 : size(smld.x, 1))
subind1 = collect(1 : convert(Int, size(smld.x, 1) / 2))
subind2 = collect(convert(Int, size(smld.x, 1) / 2) + 1 : size(smld.x, 1))
smld1 = SMLMData.isolatesmld(smld, subind1)
smld2 = SMLMData.isolatesmld(smld, subind2)
#findshift2D(smld1, smld2)

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
#findshift2D(smld_true, smld_model)