using Revise
using JLD2
using FileIO
using SMLMData
using SMLMDriftCorrection

function convert2D(smld3::SMLMData.SMLD3D)
    smld2 = SMLMData.SMLD2D(smld3.ndatasets)
    smld2.connectID = smld3.connectID
    smld2.x = smld3.x
    smld2.y = smld3.y
    smld2.σ_x = smld3.σ_x
    smld2.σ_y = smld3.σ_y
    smld2.photons = smld3.photons
    smld2.σ_photons = smld3.σ_photons
    smld2.bg = smld3.bg
    smld2.σ_bg = smld3.σ_bg
    smld2.framenum = smld3.framenum
    smld2.datasetnum = smld3.datasetnum
    #smld2.datasize = smld3.datasize
    smld2.nframes = maximum(smld3.framenum)
    smld2.ndatasets = maximum(smld3.datasetnum)
    return smld2
end

dir = "Y:\\Projects\\Super Critical Angle Localization Microscopy\\Data\\10-06-2023\\Data2\\old insitu psf and stg pos"
file = "Data2-2023-10-6-17-11-54deepfit1.jld2"
filepath = joinpath(dir, file)
# Load the file
data = load(filepath) #To check keys use, varnames = keys(data)
# Get smld
smld3 = data["smld"]

smld2 = convert2D(smld3)

smld_DC = SMLMDriftCorrection.driftcorrect(smld2; verbose = 1, cost_fun = "Kdtree")