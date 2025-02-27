using Revise
using JLD2
using FileIO
using GLMakie
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
    smld2.ndatasets = smld3.ndatasets
    smld2.nframes = smld3.nframes
    if smld2.ndatasets == 0
        smld2.ndatasets = maximum(smld2.datasetnum)
    end
    if smld2.nframes == 0
        smld2.nframes = maximum(smld2.framenum)
    end
    return smld2
end

function convert_SMLD2D_to_Emitter2DFit(smld2::SMLMData.SMLD2D)
    smld2_fit = Emitter2DFit{Float64}(smld2.x, smld2.y, smld2.photons,
                    smld2.bg, smld2.σ_x, smld2.σ_y, smld2.σ_photons,
                    smld2.σ_bg; frame=smld2.framenum, dataset=smld2.datasetnum,
                    track_id=smld2.connectID)
    return smld2_fit
end

dir = "Y:\\Projects\\Super Critical Angle Localization Microscopy\\Data\\10-06-2023\\Data2\\old insitu psf and stg pos"
file = "Data2-2023-10-6-17-11-54deepfit1.jld2"
filepath = joinpath(dir, file)
# Load the file if not done previously
if !isdefined(Main, :data)
    println("Loading file: $file")
    @time data = load(filepath) #To check keys use, varnames = keys(data)
    println("Loaded file: $file")
end
# Get smld
smld3 = data["smld"]
if smld3.ndatasets == 0
    smld3.ndatasets = maximum(smld3.datasetnum)
end
if smld3.nframes == 0
    smld3.nframes = maximum(smld3.framenum)
end
smld2 = convert2D(smld3)
smld2_fix = convert_SMLD2D_to_Emitter2DFit(smld2)
println("N_smld2 = $(length(smld2.x))")
subind = (smld2.x .> 10.0) .& (smld2.x .< 15.0) .& (smld2.y .> 10.0) .& (smld2.y .< 15.0)
smld2roi = SMLMData.isolatesmld(smld2, subind)
println("N_smld2 = $(length(smld2roi.x))")

smld2_DC = SMLMDriftCorrection.driftcorrect(smld2roi; verbose = 1, cost_fun = "Kdtree")

println("N_smld3 = $(length(smld3.x))")
zmin = minimum(smld3.z)
zmax = maximum(smld3.z)
subind = (smld3.x .> 10.0) .& (smld3.x .< 15.0) .& (smld3.y .> 10.0) .& (smld3.y .< 15.0) .& (smld3.z .> zmin) .& (smld3.z .< zmax)
smld3roi = SMLMData.isolatesmld(smld3, subind)
println("N_smld3 = $(length(smld3roi.x))")

smld3_DC = SMLMDriftCorrection.driftcorrect(smld3roi; verbose = 1, cost_fun = "Kdtree")

f = Figure()
ax11 = Axis(f[1, 1], aspect=DataAspect(), title="smld2 roi")
scatter!(smld2roi.x, smld2roi.y, markersize = 5)
ax12 = Axis(f[1, 2], aspect=DataAspect(), title="DC2")
scatter!(smld2_DC.x, smld3_DC.y, markersize = 5)
ax21 = Axis(f[2, 1], aspect=DataAspect(), title="smld3 roi")
scatter!(smld3roi.x, smld3roi.y, markersize = 5)
ax22 = Axis(f[2, 2], aspect=DataAspect(), title="DC3")
scatter!(smld3_DC.x, smld3_DC.y, markersize = 5)
linkxaxes!(ax11, ax12)
linkxaxes!(ax11, ax21)
linkxaxes!(ax11, ax22)
linkyaxes!(ax11, ax12)
linkyaxes!(ax11, ax21)
linkyaxes!(ax11, ax22)
display(f)
