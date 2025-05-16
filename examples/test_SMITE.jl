using Revise
using GLMakie
using SMLMData
using SMLMDriftCorrection

# path/file local to LidkeLab
path = raw"Y:\Personal Folders\MJW\tmp"
file = raw"Cell_01_Label_01_Data_2023-10-12-12-51-39_Results.mat"
smd = SmiteSMD(path, file)
# Load the file if not done previously
if !isdefined(Main, :smld2)
    println("Loading file: $file")
    @time smld2 = load_smite_2d(smd) #To check keys use, varnames = keys(smld2)
    println("Loaded file: $file")
end

# --- 2D ---

println("N_smld2 = $(length(smld2.emitters))")
smld2_x = [e.x for e in smld2.emitters]
smld2_y = [e.y for e in smld2.emitters]
subind = (smld2_x .> 64.0) .& (smld2_x .< 128.0) .&
         (smld2_y .> 64.0) .& (smld2_y .< 128.0)
smld2roi = DC.filter_emitters(smld2, subind)
println("N_smld2 = $(length(smld2roi.emitters))")

smld2_DC = driftcorrect(smld2roi; verbose = 1, cost_fun = "Kdtree")

# ----------

f = Figure()
smld2roi_x = [e.x for e in smld2roi.emitters]
smld2roi_y = [e.y for e in smld2roi.emitters]
smld2_DC_x = [e.x for e in smld2_DC.emitters]
smld2_DC_y = [e.y for e in smld2_DC.emitters]
ax11 = Axis(f[1, 1], aspect=DataAspect(), title="smld2 roi")
scatter!(smld2roi_x, smld2roi_y, markersize = 5)
ax12 = Axis(f[1, 2], aspect=DataAspect(), title="DC2")
scatter!(smld2_DC_x, smld2_DC_y, markersize = 5)
linkxaxes!(ax11, ax12)
linkyaxes!(ax11, ax12)
display(f)
