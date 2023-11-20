using Revise
using JLD2
using FileIO
using SMLMData

dirname = "Y:\\Projects\\Super Critical Angle Localization Microscopy\\Data\\10-06-2023\\Data2\\old insitu psf and stg pos"
file = "Data2-2023-10-6-17-11-54deepfit1.jld2"
filepath = joinpath(dirname, file)
# Load the file
data = load(filepath) #To check keys use, varnames = keys(data)
# Get smld
smld = data["smld"]

findshift2D(smld, smld)