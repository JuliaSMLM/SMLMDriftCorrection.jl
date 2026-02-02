module SMLMDriftCorrection

using FourierTools
using LinearAlgebra
using NearestNeighbors
using Optim
using SMLMData
using Statistics
using StatsFuns

include("typedefs.jl")
include("cost_entropy.jl")
include("costfuns.jl")
include("crosscorr.jl")
include("legendre.jl")
include("intrainter.jl")
include("utilities.jl")
include("interface.jl")

export driftcorrect
export DriftInfo
export filter_emitters
export drift_trajectory

end
