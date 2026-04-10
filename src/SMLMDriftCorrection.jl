module SMLMDriftCorrection

using FourierTools
using LinearAlgebra
using NearestNeighbors
using Optim
using SMLMData
using Statistics
using StatsFuns

include("typedefs.jl")
include("affine.jl")
include("cost_entropy.jl")
include("costfuns.jl")
include("crosscorr.jl")
include("legendre.jl")
include("intrainter.jl")
include("utilities.jl")
include("roi_selection.jl")
include("interface.jl")
include("align.jl")
include("diagnostics.jl")

export driftcorrect
export DriftConfig
export DriftInfo
export filter_emitters
export drift_trajectory
export align_smld
export AlignConfig
export AlignInfo
export AbstractAlignTransform
export ShiftTransform
export AffineTransform2D
export AffineTransform3D
export position_frame_correlation

end
