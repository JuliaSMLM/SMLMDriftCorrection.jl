"""
    SMLMDriftCorrection

Fiducial-free drift correction for single-molecule localization microscopy data, in 2D and 3D.
Intra-dataset drift is fit with Legendre polynomials by entropy minimization; datasets are
aligned to each other by cross-correlation refined with entropy. The main entry point is
[`driftcorrect`](@ref); [`align_smld`](@ref) aligns independent SMLDs.
"""
module SMLMDriftCorrection

using FourierTools: FourierTools, fft, fft2d, fftshift, fftshift2d, ifft2d
using LinearAlgebra: LinearAlgebra, norm
using NearestNeighbors: NearestNeighbors, KDTree, knn, nn
using Optim: Optim, BFGS, optimize
using SMLMData: SMLMData, AbstractSMLMConfig, AbstractSMLMInfo, BasicSMLD, Emitter2D,
    Emitter2DFit, Emitter3D, Emitter3DFit, IdealCamera, SMLD
using Statistics: Statistics, cor, mean, median, std
using StatsFuns: StatsFuns

include("typedefs.jl")
include("affine.jl")
include("cost_entropy.jl")
include("costfuns.jl")
include("crosscorr.jl")
include("legendre.jl")
include("intrainter.jl")
include("utilities.jl")
include("roi_selection.jl")
include("occupancy.jl")
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
