# Test helper (defines `points_smld`, runs no tests): hand-placed localizations as an SMLD.
using SMLMSim

"""
SMLD of the given localizations, `pts_by_ds[d]` being the (x, y) or (x, y, z) points of dataset `d`,
each with isotropic precision `σ` (μm; the default is the σ the SMLMSim-built fixtures had, which
the P2 fixture needs: at 0.005 its entropy refine moves 5.5 nm), frame 1 and 300 frames.
"""
function points_smld(pts_by_ds; σ = 0.014)
    n_dims = length(first(first(pts_by_ds)))
    em = [n_dims == 2 ?
              Emitter2DFit{Float64}(p[1], p[2], 1000.0, 10.0, σ, σ, 0.0, 0.0;
                                    σ_xy = 0.0, frame = 1, dataset = d) :
              Emitter3DFit{Float64}(p[1], p[2], p[3], 1000.0, 10.0, σ, σ, σ, 0.0, 0.0;
                                    frame = 1, dataset = d)
          for (d, pts) in enumerate(pts_by_ds) for p in pts]
    return BasicSMLD(em, IdealCamera(1:64, 1:64, 0.1), 300, length(pts_by_ds), Dict{String,Any}())
end
