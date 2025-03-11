# Intra+Inter 

abstract type AbstractIntraDrift1D end

abstract type AbstractIntraDrift end

abstract type AbstractIntraInter <: AbstractDriftModel end

mutable struct InterShift
    ndims::Int
    dm::Vector{<:Real}
end

function InterShift(ndims::Int)
    return InterShift(ndims, zeros(ndims))
end

function inter2theta(s::InterShift)
    θ=s.dm
end

function theta2inter!(s::InterShift,θ)
    s.dm.=θ
end

"""
Apply drift to simulated data.
"""
function applydrift(x::AbstractFloat, s::InterShift, dim::Int)
    return x + s.dm[dim]
end

"""
Apply drift correction to drifted data.
"""
function correctdrift(x::AbstractFloat, s::InterShift, dim::Int)
    return x - s.dm[dim]
end

"""
Apply x- and y-drift to the data in the smld structure.
"""
function applydrift!(smld::BasicSMLD{Float64, Emitter2DFit{Float64}}, dm::AbstractIntraInter)
    for nn = 1:length(smld.emitters)
        smld.emitters[nn].x = applydrift(smld.emitters[nn].x, smld.emitters[nn].frame, dm.intra[smld.emitters[nn].dataset].dm[1])
        smld.emitters[nn].x = applydrift(smld.emitters[nn].x, dm.inter[smld.emitters[nn].dataset], 1)

        smld.emitters[nn].y = applydrift(smld.emitters[nn].y, smld.emitters[nn].frame, dm.intra[smld.emitters[nn].dataset].dm[2])
        smld.emitters[nn].y = applydrift(smld.emitters[nn].y, dm.inter[smld.emitters[nn].dataset], 2)
    end
end


function applydrift!(smld::BasicSMLD{Float64, Emitter3DFit{Float64}}, dm::AbstractIntraInter)
    for nn = 1:length(smld.emitters)
        smld.emitters[nn].x = applydrift(smld.emitters[nn].x, smld.emitters[nn].frame, dm.intra[smld.emitters[nn].dataset].dm[1])
        smld.emitters[nn].x = applydrift(smld.emitters[nn].x, dm.inter[smld.emitters[nn].dataset], 1)

        smld.emitters[nn].y = applydrift(smld.emitters[nn].y, smld.emitters[nn].frame, dm.intra[smld.emitters[nn].dataset].dm[2])
        smld.emitters[nn].y = applydrift(smld.emitters[nn].y, dm.inter[smld.emitters[nn].dataset], 2)

        smld.emitters[nn].z = applydrift(smld.emitters[nn].z, smld.emitters[nn].frame, dm.intra[smld.emitters[nn].dataset].dm[3])
        smld.emitters[nn].z = applydrift(smld.emitters[nn].z, dm.inter[smld.emitters[nn].dataset], 3)
    end
end


""" 
  applydrift(smld::SMLMData.Emitter2DFit, driftmodel::AbstractIntraInter) -> SMLMData.Emitter2DFit

Applies a drift model to the Single-Molecule Localization Microscopy (SMLM) data and returns the drift-corrected data.

# Arguments
- `smld::SMLMData.Emitter2DFit`: The SMLM data structure containing the original localization data.
- `driftmodel::AbstractIntraInter`: The drift model to be applied to the SMLM data. This model should account for both intra- and inter-frame drift corrections.

# Returns
- `SMLMData.Emitter2DFit`: A new SMLM data structure with the drift corrections applied.

"""
function applydrift(smld::BasicSMLD{Float64, Emitter2DFit{Float64}}, driftmodel::AbstractIntraInter)
    smld_shifted = deepcopy(smld)
    applydrift!(smld_shifted::BasicSMLD{Float64, Emitter2DFit{Float64}}, driftmodel::AbstractIntraInter)
    return smld_shifted
end

function applydrift(smld::BasicSMLD{Float64, Emitter3DFit{Float64}}, driftmodel::AbstractIntraInter)
    smld_shifted = deepcopy(smld)
    applydrift!(smld_shifted::BasicSMLD{Float64, Emitter3DFit{Float64}}, driftmodel::AbstractIntraInter)
    return smld_shifted
end

function correctdrift!(smld::BasicSMLD{Float64, Emitter2DFit{Float64}}, dm::AbstractIntraInter)
    for nn = 1:length(smld.emitters)
        smld.emitters[nn].x = correctdrift(smld.emitters[nn].x, smld.emitters[nn].frame, dm.intra[smld.emitters[nn].dataset].dm[1])
        smld.emitters[nn].x = correctdrift(smld.emitters[nn].x, dm.inter[smld.emitters[nn].dataset], 1)

        smld.emitters[nn].y = correctdrift(smld.emitters[nn].y, smld.emitters[nn].frame, dm.intra[smld.emitters[nn].dataset].dm[2])
        smld.emitters[nn].y = correctdrift(smld.emitters[nn].y, dm.inter[smld.emitters[nn].dataset], 2)
    end
end

function correctdrift!(smld::BasicSMLD{Float64, Emitter3DFit{Float64}}, dm::AbstractIntraInter)
    for nn = 1:length(smld.emitters)
        smld.emitters[nn].x = correctdrift(smld.emitters[nn].x, smld.emitters[nn].frame, dm.intra[smld.emitters[nn].dataset].dm[1])
        smld.emitters[nn].x = correctdrift(smld.emitters[nn].x, dm.inter[smld.emitters[nn].dataset], 1)

        smld.emitters[nn].y = correctdrift(smld.emitters[nn].y, smld.emitters[nn].frame, dm.intra[smld.emitters[nn].dataset].dm[2])
        smld.emitters[nn].y = correctdrift(smld.emitters[nn].y, dm.inter[smld.emitters[nn].dataset], 2)

        smld.emitters[nn].z = correctdrift(smld.emitters[nn].z, smld.emitters[nn].frame, dm.intra[smld.emitters[nn].dataset].dm[3])
        smld.emitters[nn].z = correctdrift(smld.emitters[nn].z, dm.inter[smld.emitters[nn].dataset], 3)
    end
end

function correctdrift(smld::BasicSMLD{Float64, Emitter2DFit{Float64}}, driftmodel::AbstractIntraInter)
    smld_shifted = deepcopy(smld)
    correctdrift!(smld_shifted, driftmodel)
    return smld_shifted
end

function correctdrift(smld::BasicSMLD{Float64, Emitter3DFit{Float64}}, driftmodel::AbstractIntraInter)
    smld_shifted = deepcopy(smld)
    correctdrift!(smld_shifted, driftmodel)
    return smld_shifted
end

#function correctdrift!(smld::SMLMData.Emitter2DFit, shift::Vector{AbstractFloat})
function correctdrift!(smld::BasicSMLD{Float64, Emitter2DFit{Float64}}, shift::Vector{Float64})
    #smld_shifted = deepcopy(smld)
    #println("correctdrift!: shift = $shift")
    #smld.x .-= shift[1]
    #smld.y .-= shift[2]
    for nn = 1 : length(smld.emitters)
        smld.emitters[nn].x -= shift[1]
        smld.emitters[nn].y -= shift[2]
    end
end

#function correctdrift!(smld::SMLMData.Emitter3DFit, shift::Vector{AbstractFloat})
function correctdrift!(smld::BasicSMLD{Float64, Emitter3DFit{Float64}}, shift::Vector{Float64})
    #smld_shifted = deepcopy(smld)
    #println("correctdrift!: shift = $shift")
    #smld.x .-= shift[1]
    #smld.y .-= shift[2]
    #smld.z .-= shift[3]
    for nn = 1 : length(smld.emitters)
        smld.emitters[nn].x -= shift[1]
        smld.emitters[nn].y -= shift[2]
        smld.emitters[nn].z -= shift[3]
    end
end

"""
Find and correct intra-detaset drift.

# Fields:
- intra:            intra-dataset structure
- cost_fun:         cost function: {"Kdtree", "Entropy"}
- smld:             data structure containing coordinate data
- dataset           dataset number to operate on
- d_cutoff:         cutoff distance
- maxn:             maximum number of neighbors considered
"""
function findintra!(intra::AbstractIntraDrift,
    cost_fun::String,
    smld::BasicSMLD,
    dataset::Int,
    d_cutoff::AbstractFloat,
    maxn::Int)

#   idx = smld.datasetnum .== dataset
    idx = [e.dataset for e in smld.emitters] .== dataset
    if intra.ndims == 2
        coords = cat(dims = 2, [e.x for e in smld.emitters[idx]],
                               [e.y for e in smld.emitters[idx]])
        stderr = cat(dims = 2, [e.σ_x for e in smld.emitters[idx]],
                               [e.σ_y for e in smld.emitters[idx]])
    elseif intra.ndims == 3
        coords = cat(dims = 2, [e.x for e in smld.emitters[idx]],
                               [e.y for e in smld.emitters[idx]],
                               [e.z for e in smld.emitters[idx]])
        stderr = cat(dims = 2, [e.σ_x for e in smld.emitters[idx]],
                               [e.σ_y for e in smld.emitters[idx]],
                               [e.σ_z for e in smld.emitters[idx]])
    end
    #coords = cat(dims = 2, smld.x[idx], smld.y[idx])
    #stderr = cat(dims = 2, smld.σ_x[idx], smld.σ_y[idx])
    framenum = [e.frame for e in smld.emitters[idx]]
    data = transpose(coords)
    se = transpose(stderr)

    rscale = 0.01
    if intra.ndims == 2
        coords = cat(dims = 2, [e.x for e in smld.emitters[idx]],
                               [e.y for e in smld.emitters[idx]])
        stderr = cat(dims = 2, [e.σ_x for e in smld.emitters[idx]],
                               [e.σ_y for e in smld.emitters[idx]])
    elseif intra.ndims == 3
        coords = cat(dims = 2, [e.x for e in smld.emitters[idx]],
                               [e.y for e in smld.emitters[idx]],
                               [e.z for e in smld.emitters[idx]])
        stderr = cat(dims = 2, [e.σ_x for e in smld.emitters[idx]],
                               [e.σ_y for e in smld.emitters[idx]],
                               [e.σ_z for e in smld.emitters[idx]])
    end
    #coords = cat(dims = 2, smld.x[idx], smld.y[idx])
    #stderr = cat(dims = 2, smld.σ_x[idx], smld.σ_y[idx])
#   framenum = smld.framenum[idx]
    framenum = [e.frame for e in smld.emitters[idx]]
    data = transpose(coords)
    se = transpose(stderr)

    rscale = 0.01
    nframes = smld.n_frames
    for jj = 1:intra.ndims
        degree = intra.dm[jj].degree
        intra.dm[jj].coefficients = rscale * randn() ./ (nframes .^ (1:degree))
    end

    #convert all intra drift parameters to a single vector for optimization
    θ0 = Float64.(intra2theta(intra))

    if cost_fun == "Kdtree"
        myfun = θ -> costfun(θ, data, framenum, d_cutoff, intra)
    elseif cost_fun == "Entropy"
        myfun = θ -> costfun(θ, data, se, framenum, maxn, intra)
    end
    # println(myfun(θ0))
    opt = Optim.Options(iterations = 10000, show_trace = false)
    res = optimize(myfun, θ0, opt)
    θ_found = res.minimizer

    theta2intra!(intra, θ_found)
end

"""
Find and correct inter-detaset drift.

# Fields:
- dm:               inter-dataset structure
- cost_fun:         cost function: {"Kdtree", "Entropy"}
- smld_uncorrected: data structure containing uncorrected coordinate data
- dataset1:         dataset number for the reference dataset
- dataset2:         dataset numbers to operate on
- d_cutoff:         cutoff distance
- maxn:             maximum number of neighbors considered
- histbinsize:      histogram bin size for optional cross-correlation correction
"""
function findinter!(dm::AbstractIntraInter,
    cost_fun::String,
    smld_uncorrected::BasicSMLD{Float64, Emitter2DFit{Float64}},
    dataset1::Int,
    dataset2::Vector{Int},
    d_cutoff::AbstractFloat,
    maxn::Int,
    histbinsize::AbstractFloat
    )

    # get uncorrected coords for dataset 1 
#   idx1 = smld_uncorrected.datasetnum .== dataset1
    idx1 = [e.dataset for e in smld_uncorrected.emitters] .== dataset1
    coords1 = cat(dims = 2, [e.x for e in smld_uncorrected.emitters[idx1]],
                            [e.y for e in smld_uncorrected.emitters[idx1]])
    data = transpose(coords1)
    stderr1 = cat(dims = 2, [e.σ_x for e in smld_uncorrected.emitters[idx1]],
                            [e.σ_y for e in smld_uncorrected.emitters[idx1]])
    se = transpose(stderr1)

    # correct everything
    smld = correctdrift(smld_uncorrected, dm)

    # get corrected coords for reference datasets
    # (in other words, make a sum image)
    idx2 = zeros(Bool, length(smld.emitters))
    for nn=1:length(dataset2)
#       idx2 = idx2 .| (smld.emitters.dataset .== dataset2[nn])
        idx2 = idx2 .| ([e.dataset for e in smld.emitters] .== dataset2[nn])
    end   
    coords2 = cat(dims = 2, [e.x for e in smld.emitters[idx2]],
                            [e.y for e in smld.emitters[idx2]])
    data_ref = transpose(coords2)

    if histbinsize > 0.0
        # Apply an optional cross-correlation correction.
        #println("=== dataset1 = $dataset1, dataset2 = $dataset2")
#       smld1 = SMLMData.isolatesmld(smld, idx1)
#       smld2 = SMLMData.isolatesmld(smld, idx2)
        smld1 = filter_emitters(smld, idx1)
        smld2 = filter_emitters(smld, idx2)
        shift = findshift2D(smld1, smld2; histbinsize=histbinsize)
        #shift = .-shift # correct sign of shift
        #println("shift = $shift")
        correctdrift!(smld1, shift)
#       smld.x[idx1] = smld1.x
#       smld.y[idx1] = smld1.y
        for nn = 1:length(idx1)
	    smld.emitters[idx1[nn]].x = smld1.emitters[nn].x
	    smld.emitters[idx1[nn]].y = smld1.emitters[nn].y
	end
        if cost_fun == "None"
            theta2inter!(dm.inter[dataset1], shift)
            return 0.0
        end
    end

    # build static kdtree from ref data
    kdtree = KDTree(data_ref; leafsize = 10)

    # use current model as starting point 
    inter=dm.inter[dataset1]
    θ0 = Float64.(inter2theta(inter))
    
    if cost_fun == "Kdtree"
        myfun = θ -> costfun(θ, data, kdtree, d_cutoff, inter)
    elseif cost_fun == "Entropy"
        myfun = θ -> costfun(θ, data, se, maxn, inter)
    else
        error("cost_fun not recognized")
    end
    #println(myfun(θ0))
    opt = Optim.Options(iterations = 10000, show_trace = false)
    res = optimize(myfun, θ0, opt)
    θ_found = res.minimizer
    #println("θ_found = $θ_found")
    
    theta2inter!(inter, θ_found)
    return res.minimum
end

"""
Find and correct inter-detaset drift.

# Fields:
- dm:               inter-dataset structure
- cost_fun:         cost function: {"Kdtree", "Entropy"}
- smld_uncorrected: data structure containing uncorrected coordinate data
- dataset1:         dataset number for the reference dataset
- d_cutoff:         cutoff distance
- maxn:             maximum number of neighbors considered
- histbinsize:      histogram bin size for optional cross-correlation correction
"""
function findinter!(dm::AbstractIntraInter,
    cost_fun::String,
    smld_uncorrected::BasicSMLD,
    dataset1::Int,
    d_cutoff::AbstractFloat,
    maxn::Int,
    histbinsize::AbstractFloat)
    refdatasets = Int.(1:smld_uncorrected.n_datasets)
    deleteat!(refdatasets, dataset1)
    return findinter!(dm, cost_fun,  smld_uncorrected, dataset1, refdatasets,
                      d_cutoff, maxn, histbinsize)   
end

"""
Experimental.
"""
function globalcost(smld::BasicSMLD; k::Int=4, d_cutoff=1.0)
    
    coords1 = cat(dims = 2, [e.x for e in smld.emitters[idx]],
                            [e.y for e in smld.emitters[idx]])
    data = transpose(coords1)
    
    kdtree = KDTree(data; leafsize = 10)
    idxs, dists = knn(kdtree, data, k+1, true)
    
    cost = 0.0
    for nn = 2:size(data, 2)    
        # cost += sum(min.(dists[nn], d_cutoff))
        cost -= sum(exp.(-dists[nn]./d_cutoff))
    end
    return cost
end
