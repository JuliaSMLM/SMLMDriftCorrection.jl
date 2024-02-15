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
function applydrift!(smld::SMLMData.SMLD2D, dm::AbstractIntraInter)
    for nn = 1:length(smld.x)
        smld.x[nn] = applydrift(smld.x[nn], smld.framenum[nn], dm.intra[smld.datasetnum[nn]].dm[1])
        smld.x[nn] = applydrift(smld.x[nn], dm.inter[smld.datasetnum[nn]], 1)

        smld.y[nn] = applydrift(smld.y[nn], smld.framenum[nn], dm.intra[smld.datasetnum[nn]].dm[2])
        smld.y[nn] = applydrift(smld.y[nn], dm.inter[smld.datasetnum[nn]], 2)
    end
end

function applydrift(smld::SMLMData.SMLD2D, driftmodel::AbstractIntraInter)
    smld_shifted = deepcopy(smld)
    applydrift!(smld_shifted::SMLMData.SMLD2D, driftmodel::AbstractIntraInter)
    return smld_shifted
end

function correctdrift!(smld::SMLMData.SMLD2D, dm::AbstractIntraInter)
    for nn = 1:length(smld.x)
        smld.x[nn] = correctdrift(smld.x[nn], smld.framenum[nn], dm.intra[smld.datasetnum[nn]].dm[1])
        smld.x[nn] = correctdrift(smld.x[nn], dm.inter[smld.datasetnum[nn]], 1)

        smld.y[nn] = correctdrift(smld.y[nn], smld.framenum[nn], dm.intra[smld.datasetnum[nn]].dm[2])
        smld.y[nn] = correctdrift(smld.y[nn], dm.inter[smld.datasetnum[nn]], 2)
    end
end

function correctdrift(smld::SMLMData.SMLD2D, driftmodel::AbstractIntraInter)
    smld_shifted = deepcopy(smld)
    correctdrift!(smld_shifted, driftmodel)
    return smld_shifted
end

#function correctdrift!(smld::SMLMData.SMLD2D, shift::Vector{AbstractFloat})
function correctdrift!(smld::SMLMData.SMLD2D, shift::Vector{Float64})
    #smld_shifted = deepcopy(smld)
    println("correctdrift!: shift = $shift")
    smld.x .-= shift[1]
    smld.y .-= shift[2]
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
    smld::SMLMData.SMLD2D,
    dataset::Int,
    d_cutoff::AbstractFloat,
    maxn::Int)

    idx = smld.datasetnum .== dataset
    coords = cat(dims = 2, smld.x[idx], smld.y[idx])
    stderr = cat(dims = 2, smld.σ_x[idx], smld.σ_y[idx])
    framenum = smld.framenum[idx]
    data = transpose(coords)
    se = transpose(stderr)

    rscale = 0.01
    nframes = smld.nframes
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
    smld_uncorrected::SMLMData.SMLD2D,
    dataset1::Int,
    dataset2::Vector{Int},
    d_cutoff::AbstractFloat,
    maxn::Int,
    histbinsize::AbstractFloat
    )

    if histbinsize > 0.0
        # Apply an optional cross-correlation correction before applying the
        # kD-tree correction.
        subind1 = smld_uncorrected.datasetnum .== dataset1
        smld1 = SMLMData.isolatesmld(smld_uncorrected, subind1)
        println("=== dataset1 = $dataset1, dataset2 = $dataset2")
        for nn = 1:length(dataset2)
            subind2 = smld_uncorrected.datasetnum .== dataset2[nn]
            smld2 = SMLMData.isolatesmld(smld_uncorrected, subind2)
            shift = findshift2D(smld1, smld2; histbinsize=histbinsize)
            shift = .-shift # correct sign of shift
            println()
            println("nn = $nn, shift = $shift")
            #println("--- A smld2.x[1:3] = $(smld2.x[1:3])")
            correctdrift!(smld2, shift)
            #println("--- B smld2.x[1:3] = $(smld2.x[1:3])")
            smld_uncorrected.x[subind2] = smld2.x
            smld_uncorrected.y[subind2] = smld2.y
        end
    end
    
    # get uncorrected coords for dataset 1 
    idx1 = smld_uncorrected.datasetnum .== dataset1
    coords1 = cat(dims = 2, smld_uncorrected.x[idx1], smld_uncorrected.y[idx1])
    data = transpose(coords1)
    stderr1 = cat(dims = 2, smld_uncorrected.σ_x[idx1], smld_uncorrected.σ_y[idx1])
    se = transpose(stderr1)

    # correct everything
    smld = correctdrift(smld_uncorrected,dm)

    # get corrected coords for reference datasets 
    idx2=zeros(Bool,length(smld.datasetnum))
    for nn=1:length(dataset2)
        idx2 = idx2.|(smld.datasetnum .== dataset2[nn])
    end   
    coords2 = cat(dims = 2, smld.x[idx2], smld.y[idx2])
    data_ref = transpose(coords2)

    # build static kdtree from ref data
    kdtree = KDTree(data_ref; leafsize = 10)

    # use current model as starting point 
    inter=dm.inter[dataset1]
    θ0 = Float64.(inter2theta(inter))
    
    if cost_fun == "Kdtree"
        myfun = θ -> costfun(θ, data, kdtree, d_cutoff, inter)
    elseif cost_fun == "Entropy"
        myfun = θ -> costfun(θ, data, se, maxn, inter)
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
    smld_uncorrected::SMLMData.SMLD2D,
    dataset1::Int,
    d_cutoff::AbstractFloat,
    maxn::Int,
    histbinsize::AbstractFloat)
    refdatasets=Int.(1:smld_uncorrected.ndatasets)
    deleteat!(refdatasets,dataset1)
    return findinter!(dm, cost_fun,  smld_uncorrected, dataset1, refdatasets,
                      d_cutoff, maxn, histbinsize)   
end

"""
Experimental.
"""
function globalcost(smld::SMLMData.SMLD2D; k::Int=4, d_cutoff=1.0)
    
    coords1 = cat(dims = 2, smld.x, smld.y)
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