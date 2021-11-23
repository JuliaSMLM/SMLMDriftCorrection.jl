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


function applydrift(x::AbstractFloat, s::InterShift, dim::Int)
    return x + s.dm[dim]
end

function correctdrift(x::AbstractFloat, s::InterShift, dim::Int)
    return x - s.dm[dim]
end

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



function costfun(θ, data_uncorrected, framenum::Vector{Int}, d_cutoff::AbstractFloat, intra::AbstractIntraDrift)
    theta2intra!(intra, θ)
    data = deepcopy(data_uncorrected)

    ndims = size(data, 1)
    nloc = size(data, 2)
    for nn = 1:ndims, ii = 1:nloc
        data[nn, ii] = correctdrift(data_uncorrected[nn, ii], framenum[ii], intra.dm[nn])
    end

    k = 10
    kdtree = KDTree(data; leafsize = 10)
    idxs, dists = knn(kdtree, data, k, true)
    # println(size(dists))
    cost = 0.0
    for nn = 2:size(dists, 1)
        # cost += sum(min.(dists[nn], d_cutoff))
         cost -= sum(exp.(-dists[nn]./d_cutoff))
        
    end
    
    return cost
end

function costfun(θ, data_uncorrected, ref_data, d_cutoff::AbstractFloat, inter::InterShift)
    
    theta2inter!(inter, θ)
    data = deepcopy(data_uncorrected)

    ndims = size(data, 1)
    nloc = size(data, 2)
    for nn = 1:ndims, ii = 1:nloc
        data[nn, ii] = correctdrift(data_uncorrected[nn, ii], inter, nn)
    end

    k = 10
    kdtree = KDTree(ref_data; leafsize = 10)
    idxs, dists = knn(kdtree, data, k, true)
    
    cost = 0.0
    for nn = 1:size(data, 2)    
        # cost += sum(min.(dists[nn], d_cutoff))
        cost -= sum(exp.(-dists[nn]./d_cutoff))
    end
    return cost
end


function findintra!(intra::AbstractIntraDrift, smld::SMLMData.SMLD2D, dataset::Int, d_cutoff::AbstractFloat)
    idx = smld.datasetnum .== dataset
    coords = cat(dims = 2, smld.x[idx], smld.y[idx])
    framenum = smld.framenum[idx]
    data = transpose(coords)

    rscale = 0.01
    nframes = smld.nframes
    for jj = 1:intra.ndims
        degree = intra.dm[jj].degree
        intra.dm[jj].coefficients = rscale * randn() ./ (nframes .^ (1:degree))
    end

    #convert all intra drift parameters to a single vector for optimization
    θ0 = Float64.(intra2theta(intra))

    myfun = θ -> costfun(θ, data, framenum, d_cutoff, intra)
    # println(myfun(θ0))
    opt = Optim.Options(iterations = 10000, show_trace = false)
    res = optimize(myfun, θ0, opt)
    θ_found = res.minimizer

    theta2intra!(intra, θ_found)
end

function findinter!(dm::AbstractIntraInter, smld_uncorrected::SMLMData.SMLD2D, dataset1::Int, dataset2::Int, d_cutoff::AbstractFloat)
    
    # correct everything but inter dataset 1
    
    # set inter to zero for dataset 1
    inter=dm.inter[dataset1]
    for jj = 1:inter.ndims
        inter.dm[jj] = 0.0
    end
    # correct everything else
    smld=correctdrift(smld_uncorrected,dm)
    
    idx1 = smld.datasetnum .== dataset1
    idx2 = smld.datasetnum .== dataset2
    
    coords1 = cat(dims = 2, smld.x[idx1], smld.y[idx1])
    data = transpose(coords1)
    
    coords2 = cat(dims = 2, smld.x[idx2], smld.y[idx2])
    data_ref = transpose(coords2)

    #convert all intra drift parameters to a single vector for optimization
    θ0 = Float64.(inter2theta(inter))
    
    myfun = θ -> costfun(θ, data, data_ref, d_cutoff, inter)
    # println(myfun(θ0))
    opt = Optim.Options(iterations = 10000, show_trace = false)
    res = optimize(myfun, θ0, opt)
    θ_found = res.minimizer

    theta2inter!(inter, θ_found)
end

function findinter!(dm::AbstractIntraInter, smld_uncorrected::SMLMData.SMLD2D, dataset1::Int,  d_cutoff::AbstractFloat)
    
    # set inter to zero for dataset 1
    inter=dm.inter[dataset1]
    for jj = 1:inter.ndims
        inter.dm[jj] = 0.0
    end
    # correct everything else
    smld=correctdrift(smld_uncorrected,dm)
    
    idx1 = smld.datasetnum .== dataset1
    idx2 = smld.datasetnum .!= dataset1
    
    coords1 = cat(dims = 2, smld.x[idx1], smld.y[idx1])
    data = transpose(coords1)
    
    coords2 = cat(dims = 2, smld.x[idx2], smld.y[idx2])
    data_ref = transpose(coords2)

    inter=dm.inter[dataset1]
    
    # convert all intra drift parameters to a single vector for optimization
    θ0 = Float64.(inter2theta(inter))

    myfun = θ -> costfun(θ, data, data_ref, d_cutoff, inter)
    opt = Optim.Options(iterations = 10000, show_trace = false)
    res = optimize(myfun, θ0, opt)
    θ_found = res.minimizer

    theta2inter!(inter, θ_found)    
end



