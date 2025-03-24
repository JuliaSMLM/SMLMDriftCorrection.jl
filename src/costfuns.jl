"""
INTRA-KNN
Cost function computes the cost of an intra-dataset drift correction proposal.
    Here, the cost is the sum of the scaled negative exponentials of the
    nearest neighbor distances computed frame-by-frame within a dataset.

# Fields:
- θ:                parameters for a drift correction proposal
- data_uncorrected: uncorrected coordinate data for each localization
- framenum:         frame number for each localization
- d_cutoff:         cutoff distance
- intra:            intra-dataset data structure
"""
function costfun(θ, data_uncorrected, framenum::Vector{Int}, d_cutoff::AbstractFloat, intra::AbstractIntraDrift)
    theta2intra!(intra, θ)
    data = deepcopy(data_uncorrected)

    # Apply the current model to the uncorrected data.
    ndims = size(data, 1)
    nloc = size(data, 2)
    for nn = 1:ndims, ii = 1:nloc
        data[nn, ii] = correctdrift(data_uncorrected[nn, ii], framenum[ii], intra.dm[nn])
    end

    k = 4   # number of nearest neighbors
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

"""
INTER-KNN ref_data
Cost function computes the cost of an inter-dataset drift correction proposal.
    Here, the cost is the sum of the scaled negative exponentials of the
    k nearest neighbor distances computed dataset-by-dataset.
# Fields:
- θ:                parameters for a drift correction proposal
- data_uncorrected: uncorrected coordinate data for each localization
- ref_data:         reference data for producing the KDTree
- d_cutoff:         cutoff distance
- inter:            inter-dataset data structure
"""
function costfun(θ, data_uncorrected, ref_data, d_cutoff::AbstractFloat, inter::InterShift)
    
    theta2inter!(inter, θ)
    data = deepcopy(data_uncorrected)

    # Apply the current model to the uncorrected data.
    ndims = size(data, 1)
    nloc = size(data, 2)
    for nn = 1:ndims, ii = 1:nloc
        data[nn, ii] = correctdrift(data_uncorrected[nn, ii], inter, nn)
    end

    k = 4   # number of nearest neighbors.
    kdtree = KDTree(ref_data; leafsize = 10)
    idxs, dists = knn(kdtree, data, k, true)
    
    cost = 0.0
    for nn = 1:size(data, 2)    
        # cost += sum(min.(dists[nn], d_cutoff))
        cost -= sum(exp.(-dists[nn]./d_cutoff))
    end
    return cost
end

"""
INTER-KNN kdtree
Cost function computes the cost of an inter-dataset drift correction proposal.
    Here, the cost is the sum of the scaled negative exponentials of the
    k nearest neighbor distances computed dataset-by-dataset against a static
    reference k-D tree.
# Fields:
- θ:                parameters for a drift correction proposal
- data_uncorrected: uncorrected coordinate data for each localization
- kdtree:           kdtree constructed from a reference dataset (see routine above)
- d_cutoff:         cutoff distance
- inter:            inter-dataset data structure
"""
function costfun(θ, data_uncorrected, kdtree::KDTree, d_cutoff::AbstractFloat, inter::InterShift)
    
    theta2inter!(inter, θ)
    data = deepcopy(data_uncorrected)

    # Apply the current model to the uncorrected data.
    ndims = size(data, 1)
    nloc = size(data, 2)
    for nn = 1:ndims, ii = 1:nloc
        data[nn, ii] = correctdrift(data_uncorrected[nn, ii], inter, nn)
    end

    k = 4   # number of nearest neighbors

    idxs, dists = knn(kdtree, data, k, true)
    
    cost = 0.0
    for nn = 1:size(data, 2)    
        # cost += sum(min.(dists[nn], d_cutoff))
        cost -= sum(exp.(-dists[nn]./d_cutoff))
    end
    return cost
end

"""
INTRA-ENTROPY
Cost function computes the cost of an intra-dataset drift correction proposal.
    Here, the cost is the entropy upper bound computed frame-by-frame within a dataset.

# Fields:
- θ:                parameters for a drift correction proposal
- data_uncorrected: uncorrected coordinate data for each localization
- se:               standard errors for each localization
- framenum:         frame number for each localization
- maxn:             maxinum number of neighbors considered
- intra:            intra-dataset data structure
"""
function costfun(θ, data_uncorrected, se, framenum::Vector{Int}, maxn::Int, intra::AbstractIntraDrift)
    theta2intra!(intra, θ)
    data = deepcopy(data_uncorrected)

    # Apply the current model to the uncorrected data.
    ndims = size(data, 1)
    nloc = size(data, 2)
    for nn = 1:ndims, ii = 1:nloc
        data[nn, ii] = correctdrift(data_uncorrected[nn, ii], framenum[ii], intra.dm[nn])
    end

    x = data[1, :]
    y = data[2, :]
    sx = se[1, :]
    sy = se[2, :]
    cost = ub_entropy(x, y, sx, sy; maxn = maxn)
    
    return cost
end

"""
INTER-ENTROPY
Cost function computes the cost of an inter-dataset drift correction proposal.
    Here, the cost is the entropy upper bound computed dataset-by-dataset.

# Fields:
- θ:                parameters for a drift correction proposal
- data_uncorrected: uncorrected coordinate data for each localization
- se:               standard errors for each localization
- maxn:             maximum number of neighbors considered
- inter:            inter-dataset data structure
"""
function costfun(θ, data_uncorrected, se, maxn::Int, inter::InterShift)
    
    theta2inter!(inter, θ)
    data = deepcopy(data_uncorrected)

    # Apply the current model to the uncorrected data.
    ndims = size(data, 1)
    nloc = size(data, 2)
    for nn = 1:ndims, ii = 1:nloc
        data[nn, ii] = correctdrift(data_uncorrected[nn, ii], inter, nn)
    end

    x = data[1, :]
    y = data[2, :]
    sx = se[1, :]
    sy = se[2, :]
    cost = ub_entropy(x, y, sx, sy; maxn = maxn)

    return cost
end
