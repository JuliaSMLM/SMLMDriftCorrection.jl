"""
Compute a mearest neighbor cost function for the drift correction algorithm.
    Currently, the cost function is the sum of the minimum nearest neighbor
    distance for each localization, applying a distance cutoff which replaces
    a distance too far away from its closest neighbor by the cutoff.
"""

"""
NND computes the nearest neighbor distance of the given coordinates using a k-D
    tree.

# Fields
- coords:   (X, Y) coordinates
- d_cutoff: cutoff distance
- kdtree:   KDTree structure
- k:        10

See NearestNeighbors.jl for further info.
"""
function NND(coords,d_cutoff::AbstractFloat)
    data=transpose(coords)
    kdtree = KDTree(data; leafsize = 10)
    return NND(coords,kdtree::KDTree,d_cutoff::AbstractFloat)
end

function NND(coords,kdtree::KDTree,d_cutoff::AbstractFloat; k::Int=10)
    data=transpose(coords)
    idxs, dists=knn(kdtree, data, k,true)

    cost=0.0
    for nn=1:size(coords,1)
        cost+=sum(min.(dists[nn],d_cutoff))
    end
    
    return cost
end

"""
finddrift uses a polynomial drift model of the given degree and the NND cost
    function to predict the drift correction.

#Fields
- smld:   SMLMData.SMLM2D structure containing X and Y coordinates
- degree: polyomial degree = 2
"""
function finddrift(smld::SMLMData.SMLD2D; degree::Int=2)
    p=Polynomial(smld;degree=degree,initialize="zeros")
    θ=model2theta(p)
    coords=cat(dims=2,smld.x,smld.y)


    f=θ->costfun(θ,smld,p)
    f(θ)
    # res=optimize(f, θ)
    # println(res)
    # return theta2model(minimizer(res),p)
end