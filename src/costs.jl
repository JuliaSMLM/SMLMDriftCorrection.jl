

function NND(coords,d_cutoff::AbstractFloat)
    data=transpose(coords)
    kdtree = KDTree(data; leafsize = 10)
    return NND(coords,kdtree::KDTree,d_cutoff::AbstractFloat)
end

function NND(coords,kdtree::KDTree,d_cutoff::AbstractFloat)
    k=10
    data=transpose(coords)
    idxs, dists=knn(kdtree, data, k,true)

    cost=0.0
    for nn=1:size(coords,1)
        cost+=sum(min.(dists[nn],d_cutoff))
    end
    
    return cost
end


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


