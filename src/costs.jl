

function NND(smld::SMLMData.SMLD2D)
    d_cutoff=mean([mean(smld.σ_x),mean(smld.σ_y)])
    k=10
    data=cat(dims=2,smld.x,smld.y)'
    kdtree = KDTree(data; leafsize = 10)
    idxs, dists=knn(kdtree, data, k,true)

    cost=0.0
    for nn=1:length(smld.x)
        cost+=sum(max.(dists[nn],d_cutoff))
    end
    return cost
end

function costfun(θ,smld::SMLMData.SMLD2D,p::Polynomial)
    pnew=theta2model(θ, p::Polynomial)   
    smld_dc=correctdrift(smld,pnew)
    return NND(smld_dc)
end


