

function NND(smld::SMLMData.SMLD2D,dm::DriftModel)
    d_cutoff=mean(mean(smld.σ_x),mean(smld.σ_y))
    k=10
    data=cat(dims=2,smld.x,smld.y)
    kdtree = KDTree(data; leafsize = 10)
    idxs, dists=knn(kdtree, data, k,true)

    cost=0.0
    for nn=1:length(smld.x)
        id=findall(dist[nn].<d_cutoff)
        cost+=sum(dist[nn][id])
    end
end




# ##
# using NearestNeighbors

# k=2
# data = rand(k, 10^4)
# points=rand(k,10)
# kdtree = KDTree(data; leafsize = 10)

# idxs, dists=knn(kdtree, data, 10,true)


