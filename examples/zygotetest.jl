using Revise
using SMLMDriftCorrection
using SMLMData
using NearestNeighbors
using Zygote



# can we AD through knn?

k=2
data = rand(k, 10^4)
kdtree = KDTree(data; leafsize = 10)
    

function cost(θ,data,kdtree)
    points=data.+θ
    k=1
    idxs, dists=knn(kdtree, points, k,true)
    
    c=0.0
    for ii=1:length(dists)
        c+=dists[ii][1]
    end
    return c
    end


g= Zygote.forwarddiff(θ->cost(θ ,data,kdtree),θ)

