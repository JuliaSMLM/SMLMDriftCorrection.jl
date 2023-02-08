using Revise
using SMLMDriftCorrection
using SMLMData
using NearestNeighbors
using Zygote
using Optim
using BenchmarkTools

# p=DC.Polynomial(smd_drift;initialize="random")

##


θ=DC.model2theta(driftmodel)
θ0=DC.model2theta(DC.Polynomial(smd_drift; degree=2, initialize="zeros"))
θrand=DC.model2theta(DC.Polynomial(smd_drift; degree=2, initialize="random"))

xy=cat(dims=2,smd_drift.x,smd_drift.y)
coords=Matrix{Real}(undef,size(xy))
coords.=xy
kdtree=KDTree(xy')

plt=PlotlyJS.plot(scattergl(x=smd_drift.x, y=smd_drift.y, mode="markers"))
display(plt)

DC.costfun(θ,xy,smd_drift,driftmodel)
DC.costfun(θ0,xy,smd_drift,driftmodel)
DC.costfun(θrand,xy,smd_drift,driftmodel)


myfun=θ->DC.costfun(θ,xy,smd_drift,driftmodel)
opt=Optim.Options(iterations = 10000,show_trace=true,show_every=10)
res=optimize(myfun, θ0, opt)
θ_found=res.minimizer

# DC.costfun(θ,xy,kdtree,smd_drift,driftmodel)
# kmyfun=θ->DC.costfun(θ,coords,kdtree,smd_drift,driftmodel)
# kmyfun(θ)

# fp = θ-> Zygote.forward_jacobian(kmyfun, θ)
# fp(θ)



dm_found=DC.theta2model(θ_found,driftmodel)
smld_fit=DC.correctdrift(smd_drift,dm_found)

plt=PlotlyJS.plot(scattergl(x=smd_drift.x, y=smd_drift.y, mode="markers"))
display(plt)

plt=PlotlyJS.plot(scattergl(x=smld_fit.x, y=smld_fit.y, mode="markers"))
display(plt)




# g= Zygote.forward_jacobian(myfun,θ)

# fp = (θ,smd_drift,p) -> Zygote.forward_jacobian(θ -> DC.costfun(θ,smd_drift,p), θ)
# fp(θ)


# res=optimize(f, θ, LBFGS(); autodiff = :forward)
# println(res)
# return theta2model(minimizer(res),p)



# # # can we AD through knn?

# k=2
# data = rand(k, 10^4)
# kdtree = KDTree(data; leafsize = 10)
    

# function cost(θ,data,kdtree)
#     points=data.+θ
#     k=1
#     idxs, dists=knn(kdtree, points, k,false)
    
#     c=0.0
#     for ii=1:length(dists)
#         c+=dists[ii][1]
#     end
#     return c
# end


# θ=[1]
# g= ForwardDiff.gradient(θ->cost(θ ,data,kdtree),θ)



