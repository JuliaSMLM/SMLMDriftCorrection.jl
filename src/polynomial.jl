# Define polynomial methods

mutable struct Polynomial1D <: AbstractIntraDrift1D
    degree::Int
    coefficients::Vector{Real}
end
function Polynomial1D(degree::Int)
    return Polynomial1D(degree,zeros(degree))
end

mutable struct IntraPolynomial <: AbstractIntraDrift
    ndims::Int
    dm::Vector{Polynomial1D}
end
function IntraPolynomial(ndims::Int; degree::Int=2)
    dm=Vector{Polynomial1D}(undef,ndims)
    for ii=1:ndims
        dm[ii]=Polynomial1D(degree)
    end
    return IntraPolynomial(ndims,dm)
end

mutable struct Polynomial <: AbstractIntraInter
    ndatasets::Int
    intra::Vector{IntraPolynomial}
    inter::Vector{InterShift}
end
function Polynomial(ndims::Int,ndatasets::Int, nframes::Int; degree=2, initialize::String="zeros")
    intra=Vector{IntraPolynomial}(undef,ndatasets)
    inter=Vector{InterShift}(undef,ndatasets)

    for ii=1:ndatasets
        intra[ii]=IntraPolynomial(ndims; degree=degree)
        inter[ii]=InterShift(ndims)
    end

    if initialize=="random"
        rscale=0.1
        for ii=1:ndatasets, jj=1:ndims
            inter[ii].dm[jj]=rscale*randn()
            intra[ii].dm[jj].coefficients=rscale*randn()./(nframes.^(1:degree))
        end
    end
    return Polynomial(ndatasets,intra,inter)
end

function Polynomial(smld::SMLMData.SMLD2D; degree::Int=2, initialize::String="zeros")
    return Polynomial(2,smld.ndatasets,smld.nframes;degree=degree, initialize=initialize)
end

function applydrift(x::AbstractFloat,framenum::Int,p::Polynomial1D)
    for nn=1:p.degree
        x+=p.coefficients[nn]*framenum^nn
    end
    return x
end

function correctdrift(x::AbstractFloat,framenum::Int,p::Polynomial1D)
    for nn=1:p.degree
        x-=p.coefficients[nn]*framenum^nn
    end
    return x
end



function model2theta(p::Polynomial)
    # assume all polynomials are same
    nc=p.model[1].degree+1
    θ=Array{Float64}(undef,p.ndatasets*p.ndims*nc)

    for ii=1:p.ndatasets,jj=1:p.ndims
        idx=(ii-1)*p.ndims*nc+(jj-1)*nc+1        
        θ[idx:idx+nc-1]=p.model[ii,jj].coefficients
    end
    # remove first dataset offset
    deleteat!(θ,(0:p.ndims-1).*(p.model[1].degree+1).+1)

    return θ
end

function theta2model(θ, pref::Polynomial)
    p=deepcopy(pref)
    nc=p.model[1].degree+1
    
    for ii=1:p.ndims
        insert!(θ,(ii-1)*(nc)+1,0.0)
    end

    for ii=1:p.ndatasets,jj=1:p.ndims
        idx=(ii-1)*p.ndims*nc+(jj-1)*nc+1        
        p.model[ii,jj].coefficients=θ[idx:idx+nc-1]
    end
    
    return p
end

function correctdriftPoly(θ,coords_uncorrected,smld,degree)
    
    coords=deepcopy(coords_uncorrected)
    ndims=size(coords_uncorrected,2)
    nc=degree+1
    θ_full=deepcopy(θ)
    for ii=1:ndims
        insert!(θ_full,(ii-1)*(nc)+1,0.0)
    end

    for ii=1:size(coords,1),jj=1:ndims
        f=smld.framenum[ii]
        d=smld.datasetnum[ii]    
        idx=(d-1)*ndims*nc+(jj-1)*nc+1        
        coefficients=θ_full[idx:idx+nc-1]
        # println(size(coefficients))
        for nn=0:size(coefficients,1)-1
            # println(coords[ii,jj])
            # println(-coefficients[nn+1]*f^nn)
            coords[ii,jj]=coords[ii,jj]-coefficients[nn+1]*f^nn
        end
    end
    return coords
end


function costfun(θ,coords_uncorrected,smld::SMLMData.SMLD2D,p::Polynomial)
    #  println("max: $(maximum(θ)) min: $(minimum(θ))")
        
    coords=correctdriftPoly(θ,coords_uncorrected,smld,p.model[1].degree)
    d_cutoff=mean([mean(smld.σ_x),mean(smld.σ_y)])

    return NND(coords,d_cutoff)+sum(θ)
end

function costfun(θ,coords_uncorrected,kdtree::KDTree,smld::SMLMData.SMLD2D,p::Polynomial)
    #  println("max: $(maximum(θ)) min: $(minimum(θ))")
        
    coords=correctdriftPoly(θ,coords_uncorrected,smld,p.model[1].degree)
    d_cutoff=mean([mean(smld.σ_x),mean(smld.σ_y)])
    
    return NND(coords,kdtree,d_cutoff)+sum(θ)
end

function correctdriftdataset(coords)
 

end


function optimizeintra!(θ,smld::SMLMData.SMLD2D,p::Polynomial,dataset::Int)
    d_cutoff=mean([mean(smld.σ_x),mean(smld.σ_y)])
    dm=theta2model(θ,p)    
    idx=smld.datasetnum.==dataset
    coords=cat(dims=2,smld.x[idx],smld.y[idx])


end