# Define polynomial methods

mutable struct Polynomial1D <: AbstractIntraDrift1D
    degree::Int
    coefficients::Vector{<:Real}
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
function Polynomial(ndims::Int,ndatasets::Int, nframes::Int; 
            degree=2, initialize::String="zeros", rscale=0.1)
    intra=Vector{IntraPolynomial}(undef,ndatasets)
    inter=Vector{InterShift}(undef,ndatasets)

    for ii=1:ndatasets
        intra[ii]=IntraPolynomial(ndims; degree=degree)
        inter[ii]=InterShift(ndims)
    end

    if initialize=="random"
        for ii=1:ndatasets, jj=1:ndims
            inter[ii].dm[jj]=rscale*randn()
            intra[ii].dm[jj].coefficients=rscale*randn()./(nframes.^(1:degree))
        end
    end

    if initialize=="continous"
        for ii=1:ndatasets, jj=1:ndims
            if ii==1
                inter[ii].dm[jj]=rscale*randn()
            else
                inter[ii].dm[jj]=inter[ii-1].dm[jj]+applydrift(0.0,nframes, intra[ii-1].dm[jj])     
            end
            intra[ii].dm[jj].coefficients=rscale*randn()./(nframes.^(1:degree))
        end
    end

    return Polynomial(ndatasets,intra,inter)
end

function Polynomial(smld::SMLMData.SMLD2D; degree::Int=2, initialize::String="zeros",rscale=0.1)
    return Polynomial(2,smld.ndatasets,smld.nframes;degree=degree, initialize=initialize,rscale=rscale)
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


function intra2theta(p::IntraPolynomial)
    degree=p.dm[1].degree
    l=p.ndims*degree
    θ=zeros(Real,l)
    for ii=1:p.ndims, jj=1:degree
        θ[jj+(ii-1)*degree]=p.dm[ii].coefficients[jj]
    end
    return θ
end

function theta2intra!(p::IntraPolynomial,θ::Vector{<:Real})
    degree=p.dm[1].degree
    l=p.ndims*degree
    for ii=1:p.ndims, jj=1:degree
        p.dm[ii].coefficients[jj]=θ[jj+(ii-1)*degree]
    end
end

