# Define polynomial methods

"""
Define a univariate polynomial data type.
"""
mutable struct Polynomial1D <: AbstractIntraDrift1D
    degree::Int
    coefficients::Vector{<:Real}
end
function Polynomial1D(degree::Int)
    return Polynomial1D(degree,zeros(degree))
end

"""
Define a data type for intra-dataset drifts, which will be a collection of
univariate polynomials indexed by each coordinate dimension and frame number.
"""
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

"""
Initialize polynomials.

# Fields:
- ndims:      number of dimensions
- ndatasets:  number of datasets
- nframes:    number of frames
- degree:     polymomial degree = 2
- initialize: string indicating possible initializations
              ("zeros" [default], "random", "continuous")
- rscale:     = scale factor for normalized random numbers = 0.01 μm
"""
mutable struct Polynomial <: AbstractIntraInter
    ndatasets::Int
    intra::Vector{IntraPolynomial}
    inter::Vector{InterShift}
end
function Polynomial(ndims::Int, ndatasets::Int, nframes::Int; 
            degree=2, initialize::String="zeros", rscale=0.01)
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

    if initialize=="continuous"
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

#function Polynomial(smld::SMLD{T, E}; degree::Int=2, initialize::String="zeros",rscale=0.1)
#    where {T<:AbstractFloat, E<:Emitter2DFit{T}}
#    return Polynomial(2,smld.n_datasets,smld.n_frames;degree=degree, initialize=initialize,rscale=rscale)
#end

function Polynomial(smld::SMLD; degree::Int=2, initialize::String="zeros",rscale=0.1)
    return Polynomial(2,smld.n_datasets,smld.n_frames;degree=degree, initialize=initialize,rscale=rscale)
end

function Polynomial(smld::BasicSMLD{Float64, Emitter3DFit{Float64}}; degree::Int=2, initialize::String="zeros",rscale=0.1)
    return Polynomial(3,smld.n_datasets,smld.n_frames;degree=degree, initialize=initialize,rscale=rscale)
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

"""
Convert intra-dataset polynomials (p) to coefficients (θ).
"""
function intra2theta(p::IntraPolynomial)
    degree=p.dm[1].degree
    l=p.ndims*degree
    θ=zeros(Real,l)
    for ii=1:p.ndims, jj=1:degree
        θ[jj+(ii-1)*degree]=p.dm[ii].coefficients[jj]
    end
    return θ
end

"""
Convert polynomial coefficients (θ) to intra-dataset polynomials (p).
"""
function theta2intra!(p::IntraPolynomial,θ::Vector{<:Real})
    degree=p.dm[1].degree
    l=p.ndims*degree
    for ii=1:p.ndims, jj=1:degree
        p.dm[ii].coefficients[jj]=θ[jj+(ii-1)*degree]
    end
end

"""
Initialize polynomial coefficients with random values.
Called by findintra! before optimization.
"""
function initialize_random!(p::IntraPolynomial, rscale::Real, nframes::Int)
    for jj = 1:p.ndims
        degree = p.dm[jj].degree
        p.dm[jj].coefficients = rscale * randn() ./ (nframes .^ (1:degree))
    end
end

"""
    evaluate_at_frame(p::Polynomial1D, frame::Int)

Evaluate polynomial drift at a specific frame number.
Returns the drift value (not the corrected coordinate).
"""
function evaluate_at_frame(p::Polynomial1D, frame::Int)
    val = 0.0
    for nn in 1:p.degree
        val += p.coefficients[nn] * frame^nn
    end
    return val
end

"""
    evaluate_drift(intra::IntraPolynomial, frame::Int)

Evaluate intra-dataset drift at a specific frame.
Returns vector of drift values [dx, dy] or [dx, dy, dz].
"""
function evaluate_drift(intra::IntraPolynomial, frame::Int)
    drift = zeros(intra.ndims)
    for dim in 1:intra.ndims
        drift[dim] = evaluate_at_frame(intra.dm[dim], frame)
    end
    return drift
end

"""
    endpoint_drift(intra::IntraPolynomial, n_frames::Int)

Evaluate drift at the endpoint (last frame) of a dataset.
"""
function endpoint_drift(intra::IntraPolynomial, n_frames::Int)
    return evaluate_drift(intra, n_frames)
end

"""
    startpoint_drift(intra::IntraPolynomial)

Evaluate drift at the startpoint (frame 1) of a dataset.
For standard polynomials, this is small (c1 + c2 + ... ≈ small).
"""
function startpoint_drift(intra::IntraPolynomial)
    return evaluate_drift(intra, 1)
end
