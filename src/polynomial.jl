# Define polynomial methods

mutable struct Polynomial1D <: DriftModel1D
    degree::Int
    coefficients::Vector{AbstractFloat}
end
function Polynomial1D(degree::Int)
    return Polynomial1D(degree,zeros(degree+1))
end


mutable struct Polynomial <: DriftModel
    ndatasets::Int
    ndims::Int
    model::Array{Polynomial1D}
end

function Polynomial(smld::SMLMData.SMLD2D; degree::Int=2, initialize::String="zeros")
    
    dm=Polynomial(smld.ndatasets,2,Array{Polynomial1D}(undef,smld.ndatasets,2))

    if initialize=="zeros"
        for ii=1:length(dm.model)
            dm.model[ii]=Polynomial1D(degree)
        end
    end

    if initialize=="random"
        for ii=1:length(dm.model)
            dm.model[ii]=Polynomial1D(degree)
            dm.model[ii].coefficients=0.1*randn()./(smld.nframes.^(0:degree))
        end
    end

    return dm
end



function applydrift(p::Polynomial1D,x::AbstractFloat,framenum::Int)
    for nn=0:p.degree
        x+=p.coefficients[nn+1]*framenum^nn
    end
    return x
end

function correctdrift(p::Polynomial1D,x::AbstractFloat,framenum::Int)
    for nn=0:p.degree
        x-=p.coefficients[nn+1]*framenum^nn
    end
    return x
end

function applydrift!(smld::SMLMData.SMLD2D,driftmodel::Polynomial)
    for nn=1:length(smld.x)        
        smld.x[nn]=applydrift(driftmodel.model[smld.datasetnum[nn],1],smld.x[nn],smld.framenum[nn])
        smld.y[nn]=applydrift(driftmodel.model[smld.datasetnum[nn],2],smld.y[nn],smld.framenum[nn])
    end
end

function applydrift(smld::SMLMData.SMLD2D,driftmodel::Polynomial)
    smld_shifted=deepcopy(smld)
    applydrift!(smld_shifted::SMLMData.SMLD2D,driftmodel::Polynomial)
    return smld_shifted
end

function correctdrift!(smld::SMLMData.SMLD2D,driftmodel::Polynomial)    
    for nn=1:length(smld.x)        
        smld.x[nn]=correctdrift(driftmodel.model[smld.datasetnum[nn],1],smld.x[nn],smld.framenum[nn])
        smld.y[nn]=correctdrift(driftmodel.model[smld.datasetnum[nn],2],smld.y[nn],smld.framenum[nn])
    end
end

function correctdrift(smld::SMLMData.SMLD2D,driftmodel::Polynomial)
    smld_shifted=deepcopy(smld)
    correctdrift!(smld_shifted::SMLMData.SMLD2D,driftmodel::Polynomial)
    return smld_shifted
end







