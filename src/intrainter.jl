# Intra+Inter 

abstract type AbstractIntraDrift1D end

abstract type AbstractIntraDrift end

abstract type AbstractIntraInter <: AbstractDriftModel end

mutable struct InterShift 
    ndims::Int
    dm::Vector{Real}    
end
function InterShift(ndims::Int)
    return InterShift(ndims,zeros(ndims))
end

function applydrift(x::AbstractFloat,s::InterShift,dim::Int)
    return x+s.dm[dim]
end

function correctdrift(x::AbstractFloat,s::InterShift,dim::Int)
    return x-s.dm[dim]
end

function applydrift!(smld::SMLMData.SMLD2D,dm::AbstractIntraInter)
    for nn=1:length(smld.x)        
        smld.x[nn]=applydrift(smld.x[nn],smld.framenum[nn],dm.intra[smld.datasetnum[nn]].dm[1])
        smld.x[nn]=applydrift(smld.x[nn],dm.inter[smld.datasetnum[nn]],1)
       
        smld.y[nn]=applydrift(smld.y[nn],smld.framenum[nn],dm.intra[smld.datasetnum[nn]].dm[2])
        smld.y[nn]=applydrift(smld.y[nn],dm.inter[smld.datasetnum[nn]],2)
    end
end

function applydrift(smld::SMLMData.SMLD2D,driftmodel::AbstractIntraInter)
    smld_shifted=deepcopy(smld)
    applydrift!(smld_shifted::SMLMData.SMLD2D,driftmodel::AbstractIntraInter)
    return smld_shifted
end

function correctdrift!(smld::SMLMData.SMLD2D,dm::AbstractIntraInter)    
    for nn=1:length(smld.x)        
        smld.x[nn]=correctdrift(smld.x[nn],smld.framenum[nn],dm.intra[smld.datasetnum[nn]].dm[1])
        smld.x[nn]=correctdrift(smld.x[nn],dm.inter[smld.datasetnum[nn]],1)
       
        smld.y[nn]=correctdrift(smld.y[nn],smld.framenum[nn],dm.intra[smld.datasetnum[nn]].dm[2])
        smld.y[nn]=correctdrift(smld.y[nn],dm.inter[smld.datasetnum[nn]],2)
    end
end

function correctdrift(smld::SMLMData.SMLD2D,driftmodel::AbstractIntraInter)
    smld_shifted=deepcopy(smld)
    correctdrift!(smld_shifted,driftmodel)
    return smld_shifted
end
