"""
Select the emitters indexed in `keep` from the SMLD structure.
`keep` may be a single positive integer, a range or a vector of positive
integers.

"""
function filter_emitters(smld::SMLD, keep::Union{AbstractVector,AbstractRange})
    return typeof(smld)(
        smld.emitters[keep],
        smld.camera,
        smld.n_frames,
        smld.n_datasets,
        copy(smld.metadata)
    )
end

function filter_emitters(smld::SMLD, keep::Integer)
    return(filter_emitters(smld, [keep]))
end

"""
Determines from the type of the input smld if the data is 2D or 3D.
"""
function nDims(smld::SMLD)
    return occursin("2D", string(typeof(smld))) ? 2 : 3
end
