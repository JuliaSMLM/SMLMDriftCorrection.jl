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

"""
    compute_chunk_params(n_frames; chunk_frames=0, n_chunks=0)

Compute chunking parameters for continuous drift correction.
Returns (frames_per_chunk, n_chunks) where all chunks have equal size
except the last which absorbs any remainder.

# Arguments
- `n_frames`: total number of frames
- `chunk_frames`: desired frames per chunk (0 = no chunking)
- `n_chunks`: desired number of chunks (0 = no chunking)

If both are specified, `n_chunks` takes priority.
"""
function compute_chunk_params(n_frames::Int; chunk_frames::Int=0, n_chunks::Int=0)
    if n_chunks > 0
        # n_chunks specified: divide evenly
        frames_per_chunk = n_frames ÷ n_chunks
        return (frames_per_chunk=frames_per_chunk, n_chunks=n_chunks)
    elseif chunk_frames > 0
        # chunk_frames specified: compute n_chunks, then make equal
        n_chunks = max(1, round(Int, n_frames / chunk_frames))
        frames_per_chunk = n_frames ÷ n_chunks
        return (frames_per_chunk=frames_per_chunk, n_chunks=n_chunks)
    else
        # No chunking
        return (frames_per_chunk=n_frames, n_chunks=1)
    end
end

"""
    chunk_smld(smld; chunk_frames=0, n_chunks=0)

Split an SMLD into chunks for finer-grained drift correction.
Returns a new SMLD where:
- Original dataset boundaries are preserved
- Within each dataset, frames are grouped into chunks
- New "dataset" number = (original_dataset - 1) * n_chunks + chunk_index
- Frame numbers are converted to per-chunk (1 to frames_per_chunk)

Also returns chunk metadata needed to reconstruct original structure.

# Returns
NamedTuple with:
- `smld`: chunked SMLD with reassigned dataset/frame numbers
- `n_chunks`: number of chunks per original dataset
- `frames_per_chunk`: frames in each chunk (last may have more)
- `original_n_datasets`: original number of datasets
- `original_n_frames`: original n_frames
"""
function chunk_smld(smld::SMLD; chunk_frames::Int=0, n_chunks::Int=0)
    params = compute_chunk_params(smld.n_frames; chunk_frames=chunk_frames, n_chunks=n_chunks)

    if params.n_chunks == 1
        # No chunking needed
        return (
            smld=smld,
            n_chunks=1,
            frames_per_chunk=smld.n_frames,
            original_n_datasets=smld.n_datasets,
            original_n_frames=smld.n_frames
        )
    end

    frames_per_chunk = params.frames_per_chunk
    n_chunks_per_dataset = params.n_chunks

    # Create new emitters with reassigned dataset and frame numbers
    new_emitters = similar(smld.emitters)
    is_3d = nDims(smld) == 3

    for (i, e) in enumerate(smld.emitters)
        # Compute which chunk this frame belongs to (1-indexed)
        # Last chunk absorbs remainder
        chunk_idx = min(n_chunks_per_dataset, ((e.frame - 1) ÷ frames_per_chunk) + 1)

        # New dataset = (original_dataset - 1) * n_chunks + chunk_idx
        new_dataset = (e.dataset - 1) * n_chunks_per_dataset + chunk_idx

        # New frame = frame within chunk
        chunk_start = (chunk_idx - 1) * frames_per_chunk + 1
        new_frame = e.frame - chunk_start + 1

        # Create new emitter with updated dataset and frame
        # Emitter constructors use positional args: (coords..., photons, bg, σs..., frame, dataset, track_id, id)
        if is_3d
            new_emitters[i] = typeof(e)(
                e.x, e.y, e.z, e.photons, e.bg,
                e.σ_x, e.σ_y, e.σ_z, e.σ_photons, e.σ_bg,
                new_frame, new_dataset, e.track_id, e.id
            )
        else
            new_emitters[i] = typeof(e)(
                e.x, e.y, e.photons, e.bg,
                e.σ_x, e.σ_y, e.σ_photons, e.σ_bg,
                new_frame, new_dataset, e.track_id, e.id
            )
        end
    end

    # Compute frames in last chunk (absorbs remainder)
    last_chunk_frames = smld.n_frames - (n_chunks_per_dataset - 1) * frames_per_chunk

    new_n_datasets = smld.n_datasets * n_chunks_per_dataset

    chunked_smld = typeof(smld)(
        new_emitters,
        smld.camera,
        frames_per_chunk,  # Use standard chunk size (last chunk handled internally)
        new_n_datasets,
        copy(smld.metadata)
    )

    return (
        smld=chunked_smld,
        n_chunks=n_chunks_per_dataset,
        frames_per_chunk=frames_per_chunk,
        last_chunk_frames=last_chunk_frames,
        original_n_datasets=smld.n_datasets,
        original_n_frames=smld.n_frames
    )
end
