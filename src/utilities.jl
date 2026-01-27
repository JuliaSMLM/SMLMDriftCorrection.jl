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

        # Create new emitter with updated dataset and frame using keyword constructors
        if is_3d
            new_emitters[i] = typeof(e)(;
                x=e.x, y=e.y, z=e.z, photons=e.photons, bg=e.bg,
                σ_x=e.σ_x, σ_y=e.σ_y, σ_z=e.σ_z, σ_photons=e.σ_photons, σ_bg=e.σ_bg,
                frame=new_frame, dataset=new_dataset, track_id=e.track_id, id=e.id
            )
        else
            new_emitters[i] = typeof(e)(;
                x=e.x, y=e.y, photons=e.photons, bg=e.bg,
                σ_x=e.σ_x, σ_y=e.σ_y, σ_photons=e.σ_photons, σ_bg=e.σ_bg,
                frame=new_frame, dataset=new_dataset, track_id=e.track_id, id=e.id
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

"""
    drift_trajectory(model; dataset=nothing, frames=nothing, cumulative=false)

Extract drift trajectory from a drift model for plotting.

# Arguments
- `model`: Polynomial or LegendrePolynomial drift model
- `dataset`: specific dataset to extract (default: all datasets)
- `frames`: frame range to evaluate (default: 1:n_frames for each dataset)
- `cumulative`: if true, chain datasets end-to-end showing total accumulated drift.
  Useful for registered acquisitions where you want to visualize what drift
  would look like without stage registration. Default: false.

# Returns
NamedTuple with fields ready for plotting:
- `frames`: frame numbers (global if multiple datasets)
- `x`: x-drift values (μm)
- `y`: y-drift values (μm)
- `z`: z-drift values (μm) - only present for 3D models
- `dataset`: dataset index for each point (useful for coloring)

# Example
```julia
result = driftcorrect(smld; dataset_mode=:registered)
traj = drift_trajectory(result.model)  # Each dataset relative to ds1

# Cumulative view - chains datasets end-to-end
traj_cumul = drift_trajectory(result.model; cumulative=true)
plot(traj_cumul.frames, traj_cumul.x, label="X drift (cumulative)")
```
"""
function drift_trajectory(model::AbstractIntraInter;
                          dataset::Union{Int,Nothing}=nothing,
                          frames::Union{AbstractRange,Nothing}=nothing,
                          cumulative::Bool=false)
    ndims = model.intra[1].ndims
    n_frames = model.n_frames

    if dataset !== nothing
        # Single dataset requested
        datasets_to_process = [dataset]
    else
        # All datasets
        datasets_to_process = 1:model.ndatasets
    end

    # Preallocate output arrays
    total_points = length(datasets_to_process) * n_frames
    all_frames = Vector{Int}(undef, total_points)
    all_x = Vector{Float64}(undef, total_points)
    all_y = Vector{Float64}(undef, total_points)
    all_z = ndims == 3 ? Vector{Float64}(undef, total_points) : Float64[]
    all_datasets = Vector{Int}(undef, total_points)

    idx = 1
    global_frame = 0

    # Cumulative offset - chains endpoint of ds N to start of ds N+1
    cumulative_offset = zeros(ndims)

    for ds in datasets_to_process
        # Frame range for this dataset
        ds_frames = frames !== nothing ? frames : 1:n_frames

        # For cumulative mode, compute offset to chain from previous dataset
        if cumulative && ds > first(datasets_to_process)
            # Start this dataset where the previous one ended
            # Previous endpoint is already in cumulative_offset from last iteration
            # Subtract this dataset's starting drift so it connects smoothly
            start_drift = evaluate_drift(model.intra[ds], 1)
            for dim in 1:ndims
                cumulative_offset[dim] -= start_drift[dim]
            end
        end

        for f in ds_frames
            global_frame += 1

            # Evaluate intra-dataset drift at this frame
            drift = evaluate_drift(model.intra[ds], f)

            all_frames[idx] = global_frame

            if cumulative
                # Cumulative mode: offset + intra drift only (no inter)
                all_x[idx] = cumulative_offset[1] + drift[1]
                all_y[idx] = cumulative_offset[2] + drift[2]
                if ndims == 3
                    all_z[idx] = cumulative_offset[3] + drift[3]
                end
            else
                # Standard mode: inter-shift + intra-polynomial
                all_x[idx] = model.inter[ds].dm[1] + drift[1]
                all_y[idx] = model.inter[ds].dm[2] + drift[2]
                if ndims == 3
                    all_z[idx] = model.inter[ds].dm[3] + drift[3]
                end
            end
            all_datasets[idx] = ds

            idx += 1
        end

        # Update cumulative offset with endpoint of this dataset
        if cumulative
            end_drift = evaluate_drift(model.intra[ds], n_frames)
            for dim in 1:ndims
                cumulative_offset[dim] += end_drift[dim]
            end
        end
    end

    # Trim to actual size (in case frames range was shorter)
    actual_length = idx - 1

    if ndims == 3
        return (
            frames = all_frames[1:actual_length],
            x = all_x[1:actual_length],
            y = all_y[1:actual_length],
            z = all_z[1:actual_length],
            dataset = all_datasets[1:actual_length]
        )
    else
        return (
            frames = all_frames[1:actual_length],
            x = all_x[1:actual_length],
            y = all_y[1:actual_length],
            dataset = all_datasets[1:actual_length]
        )
    end
end

"""
    drift_at_frame(model, dataset, frame)

Evaluate drift at a specific dataset and frame.
Returns vector [dx, dy] or [dx, dy, dz].

This is the low-level function for getting drift at arbitrary points.
For plotting trajectories, use `drift_trajectory()` instead.
"""
function drift_at_frame(model::AbstractIntraInter, dataset::Int, frame::Int)
    drift = evaluate_drift(model.intra[dataset], frame)
    ndims = model.intra[dataset].ndims

    result = zeros(ndims)
    for dim in 1:ndims
        result[dim] = model.inter[dataset].dm[dim] + drift[dim]
    end
    return result
end
