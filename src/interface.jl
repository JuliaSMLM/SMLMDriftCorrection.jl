"""
    driftcorrect(smld; kwargs...) -> (smld, model)

Main interface for drift correction. Uses Legendre polynomial model with
entropy-based cost function and adaptive KDTree neighbor building.

# Arguments
- `smld`: SMLD structure containing (X, Y) or (X, Y, Z) localization coordinates (μm)

# Keyword Arguments
- `degree=2`: polynomial degree for intra-dataset drift model
- `dataset_mode=:registered`: acquisition mode for multi-dataset handling:
    - `:registered`: datasets are independent (stage registered between acquisitions)
    - `:continuous`: drift accumulates across datasets (one long acquisition split into files)
- `chunk_frames=0`: for continuous mode, split each dataset into chunks of this many frames (0 = no chunking)
- `n_chunks=0`: alternative to chunk_frames - specify number of chunks per dataset (0 = no chunking)
- `maxn=200`: maximum number of neighbors for entropy calculation
- `verbose=0`: verbosity level (0=quiet, 1=info, 2=debug)

# Returns
NamedTuple with fields:
- `smld`: drift-corrected SMLD
- `model`: fitted drift model (LegendrePolynomial)
  - Use `drift_trajectory(model)` to extract plottable arrays

# Example
```julia
# Basic usage
result = driftcorrect(smld)
corrected_smld = result.smld

# Or destructure directly:
(; smld, model) = driftcorrect(smld)
smld_corr, drift_model = driftcorrect(smld)  # tuple-style also works

# Higher degree polynomial for complex drift:
result = driftcorrect(smld; degree=3)

# For continuous acquisition (drift accumulates across datasets):
result = driftcorrect(smld; dataset_mode=:continuous)

# For finer-grained drift correction, chunk into smaller pieces:
result = driftcorrect(smld; dataset_mode=:continuous, n_chunks=10)

# Extract drift trajectory for plotting:
traj = drift_trajectory(result.model)
# traj.frames, traj.x, traj.y (and traj.z for 3D) are ready for plotting
```
"""
function driftcorrect(smld::SMLD;
    degree::Int = 2,
    dataset_mode::Symbol = :registered,
    chunk_frames::Int = 0,
    n_chunks::Int = 0,
    maxn::Int = 200,
    verbose::Int = 0)

    # Handle chunking for continuous mode
    chunk_info = nothing
    smld_work = smld  # Working SMLD (may be chunked)

    if dataset_mode == :continuous && (chunk_frames > 0 || n_chunks > 0)
        chunk_info = chunk_smld(smld; chunk_frames=chunk_frames, n_chunks=n_chunks)
        smld_work = chunk_info.smld

        if verbose > 0
            @info("SMLMDriftCorrection: chunking into $(chunk_info.n_chunks) chunks per dataset " *
                  "($(chunk_info.frames_per_chunk) frames each, $(smld_work.n_datasets) total chunks)")
        end
    end

    # Use Legendre polynomial model
    driftmodel = LegendrePolynomial(smld_work; degree=degree)

    # Intra-dataset correction (parallel over datasets)
    # Skip if degree=0 (no intra-dataset parameters to optimize)
    if degree > 0
        if verbose > 0
            @info("SMLMDriftCorrection: starting intra-dataset correction")
        end
        Threads.@threads for nn = 1:smld_work.n_datasets
            findintra!(driftmodel.intra[nn], smld_work, nn, maxn)
        end
    else
        if verbose > 0
            @info("SMLMDriftCorrection: degree=0, skipping intra-dataset correction")
        end
    end

    # Inter-dataset correction
    if dataset_mode == :continuous
        # Continuous mode: initialize inter-shifts from endpoints, then optimize
        if verbose > 0
            @info("SMLMDriftCorrection: continuous mode - chaining inter-dataset shifts")
        end

        ndims = driftmodel.intra[1].ndims
        driftmodel.inter[1].dm .= 0.0

        for nn = 2:smld_work.n_datasets
            endpoint_prev = endpoint_drift(driftmodel.intra[nn-1], smld_work.n_frames)
            startpoint_curr = startpoint_drift(driftmodel.intra[nn])

            for dim in 1:ndims
                driftmodel.inter[nn].dm[dim] = driftmodel.inter[nn-1].dm[dim] +
                                                endpoint_prev[dim] -
                                                startpoint_curr[dim]
            end

            if verbose > 1
                println("  Chunk $nn initial inter: $(driftmodel.inter[nn].dm)")
            end

            findinter!(driftmodel, smld_work, nn, collect(1:(nn-1)), maxn)
        end

    elseif dataset_mode == :registered
        # Registered mode: datasets are independent
        if verbose > 0
            @info("SMLMDriftCorrection: registered mode - aligning to dataset 1")
        end
        for nn = 2:smld_work.n_datasets
            findinter!(driftmodel, smld_work, nn, [1], maxn)
        end

        if verbose > 0
            @info("SMLMDriftCorrection: refining inter-dataset alignment")
        end
        for ii = 2:smld_work.n_datasets
            findinter!(driftmodel, smld_work, ii, collect(1:(ii-1)), maxn)
        end

    else
        error("Unknown dataset_mode: $dataset_mode. Use :registered or :continuous")
    end

    # Apply corrections
    if chunk_info !== nothing && chunk_info.n_chunks > 1
        smld_work_corrected = correctdrift(smld_work, driftmodel)

        smld_corrected = deepcopy(smld)
        is_3d = nDims(smld) == 3
        for i in eachindex(smld.emitters)
            smld_corrected.emitters[i].x = smld_work_corrected.emitters[i].x
            smld_corrected.emitters[i].y = smld_work_corrected.emitters[i].y
            if is_3d
                smld_corrected.emitters[i].z = smld_work_corrected.emitters[i].z
            end
        end
    else
        smld_corrected = correctdrift(smld, driftmodel)
    end

    return (smld=smld_corrected, model=driftmodel)
end
