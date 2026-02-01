"""
    driftcorrect(smld; kwargs...) -> (smld, info)

Main interface for drift correction. Uses Legendre polynomial model with
entropy-based cost function and adaptive KDTree neighbor building.

# Arguments
- `smld`: SMLD structure containing (X, Y) or (X, Y, Z) localization coordinates (μm)

# Keyword Arguments
- `degree=2`: polynomial degree for intra-dataset drift model
- `dataset_mode=:registered`: semantic label for multi-dataset handling (algorithm is identical):
    - `:registered`: datasets are independent acquisitions (use default trajectory plotting)
    - `:continuous`: one long acquisition split into files (use `drift_trajectory(model; cumulative=true)` for plotting)
- `chunk_frames=0`: for continuous mode, split each dataset into chunks of this many frames (0 = no chunking)
- `n_chunks=0`: alternative to chunk_frames - specify number of chunks per dataset (0 = no chunking)
- `maxn=200`: maximum number of neighbors for entropy calculation
- `verbose=0`: verbosity level (0=quiet, 1=info, 2=debug)

# Returns
Tuple `(smld_corrected, info)` where:
- `smld_corrected`: drift-corrected SMLD
- `info`: `DriftInfo` struct with optimization metadata including:
  - `model`: fitted drift model (use `drift_trajectory(info.model)` to extract plottable arrays)
  - `elapsed_ns`: wall time in nanoseconds
  - `iterations`: total optimizer iterations
  - `converged`: whether all optimizations converged

# Example
```julia
# Basic usage
(smld_corrected, info) = driftcorrect(smld)

# Higher degree polynomial for complex drift:
(smld_corrected, info) = driftcorrect(smld; degree=3)

# For continuous acquisition (drift accumulates across datasets):
(smld_corrected, info) = driftcorrect(smld; dataset_mode=:continuous)

# For finer-grained drift correction, chunk into smaller pieces:
(smld_corrected, info) = driftcorrect(smld; dataset_mode=:continuous, n_chunks=10)

# Extract drift trajectory for plotting:
traj = drift_trajectory(info.model)
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

    t_start = time_ns()

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

    # Use Legendre polynomial model (normalized frame range prevents coefficient explosion)
    driftmodel = LegendrePolynomial(smld_work; degree=degree)

    # Track optimization results for DriftInfo
    intra_results = Vector{Any}(undef, smld_work.n_datasets)
    inter_costs = Float64[]

    # Intra-dataset correction (parallel over datasets)
    # Skip if degree=0 (no intra-dataset parameters to optimize)
    if degree > 0
        if verbose > 0
            @info("SMLMDriftCorrection: starting intra-dataset correction")
        end
        Threads.@threads for nn = 1:smld_work.n_datasets
            intra_results[nn] = findintra!(driftmodel.intra[nn], smld_work, nn, maxn)
        end
    else
        if verbose > 0
            @info("SMLMDriftCorrection: degree=0, skipping intra-dataset correction")
        end
        # Create dummy results for DriftInfo
        for nn = 1:smld_work.n_datasets
            intra_results[nn] = (minimum=0.0, iterations=0, converged=true)
        end
    end

    # Inter-dataset correction
    # Both modes use the same entropy-based alignment algorithm.
    # The difference is semantic: :registered assumes independent acquisitions,
    # :continuous assumes one long acquisition split into files.
    # For visualization, use drift_trajectory(model; cumulative=true) with continuous mode.
    if dataset_mode ∉ (:registered, :continuous)
        error("Unknown dataset_mode: $dataset_mode. Use :registered or :continuous")
    end

    if verbose > 0
        @info("SMLMDriftCorrection: $dataset_mode mode - aligning datasets via entropy")
    end

    # For continuous mode, compute warm start for inter-shifts from polynomial endpoints.
    # This chains the drift across chunks: endpoint of chunk n-1 should match startpoint of chunk n.
    # Without this, the optimizer starts from 0 and can find spurious solutions.
    if dataset_mode == :continuous && smld_work.n_datasets > 1
        ndims = driftmodel.intra[1].ndims
        for nn = 2:smld_work.n_datasets
            # Chain: inter[n] = inter[n-1] + endpoint(n-1) - startpoint(n)
            endpoint_prev = endpoint_drift(driftmodel.intra[nn-1], smld_work.n_frames)
            startpoint_curr = startpoint_drift(driftmodel.intra[nn])
            for dim in 1:ndims
                driftmodel.inter[nn].dm[dim] = driftmodel.inter[nn-1].dm[dim] +
                                               endpoint_prev[dim] - startpoint_curr[dim]
            end
        end
        if verbose > 0
            @info("SMLMDriftCorrection: initialized inter-shifts from polynomial endpoints")
        end
    end

    # Align each dataset to dataset 1 first
    for nn = 2:smld_work.n_datasets
        cost = findinter!(driftmodel, smld_work, nn, [1], maxn)
        push!(inter_costs, cost)
    end

    # Refine by aligning to all previous datasets
    if verbose > 0
        @info("SMLMDriftCorrection: refining inter-dataset alignment")
    end
    for nn = 2:smld_work.n_datasets
        cost = findinter!(driftmodel, smld_work, nn, collect(1:(nn-1)), maxn)
        push!(inter_costs, cost)
    end

    # For continuous mode, normalize so drift at (DS=1, frame=1) = 0
    # This removes the global offset ambiguity for continuous acquisitions
    # For registered mode, keep inter[1]=0 convention (set by algorithm)
    if dataset_mode == :continuous
        ndims = driftmodel.intra[1].ndims
        for dim in 1:ndims
            offset = evaluate_at_frame(driftmodel.intra[1].dm[dim], 1) + driftmodel.inter[1].dm[dim]
            for nn = 1:smld_work.n_datasets
                driftmodel.inter[nn].dm[dim] -= offset
            end
        end
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

    elapsed_ns = time_ns() - t_start

    # Aggregate optimization statistics
    total_iterations = sum(Optim.iterations(r) for r in intra_results)
    all_converged = all(Optim.converged(r) for r in intra_results)
    intra_costs = [Optim.minimum(r) for r in intra_results]
    total_cost = sum(intra_costs) + sum(inter_costs)
    history = vcat(intra_costs, inter_costs)

    info = DriftInfo(
        driftmodel,
        elapsed_ns,
        :cpu,
        total_iterations,
        all_converged,
        total_cost,
        history
    )

    return (smld_corrected, info)
end
