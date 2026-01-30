# Main interface for drift correction with progressive quality tiers

"""
    driftcorrect(smld; kwargs...) -> DriftResult

Main interface for drift correction. Uses Legendre polynomial model with
entropy-based cost function and adaptive KDTree neighbor building.

# Arguments
- `smld`: SMLD structure containing (X, Y) or (X, Y, Z) localization coordinates (μm)

# Keyword Arguments
- `quality=:singlepass`: Quality tier (`:fft`, `:singlepass`, `:iterative`)
- `degree=2`: Polynomial degree for intra-dataset drift model
- `dataset_mode=:registered`: Semantic label for multi-dataset handling:
    - `:registered`: datasets are independent acquisitions
    - `:continuous`: one long acquisition split into files
- `chunk_frames=0`: For continuous mode, split each dataset into chunks of this many frames
- `n_chunks=0`: Alternative to chunk_frames - specify number of chunks per dataset
- `maxn=200`: Maximum number of neighbors for entropy calculation
- `max_iterations=10`: Maximum iterations for `:iterative` mode
- `convergence_tol=0.001`: Convergence tolerance (μm) for `:iterative` mode
- `verbose=0`: Verbosity level (0=quiet, 1=info, 2=debug)

# Quality Tiers
- `:fft`: Fast cross-correlation only (~10x faster, less accurate)
- `:singlepass`: Current algorithm - parallel intra, then sequential inter (default)
- `:iterative`: Full convergence - iterates intra↔inter until shift changes < tol

# Returns
`DriftResult` with fields:
- `smld`: Drift-corrected SMLD
- `model`: Fitted drift model (LegendrePolynomial)
- `iterations`: Number of iterations completed
- `converged`: Whether convergence was achieved
- `entropy`: Final entropy value
- `history`: Entropy per iteration (for diagnostics)

# Example
```julia
# Basic usage
result = driftcorrect(smld)
corrected_smld = result.smld

# Fast FFT-only mode
result = driftcorrect(smld; quality=:fft)

# Full iterative convergence
result = driftcorrect(smld; quality=:iterative)

# Continue from previous result
result2 = driftcorrect(result; max_iterations=5)

# Extract drift trajectory for plotting
traj = drift_trajectory(result.model)
```
"""
function driftcorrect(smld::SMLD;
    quality::Symbol = :singlepass,
    degree::Int = 2,
    dataset_mode::Symbol = :registered,
    chunk_frames::Int = 0,
    n_chunks::Int = 0,
    maxn::Int = 200,
    max_iterations::Int = 10,
    convergence_tol::Float64 = 0.001,
    verbose::Int = 0)

    # Validate quality tier
    if quality ∉ (:fft, :singlepass, :iterative)
        error("Unknown quality: $quality. Use :fft, :singlepass, or :iterative")
    end

    # Validate dataset_mode
    if dataset_mode ∉ (:registered, :continuous)
        error("Unknown dataset_mode: $dataset_mode. Use :registered or :continuous")
    end

    # Handle chunking for continuous mode
    chunk_info = nothing
    smld_work = smld

    if dataset_mode == :continuous && (chunk_frames > 0 || n_chunks > 0)
        chunk_info = chunk_smld(smld; chunk_frames=chunk_frames, n_chunks=n_chunks)
        smld_work = chunk_info.smld

        if verbose > 0
            @info("SMLMDriftCorrection: chunking into $(chunk_info.n_chunks) chunks per dataset " *
                  "($(chunk_info.frames_per_chunk) frames each, $(smld_work.n_datasets) total chunks)")
        end
    end

    # Create drift model
    driftmodel = LegendrePolynomial(smld_work; degree=degree)

    # Dispatch to appropriate quality tier
    if quality == :fft
        result = _driftcorrect_fft!(driftmodel, smld_work, dataset_mode, verbose)
    elseif quality == :singlepass
        result = _driftcorrect_singlepass!(driftmodel, smld_work, dataset_mode, maxn, verbose)
    else  # :iterative
        result = _driftcorrect_iterative!(driftmodel, smld_work, dataset_mode, maxn,
                                          max_iterations, convergence_tol, verbose)
    end

    # Apply corrections to get final SMLD
    smld_corrected = _apply_final_corrections(smld, smld_work, driftmodel, chunk_info, dataset_mode)

    # Compute final entropy
    final_entropy = _compute_entropy(smld_corrected, maxn)

    return DriftResult(
        smld_corrected,
        driftmodel,
        result.iterations,
        result.converged,
        final_entropy,
        result.history
    )
end

"""
    driftcorrect(result::DriftResult; kwargs...) -> DriftResult

Continue drift correction from a previous result. Creates a new DriftResult.

# Keyword Arguments
- `max_iterations=10`: Additional iterations to run
- `convergence_tol=0.001`: Convergence tolerance (μm)
- `maxn=200`: Maximum neighbors for entropy calculation
- `verbose=0`: Verbosity level
"""
function driftcorrect(result::DriftResult;
    max_iterations::Int = 10,
    convergence_tol::Float64 = 0.001,
    maxn::Int = 200,
    verbose::Int = 0)

    # Deep copy to avoid modifying original
    smld = deepcopy(result.smld)
    model = deepcopy(result.model)

    # Continue with iterative refinement
    iter_result = _driftcorrect_iterate!(model, smld, maxn, max_iterations,
                                          convergence_tol, verbose,
                                          result.iterations, copy(result.history))

    # Apply corrections
    smld_corrected = correctdrift(smld, model)

    # Compute final entropy
    final_entropy = _compute_entropy(smld_corrected, maxn)

    return DriftResult(
        smld_corrected,
        model,
        iter_result.iterations,
        iter_result.converged,
        final_entropy,
        iter_result.history
    )
end

"""
    driftcorrect!(result::DriftResult; kwargs...) -> DriftResult

Continue drift correction from a previous result, modifying it in place.

# Keyword Arguments
Same as `driftcorrect(result; kwargs...)`
"""
function driftcorrect!(result::DriftResult;
    max_iterations::Int = 10,
    convergence_tol::Float64 = 0.001,
    maxn::Int = 200,
    verbose::Int = 0)

    # Continue with iterative refinement (modifies model in place)
    iter_result = _driftcorrect_iterate!(result.model, result.smld, maxn, max_iterations,
                                          convergence_tol, verbose,
                                          result.iterations, result.history)

    # Apply corrections
    correctdrift!(result.smld, result.model)

    # Update result fields
    result.iterations = iter_result.iterations
    result.converged = iter_result.converged
    result.entropy = _compute_entropy(result.smld, maxn)
    append!(result.history, iter_result.history[length(result.history)+1:end])

    return result
end

# ============================================================================
# Internal implementation functions
# ============================================================================

"""
FFT quality tier - fast cross-correlation only, no entropy optimization.
"""
function _driftcorrect_fft!(model::LegendrePolynomial, smld::SMLD,
                            dataset_mode::Symbol, verbose::Int)
    if verbose > 0
        @info("SMLMDriftCorrection: FFT mode - cross-correlation alignment only")
    end

    n_dims = nDims(smld)

    # For :fft mode, degree is effectively 0 (no intra-dataset correction)
    # Just compute inter-dataset shifts using cross-correlation

    if smld.n_datasets > 1
        if dataset_mode == :continuous
            # Chain from previous: CC finds LOCAL shift relative to previous chunk
            for nn = 2:smld.n_datasets
                smld_prev = filter_by_dataset(smld, nn - 1)
                smld_n = filter_by_dataset(smld, nn)

                # Apply previous inter-shift to get reference
                smld_prev_shifted = deepcopy(smld_prev)
                for i in eachindex(smld_prev_shifted.emitters)
                    smld_prev_shifted.emitters[i].x -= model.inter[nn-1].dm[1]
                    smld_prev_shifted.emitters[i].y -= model.inter[nn-1].dm[2]
                    if n_dims == 3
                        smld_prev_shifted.emitters[i].z -= model.inter[nn-1].dm[3]
                    end
                end

                try
                    cc_shift = findshift(smld_prev_shifted, smld_n; histbinsize=0.05)
                    # Chain: inter[n] = inter[n-1] + local_shift
                    for dim in 1:n_dims
                        model.inter[nn].dm[dim] = model.inter[nn-1].dm[dim] - cc_shift[dim]
                    end
                catch e
                    if verbose > 0
                        @warn("SMLMDriftCorrection: FFT failed for dataset $nn, keeping zero shift")
                    end
                end
            end
        else  # :registered - align each to dataset 1
            smld_ref = filter_by_dataset(smld, 1)
            for nn = 2:smld.n_datasets
                smld_n = filter_by_dataset(smld, nn)
                try
                    cc_shift = findshift(smld_ref, smld_n; histbinsize=0.05)
                    for dim in 1:n_dims
                        model.inter[nn].dm[dim] = -cc_shift[dim]
                    end
                catch e
                    if verbose > 0
                        @warn("SMLMDriftCorrection: FFT failed for dataset $nn, keeping zero shift")
                    end
                end
            end
        end
    end

    return (iterations=0, converged=true, history=Float64[])
end

"""
Singlepass quality tier - current algorithm with all-to-all inter refinement.
"""
function _driftcorrect_singlepass!(model::LegendrePolynomial, smld::SMLD,
                                    dataset_mode::Symbol, maxn::Int, verbose::Int)
    if verbose > 0
        @info("SMLMDriftCorrection: singlepass mode")
    end

    # Intra-dataset correction (parallel over datasets)
    if model.intra[1].dm[1].degree > 0
        if verbose > 0
            @info("SMLMDriftCorrection: starting intra-dataset correction")
        end
        Threads.@threads for nn = 1:smld.n_datasets
            findintra!(model.intra[nn], smld, nn, maxn)
        end
    else
        if verbose > 0
            @info("SMLMDriftCorrection: degree=0, skipping intra-dataset correction")
        end
    end

    # Warm start for continuous mode
    if dataset_mode == :continuous && smld.n_datasets > 1
        _warmstart_inter_continuous!(model, smld, verbose)
    end

    # Inter-dataset correction: align each to dataset 1 first
    for nn = 2:smld.n_datasets
        findinter!(model, smld, nn, [1], maxn)
    end

    # Refine: align each dataset to ALL others (not just previous)
    if verbose > 0
        @info("SMLMDriftCorrection: refining inter-dataset alignment (all-to-all)")
    end
    for nn = 2:smld.n_datasets
        others = collect(setdiff(1:smld.n_datasets, nn))
        findinter!(model, smld, nn, others, maxn)
    end

    # Normalize for continuous mode
    if dataset_mode == :continuous
        _normalize_continuous!(model)
    end

    return (iterations=1, converged=true, history=Float64[])
end

"""
Iterative quality tier - full intra↔inter convergence loop.
"""
function _driftcorrect_iterative!(model::LegendrePolynomial, smld::SMLD,
                                   dataset_mode::Symbol, maxn::Int,
                                   max_iterations::Int, convergence_tol::Float64,
                                   verbose::Int)
    if verbose > 0
        @info("SMLMDriftCorrection: iterative mode (max_iterations=$max_iterations, tol=$convergence_tol)")
    end

    # First run singlepass to initialize
    _driftcorrect_singlepass!(model, smld, dataset_mode, maxn, verbose)

    # Compute initial entropy
    smld_corrected = correctdrift(smld, model)
    initial_entropy = _compute_entropy(smld_corrected, maxn)
    history = Float64[initial_entropy]

    if verbose > 0
        @info("SMLMDriftCorrection: initial entropy = $initial_entropy")
    end

    # Continue with iterative refinement
    result = _driftcorrect_iterate!(model, smld, maxn, max_iterations - 1,
                                     convergence_tol, verbose, 1, history)

    return result
end

"""
Core iterative refinement loop - used by both :iterative and continuation.
"""
function _driftcorrect_iterate!(model::LegendrePolynomial, smld::SMLD,
                                 maxn::Int, max_iterations::Int,
                                 convergence_tol::Float64, verbose::Int,
                                 starting_iteration::Int, history::Vector{Float64})

    n_datasets = smld.n_datasets
    n_dims = model.intra[1].ndims
    converged = false
    iteration = starting_iteration

    for iter = 1:max_iterations
        iteration += 1

        # Store current inter-shifts
        inter_old = [copy(model.inter[n].dm) for n in 1:n_datasets]

        # Re-run intra with inter applied (shifted coordinates)
        smld_shifted = apply_inter_only(smld, model)
        Threads.@threads for nn = 1:n_datasets
            findintra!(model.intra[nn], smld_shifted, nn, maxn)
        end

        # All-to-all inter: align each dataset to all others
        for nn = 2:n_datasets
            others = collect(setdiff(1:n_datasets, nn))
            findinter!(model, smld, nn, others, maxn)
        end

        # Compute entropy for this iteration
        smld_corrected = correctdrift(smld, model)
        current_entropy = _compute_entropy(smld_corrected, maxn)
        push!(history, current_entropy)

        # Check convergence
        max_change = _max_inter_change(inter_old, model.inter)

        if verbose > 0
            @info("SMLMDriftCorrection: iteration $iteration, entropy=$current_entropy, max_shift_change=$max_change")
        end

        if max_change < convergence_tol
            converged = true
            if verbose > 0
                @info("SMLMDriftCorrection: converged after $iteration iterations")
            end
            break
        end
    end

    return (iterations=iteration, converged=converged, history=history)
end

# ============================================================================
# Helper functions
# ============================================================================

"""
Apply only inter-dataset shifts (not intra) - used for re-running intra in iterative mode.
"""
function apply_inter_only(smld::SMLD, model::LegendrePolynomial)
    smld_shifted = deepcopy(smld)
    n_dims = nDims(smld)

    for e in smld_shifted.emitters
        e.x -= model.inter[e.dataset].dm[1]
        e.y -= model.inter[e.dataset].dm[2]
        if n_dims == 3
            e.z -= model.inter[e.dataset].dm[3]
        end
    end

    return smld_shifted
end

"""
Check maximum change in inter-shifts between iterations.
"""
function _max_inter_change(inter_old::Vector{Vector{Float64}},
                            inter_new::Vector{InterShift})
    max_change = 0.0
    for n in eachindex(inter_old)
        for dim in 1:length(inter_old[n])
            Δ = abs(inter_new[n].dm[dim] - inter_old[n][dim])
            max_change = max(max_change, Δ)
        end
    end
    return max_change
end

"""
Warm start inter-shifts for continuous mode from polynomial endpoints.
"""
function _warmstart_inter_continuous!(model::LegendrePolynomial, smld::SMLD, verbose::Int)
    ndims = model.intra[1].ndims
    for nn = 2:smld.n_datasets
        # Chain: inter[n] = inter[n-1] + endpoint(n-1) - startpoint(n)
        endpoint_prev = endpoint_drift(model.intra[nn-1], smld.n_frames)
        startpoint_curr = startpoint_drift(model.intra[nn])
        for dim in 1:ndims
            model.inter[nn].dm[dim] = model.inter[nn-1].dm[dim] +
                                      endpoint_prev[dim] - startpoint_curr[dim]
        end
    end
    if verbose > 0
        @info("SMLMDriftCorrection: initialized inter-shifts from polynomial endpoints")
    end
end

"""
Normalize inter-shifts for continuous mode so drift at (DS=1, frame=1) = 0.
"""
function _normalize_continuous!(model::LegendrePolynomial)
    ndims = model.intra[1].ndims
    for dim in 1:ndims
        offset = evaluate_at_frame(model.intra[1].dm[dim], 1) + model.inter[1].dm[dim]
        for nn = 1:model.ndatasets
            model.inter[nn].dm[dim] -= offset
        end
    end
end

"""
Apply final corrections, handling chunking if applicable.
"""
function _apply_final_corrections(smld_original::SMLD, smld_work::SMLD,
                                   model::LegendrePolynomial,
                                   chunk_info, dataset_mode::Symbol)
    if chunk_info !== nothing && chunk_info.n_chunks > 1
        smld_work_corrected = correctdrift(smld_work, model)

        smld_corrected = deepcopy(smld_original)
        is_3d = nDims(smld_original) == 3
        for i in eachindex(smld_original.emitters)
            smld_corrected.emitters[i].x = smld_work_corrected.emitters[i].x
            smld_corrected.emitters[i].y = smld_work_corrected.emitters[i].y
            if is_3d
                smld_corrected.emitters[i].z = smld_work_corrected.emitters[i].z
            end
        end
        return smld_corrected
    else
        return correctdrift(smld_work, model)
    end
end

"""
Compute entropy of corrected SMLD.
"""
function _compute_entropy(smld::SMLD, maxn::Int)
    x = Float64[e.x for e in smld.emitters]
    y = Float64[e.y for e in smld.emitters]
    σ_x = Float64[e.σ_x for e in smld.emitters]
    σ_y = Float64[e.σ_y for e in smld.emitters]

    if nDims(smld) == 3
        z = Float64[e.z for e in smld.emitters]
        σ_z = Float64[e.σ_z for e in smld.emitters]
        return ub_entropy(x, y, z, σ_x, σ_y, σ_z; maxn=maxn)
    else
        return ub_entropy(x, y, σ_x, σ_y; maxn=maxn)
    end
end
