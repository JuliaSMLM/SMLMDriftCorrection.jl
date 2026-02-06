# Main interface for drift correction with progressive quality tiers

"""
    driftcorrect(smld; kwargs...) -> (corrected_smld, info)

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
- `warm_start=nothing`: Previous model for warm starting optimization
- `verbose=0`: Verbosity level (0=quiet, 1=info, 2=debug)
- `auto_roi=false`: Set to `true` for faster (but slightly less accurate) estimation using a dense ROI subset
- `σ_loc=0.010`: Typical localization precision (μm) for ROI sizing
- `σ_target=0.001`: Target drift precision (μm) for ROI sizing
- `roi_safety_factor=4.0`: Safety multiplier for required localizations

# Quality Tiers
- `:fft`: Fast cross-correlation only (~10x faster, less accurate)
- `:singlepass`: Current algorithm - parallel intra, then sequential inter (default)
- `:iterative`: Full convergence - iterates intra↔inter until shift changes < tol

# Returns
Tuple `(corrected_smld, info)` where `info::DriftInfo` contains:
- `model`: Fitted drift model (LegendrePolynomial)
- `elapsed_s`: Wall time in seconds
- `backend`: Computation backend (`:cpu`)
- `iterations`: Number of iterations completed
- `converged`: Whether convergence was achieved
- `entropy`: Final entropy value
- `history`: Entropy per iteration (for diagnostics)

# Example
```julia
# Basic usage
(smld_corrected, info) = driftcorrect(smld)

# Fast FFT-only mode
(smld_corrected, info) = driftcorrect(smld; quality=:fft)

# Full iterative convergence
(smld_corrected, info) = driftcorrect(smld; quality=:iterative)

# Warm start from previous result
(smld1, info1) = driftcorrect(smld1; degree=2)
(smld2, info2) = driftcorrect(smld2; warm_start=info1.model)

# Extract drift trajectory for plotting
traj = drift_trajectory(info.model)
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
    warm_start::Union{Nothing, AbstractIntraInter} = nothing,
    verbose::Int = 0,
    auto_roi::Bool = false,
    σ_loc::Float64 = 0.010,
    σ_target::Float64 = 0.001,
    roi_safety_factor::Float64 = 4.0)

    t_start = time_ns()

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

    # Handle automatic ROI subsampling for faster estimation
    smld_estimation = smld_work
    roi_indices = nothing  # Track which indices were used (nothing if not subsampled)

    if auto_roi
        n_locs_total = length(smld_work.emitters)
        n_locs_required = calculate_n_locs_required(smld_work.n_frames;
            degree=degree, σ_loc=σ_loc, σ_target=σ_target,
            safety_factor=roi_safety_factor)

        # Only subsample if we have significantly more data than needed (2x threshold)
        if n_locs_total > 2 * n_locs_required
            roi_indices = find_dense_roi(smld_work, n_locs_required)
            smld_estimation = filter_emitters(smld_work, roi_indices)
            if verbose > 0
                @info("SMLMDriftCorrection: auto_roi selected $(length(roi_indices)) of $n_locs_total locs for estimation")
            end
        elseif verbose > 1
            @info("SMLMDriftCorrection: auto_roi not applied (have $n_locs_total, need $n_locs_required)")
        end
    end

    # Create drift model (or use warm start)
    # Model is always created from full smld_work to preserve dataset structure
    if warm_start !== nothing
        driftmodel = deepcopy(warm_start)
        if verbose > 0
            @info("SMLMDriftCorrection: using warm start from previous model")
        end
    else
        driftmodel = LegendrePolynomial(smld_work; degree=degree)
    end

    # Dispatch to appropriate quality tier (using ROI-subsampled data for estimation)
    if quality == :fft
        result = _driftcorrect_fft!(driftmodel, smld_estimation, dataset_mode, verbose)
    elseif quality == :singlepass
        result = _driftcorrect_singlepass!(driftmodel, smld_estimation, dataset_mode, maxn, verbose)
    else  # :iterative
        result = _driftcorrect_iterative!(driftmodel, smld_estimation, dataset_mode, maxn,
                                          max_iterations, convergence_tol, verbose)
    end

    # Apply corrections to get final SMLD
    smld_corrected = _apply_final_corrections(smld, smld_work, driftmodel, chunk_info, dataset_mode)

    # Compute final entropy
    final_entropy = _compute_entropy(smld_corrected, maxn)

    elapsed_s = (time_ns() - t_start) / 1e9

    info = DriftInfo(
        driftmodel,
        elapsed_s,
        :cpu,
        result.iterations,
        result.converged,
        final_entropy,
        result.history,
        roi_indices
    )

    return (smld_corrected, info)
end

"""
    driftcorrect(smld, info::DriftInfo; kwargs...) -> (corrected_smld, info)

Continue drift correction from a previous result using the model from info.

# Keyword Arguments
- `dataset_mode=:registered`: Dataset mode (`:registered` or `:continuous`)
- `max_iterations=10`: Additional iterations to run
- `convergence_tol=0.001`: Convergence tolerance (μm)
- `maxn=200`: Maximum neighbors for entropy calculation
- `verbose=0`: Verbosity level
"""
function driftcorrect(smld::SMLD, info::DriftInfo;
    dataset_mode::Symbol = :registered,
    max_iterations::Int = 10,
    convergence_tol::Float64 = 0.001,
    maxn::Int = 200,
    verbose::Int = 0)

    t_start = time_ns()

    # Deep copy to avoid modifying original
    smld_work = deepcopy(smld)
    model = deepcopy(info.model)

    # Continue with iterative refinement
    iter_result = _driftcorrect_iterate!(model, smld_work, dataset_mode, maxn, max_iterations,
                                          convergence_tol, verbose,
                                          info.iterations, copy(info.history))

    # Apply corrections
    smld_corrected = correctdrift(smld_work, model)

    # Compute final entropy
    final_entropy = _compute_entropy(smld_corrected, maxn)

    elapsed_s = (time_ns() - t_start) / 1e9

    new_info = DriftInfo(
        model,
        elapsed_s,
        :cpu,
        iter_result.iterations,
        iter_result.converged,
        final_entropy,
        iter_result.history,
        info.roi_indices  # Preserve ROI from original call
    )

    return (smld_corrected, new_info)
end

# ============================================================================
# Internal implementation functions
# ============================================================================

"""
FFT quality tier - fast cross-correlation only, no entropy optimization.

Two-pass approach:
1. First pass: align each dataset to dataset 1 (rough alignment)
2. Second pass: align each dataset to all others (refinement)
"""
function _driftcorrect_fft!(model::LegendrePolynomial, smld::SMLD,
                            dataset_mode::Symbol, verbose::Int)
    if verbose > 0
        @info("SMLMDriftCorrection: FFT mode - two-pass cross-correlation alignment")
    end

    n_dims = nDims(smld)
    n_datasets = smld.n_datasets

    if n_datasets < 2
        return (iterations=0, converged=true, history=Float64[])
    end

    # Pass 1: Align each dataset to dataset 1 (rough alignment)
    # findshift(A, B) returns the shift of B relative to A
    # To correct B back to A's frame, correctdrift does: x - inter.dm
    # So inter.dm should equal the shift (positive)
    smld_ref = filter_by_dataset(smld, 1)
    for nn = 2:n_datasets
        smld_n = filter_by_dataset(smld, nn)
        cc_shift = findshift(smld_ref, smld_n; histbinsize=0.05)
        for dim in 1:n_dims
            model.inter[nn].dm[dim] = cc_shift[dim]
        end
    end

    if verbose > 0
        @info("SMLMDriftCorrection: FFT pass 1 complete (each vs DS1)")
    end

    # Pass 2: Refine by aligning each dataset to all others
    for nn = 2:n_datasets
        smld_n = filter_by_dataset(smld, nn)

        # Build merged reference from all other datasets (shifted)
        others = setdiff(1:n_datasets, nn)
        ref_emitters = eltype(smld.emitters)[]

        for other_ds in others
            smld_other = filter_by_dataset(smld, other_ds)
            for e in smld_other.emitters
                e_shifted = deepcopy(e)
                e_shifted.x -= model.inter[other_ds].dm[1]
                e_shifted.y -= model.inter[other_ds].dm[2]
                if n_dims == 3
                    e_shifted.z -= model.inter[other_ds].dm[3]
                end
                push!(ref_emitters, e_shifted)
            end
        end

        # Create reference SMLD from merged emitters
        smld_merged = typeof(smld)(
            ref_emitters,
            smld.camera,
            smld.n_frames,
            1,  # merged into single "dataset"
            copy(smld.metadata)
        )

        cc_shift = findshift(smld_merged, smld_n; histbinsize=0.05)
        for dim in 1:n_dims
            model.inter[nn].dm[dim] = cc_shift[dim]
        end
    end

    if verbose > 0
        @info("SMLMDriftCorrection: FFT pass 2 complete (each vs all others)")
    end

    # Pass 3: Detect and fix outliers using Gaussian-damped re-alignment
    shifts = [sqrt(sum(model.inter[nn].dm.^2)) for nn in 2:n_datasets]
    if length(shifts) >= 3
        median_shift = median(shifts)
        mad_shift = median(abs.(shifts .- median_shift))  # median absolute deviation
        threshold = median_shift + 5 * max(mad_shift, 0.1)  # at least 100nm MAD

        outliers = Int[]
        for nn = 2:n_datasets
            if sqrt(sum(model.inter[nn].dm.^2)) > threshold
                push!(outliers, nn)
            end
        end

        if !isempty(outliers)
            if verbose > 0
                @info("SMLMDriftCorrection: FFT detected $(length(outliers)) outliers, re-aligning with Gaussian prior")
            end

            # Compute median shift vector from non-outliers
            good_datasets = setdiff(2:n_datasets, outliers)
            if !isempty(good_datasets)
                median_dm = [median([model.inter[nn].dm[d] for nn in good_datasets]) for d in 1:n_dims]
                prior_sigma = max(median_shift, 0.5)  # at least 500nm sigma

                for nn in outliers
                    smld_n = filter_by_dataset(smld, nn)
                    cc_shift = findshift_damped(smld_ref, smld_n;
                        histbinsize=0.05,
                        prior_shift=median_dm,
                        prior_sigma=prior_sigma)
                    for dim in 1:n_dims
                        model.inter[nn].dm[dim] = cc_shift[dim]
                    end
                    if verbose > 0
                        new_mag = sqrt(sum(model.inter[nn].dm.^2)) * 1000
                        @info("SMLMDriftCorrection: DS$nn re-aligned: $(round(new_mag, digits=1)) nm")
                    end
                end
            end
        end
    end

    # Normalize for continuous mode (drift at DS1, frame 1 = 0)
    if dataset_mode == :continuous
        # Nothing special needed for FFT mode (no intra correction)
    end

    return (iterations=0, converged=true, history=Float64[])
end

"""
Singlepass quality tier - entropy-based intra and inter correction.
Matches original algorithm: intra first, then inter vs DS1, then inter vs earlier.
"""
function _driftcorrect_singlepass!(model::LegendrePolynomial, smld::SMLD,
                                    dataset_mode::Symbol, maxn::Int, verbose::Int)
    if verbose > 0
        @info("SMLMDriftCorrection: singlepass mode")
    end

    # Step 1: Intra-dataset correction (same path for both modes)
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

    # Step 2: Inter-dataset alignment
    # Both modes use entropy optimization - datasets image the same FOV/structures
    reg_lambda = 0.0
    warmstart_values = Vector{Vector{Float64}}()
    if dataset_mode == :continuous
        # Continuous mode: initialize inter-shifts from polynomial endpoints, then optimize
        _warmstart_inter_continuous!(model, smld, verbose)
        # Estimate endpoint uncertainty and regularize around warmstart
        reg_lambda = _estimate_continuous_lambda(model, smld.n_frames, verbose)
        warmstart_values = [copy(model.inter[nn].dm) for nn in 1:smld.n_datasets]
    end

    if verbose > 0
        @info("SMLMDriftCorrection: inter-dataset alignment (vs DS1)")
    end
    for nn = 2:smld.n_datasets
        findinter!(model, smld, nn, [1], maxn;
            regularization_target = isempty(warmstart_values) ? nothing : warmstart_values[nn],
            regularization_lambda = reg_lambda)
    end

    if verbose > 0
        @info("SMLMDriftCorrection: refining inter-dataset alignment (vs earlier)")
    end
    for nn = 2:smld.n_datasets
        ref_datasets = collect(1:(nn-1))
        findinter!(model, smld, nn, ref_datasets, maxn;
            regularization_target = isempty(warmstart_values) ? nothing : warmstart_values[nn],
            regularization_lambda = reg_lambda)
    end

    # Normalize continuous mode so drift at (DS=1, frame=1) = 0
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
    result = _driftcorrect_iterate!(model, smld, dataset_mode, maxn, max_iterations - 1,
                                     convergence_tol, verbose, 1, history)

    return result
end

"""
Core iterative refinement loop - used by both :iterative and continuation.
"""
function _driftcorrect_iterate!(model::LegendrePolynomial, smld::SMLD,
                                 dataset_mode::Symbol, maxn::Int, max_iterations::Int,
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

        # Update inter-shifts
        # Both modes use entropy optimization - datasets image the same FOV/structures
        reg_lambda = 0.0
        warmstart_values = Vector{Vector{Float64}}()
        if dataset_mode == :continuous
            # Continuous mode: initialize from polynomial endpoints before optimization
            _warmstart_inter_continuous!(model, smld, verbose > 1 ? verbose : 0)
            reg_lambda = _estimate_continuous_lambda(model, smld.n_frames, verbose > 1 ? verbose : 0)
            warmstart_values = [copy(model.inter[nn].dm) for nn in 1:n_datasets]
        end

        # Entropy-based all-to-all alignment
        for nn = 2:n_datasets
            others = collect(setdiff(1:n_datasets, nn))
            findinter!(model, smld, nn, others, maxn;
                regularization_target = isempty(warmstart_values) ? nothing : warmstart_values[nn],
                regularization_lambda = reg_lambda)
        end

        # Normalize continuous mode
        if dataset_mode == :continuous
            _normalize_continuous!(model)
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
Estimate endpoint uncertainty from boundary gaps between adjacent chunks.

After intra-dataset fits, the discontinuity at each boundary measures
how uncertain the polynomial endpoints are. Each gap has contributions
from both sides, so σ_endpoint ≈ mean(|gap|) / sqrt(2).

Returns regularization lambda = 1 / σ² for use with findinter!.
"""
function _estimate_continuous_lambda(model::LegendrePolynomial, n_frames::Int, verbose::Int)
    ndims = model.intra[1].ndims
    n_datasets = model.ndatasets

    if n_datasets < 2
        return 0.0
    end

    # Collect boundary gap magnitudes across all dimensions and boundaries
    gaps = Float64[]
    for nn = 2:n_datasets
        ep = endpoint_drift(model.intra[nn-1], n_frames)
        sp = startpoint_drift(model.intra[nn])
        for dim in 1:ndims
            push!(gaps, abs(ep[dim] - sp[dim]))
        end
    end

    σ_endpoint = mean(gaps) / sqrt(2)

    # Floor to prevent extreme lambda from tiny gaps
    σ_endpoint = max(σ_endpoint, 0.001)  # at least 1nm

    λ = 1.0 / σ_endpoint^2

    if verbose > 0
        @info("SMLMDriftCorrection: continuous regularization σ_endpoint=$(round(σ_endpoint*1000, digits=1))nm, λ=$(round(λ, digits=1))")
    end

    return λ
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
