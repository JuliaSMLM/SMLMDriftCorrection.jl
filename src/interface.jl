# Main interface for drift correction with progressive quality tiers

"""
    driftcorrect(smld; kwargs...) -> (corrected_smld, info)
    driftcorrect(smld, config::DriftConfig) -> (corrected_smld, info)

Main interface for drift correction. Uses Legendre polynomial model with
entropy-based cost function and adaptive KDTree neighbor building.

# Arguments
- `smld`: SMLD structure containing (X, Y) or (X, Y, Z) localization coordinates (μm)
- `config`: Optional `DriftConfig` struct (alternative to keyword arguments)

# Keyword Arguments
- `quality=:singlepass`: Quality tier (`:fft`, `:singlepass`, `:iterative`)
- `degree=2`: Polynomial degree for intra-dataset drift model
- `dataset_mode=:registered`: selects the inter-dataset algorithm (`quality=:fft` ignores it):
    - `:registered`: datasets are independent acquisitions (CC-primary inter alignment)
    - `:continuous`: one acquisition split into files or chunks (consecutive-chunk CC)
- `chunk_frames=0`: For continuous mode, split each dataset into chunks of this many frames
- `n_chunks=0`: Alternative to chunk_frames - specify number of chunks per dataset
- `maxn=100`: Maximum number of neighbors for entropy calculation
- `max_iterations=10`: Iterations per attempt for `:iterative` mode; a run that has not converged is retried from its current model up to 3 times
- `convergence_tol=0.001`: Convergence tolerance (μm) for `:iterative` mode
- `warm_start=nothing`: Previous model for warm starting optimization
- `verbose=0`: Verbosity level (0=quiet, 1=info, 2=debug)
- `auto_roi=false`: Set to `true` for faster (but slightly less accurate) estimation using a dense ROI subset
- `σ_loc=0.010`: Typical localization precision (μm) for ROI sizing
- `σ_target=0.001`: Target drift precision (μm) for ROI sizing
- `roi_safety_factor=4.0`: Safety multiplier for required localizations
- `shift_scale=1.0`: Has no effect: no dataset mode reads it (registered mode stopped using
  it as an L2 inter-shift prior in 0.2.7). Kept so existing calls and configs still
  construct; a non-default value logs a warning.

# Quality Tiers
- `:fft`: Fast cross-correlation only (~10x faster, less accurate)
- `:singlepass`: One pass of intra then inter correction (default). Intra is threaded per dataset; registered inter is threaded per dataset (continuous CC seeding is sequential, its refine threaded then sequential)
- `:iterative`: Iterates intra↔inter until both inter and intra parameter changes fall below `convergence_tol`, or the entropy cost plateaus (relative improvement < 1e-4, both dataset modes)

# Returns
Tuple `(corrected_smld, info)` where `info::DriftInfo` contains:
- `model`: Fitted drift model (LegendrePolynomial)
- `elapsed_s`: Wall time in seconds
- `backend`: Computation backend (`:cpu`)
- `iterations`: Number of iterations completed
- `converged`: Whether convergence was achieved
- `entropy`: Final entropy value; scales with the number of localizations. Divide by `length(smld.emitters)` for a size-independent number
- `history`: Entropy after the singlepass start and after each iteration (`:iterative` only; empty for `:fft` and `:singlepass`)
- `residual_correlation`: Position-frame correlation diagnostic on corrected data; see [`position_frame_correlation`](@ref)
- `dataset_support`: Per dataset, whether it co-localizes with the other datasets more than chance after
  alignment (see [`DriftInfo`](@ref)); one warning names the datasets whose alignment rests on no evidence

# Errors
A cross-correlation that fails in the FFT logs an `@error` line and stops the call
by rethrowing its original error. No quality tier continues without it. An input
the cross-correlation refuses throws its own error without an `@error` line.
On threaded paths, the original error arrives inside a `TaskFailedException` or
`CompositeException`.

# Example
```julia
# Basic usage
(smld_corrected, info) = driftcorrect(smld)

# Using DriftConfig
config = DriftConfig(; quality=:iterative, degree=3, verbose=1)
(smld_corrected, info) = driftcorrect(smld, config)

# Fast FFT-only mode
(smld_corrected, info) = driftcorrect(smld; quality=:fft)

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
    maxn::Int = 100,
    max_iterations::Int = 10,
    convergence_tol::Float64 = 0.001,
    warm_start::Union{Nothing, AbstractIntraInter} = nothing,
    verbose::Int = 0,
    auto_roi::Bool = false,
    σ_loc::Float64 = 0.010,
    σ_target::Float64 = 0.001,
    roi_safety_factor::Float64 = 4.0,
    shift_scale::Float64 = 1.0)

    config = DriftConfig(; quality, degree, dataset_mode, chunk_frames, n_chunks,
        maxn, max_iterations, convergence_tol, warm_start, verbose,
        auto_roi, σ_loc, σ_target, roi_safety_factor, shift_scale)
    return driftcorrect(smld, config)
end

function driftcorrect(smld::SMLD, config::DriftConfig)
    (; quality, degree, dataset_mode, chunk_frames, n_chunks,
       maxn, max_iterations, convergence_tol, warm_start, verbose,
       auto_roi, σ_loc, σ_target, roi_safety_factor, shift_scale) = config

    t_start = time_ns()

    # Validate quality tier
    if quality ∉ (:fft, :singlepass, :iterative)
        error("Unknown quality: $quality. Use :fft, :singlepass, or :iterative")
    end

    # Validate dataset_mode
    if dataset_mode ∉ (:registered, :continuous)
        error("Unknown dataset_mode: $dataset_mode. Use :registered or :continuous")
    end

    # shift_scale is accepted for compatibility but read by no mode (see docstring)
    if shift_scale != 1.0
        @warn("SMLMDriftCorrection: shift_scale=$shift_scale has no effect; no dataset mode " *
              "reads it (registered mode stopped using it as an inter-shift prior in 0.2.7)",
              maxlog=1)
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
            _warn_datasets_outside_roi(smld_work, smld_estimation)
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

    # Preserve warm-started intra coefficients by skipping random init in findintra!
    skip_init = warm_start !== nothing

    # Dispatch to appropriate quality tier (using ROI-subsampled data for estimation)
    if quality == :fft
        result = _driftcorrect_fft!(driftmodel, smld_estimation, dataset_mode, verbose)
    elseif quality == :singlepass
        result = _driftcorrect_singlepass!(driftmodel, smld_estimation, dataset_mode, maxn, verbose, shift_scale; skip_init=skip_init)
    else  # :iterative
        result = _driftcorrect_iterative!(driftmodel, smld_estimation, dataset_mode, maxn,
                                          max_iterations, convergence_tol, verbose, shift_scale; skip_init=skip_init)
    end

    # (Registered inter alignment is now CC-primary inside the tier — cross-correlation
    # seeds + entropy refines + overlap arbiter — so the old post-hoc CC-rescue is no
    # longer needed; :fft keeps its own pass-3 outlier re-alignment.)

    # Apply corrections to get final SMLD
    smld_corrected = _apply_final_corrections(smld, smld_work, driftmodel, chunk_info, dataset_mode)

    # Compute final entropy
    final_entropy = _compute_entropy(smld_corrected, maxn)

    # Compute residual drift diagnostic (intra + inter)
    _K = min(20, max(1, length(smld_corrected.emitters) ÷ max(1, smld_corrected.n_datasets) - 1))
    _diag_intra = position_frame_correlation(smld_corrected; K=_K, mode=:intra)
    _diag_inter = position_frame_correlation(smld_corrected; K=_K, mode=:inter)
    residual_corr = (
        intra_summary = _diag_intra.summary,
        intra_per_dataset = [(dataset=e.dataset, n_locs=e.n_locs, corr_x=e.corr_x, corr_y=e.corr_y, corr_z=e.corr_z) for e in _diag_intra.per_dataset],
        inter = (corr_x=_diag_inter.corr_x, corr_y=_diag_inter.corr_y, corr_z=_diag_inter.corr_z),
    )

    # Alignment support per dataset (per chunk when chunked); name those whose alignment rests on no evidence
    support = _support_and_warn(smld_corrected, smld_work, chunk_info !== nothing && chunk_info.n_chunks > 1)

    elapsed_s = (time_ns() - t_start) / 1e9

    info = DriftInfo(;
        model = driftmodel,
        elapsed_s,
        backend = :cpu,
        iterations = result.iterations,
        converged = result.converged,
        entropy = final_entropy,
        history = result.history,
        roi_indices,
        residual_correlation = residual_corr,
        dataset_mode,
        chunk_frames = chunk_info === nothing ? 0 : chunk_frames,
        n_chunks = chunk_info === nothing ? 0 : n_chunks,
        dataset_support = support
    )

    return (smld_corrected, info)
end

"""
    driftcorrect(smld, info::DriftInfo; kwargs...) -> (corrected_smld, info)

Continue drift correction from a previous result using the model from `info`.
Pass the original, uncorrected `smld`. Continuation reuses the original run's
`dataset_mode` and chunking, including for chunked continuous results.
Corrections preserve the input's dataset and frame numbering.

# Keyword Arguments
- `dataset_mode=info.dataset_mode`: Dataset mode (`:registered` or `:continuous`)
- `chunk_frames=info.chunk_frames`: Frames per chunk for continuous mode
- `n_chunks=info.n_chunks`: Chunks per dataset for continuous mode
- `max_iterations=10`: Additional iterations to run
- `convergence_tol=0.001`: Convergence tolerance (μm)
- `maxn=100`: Maximum neighbors for entropy calculation
- `verbose=0`: Verbosity level

# Errors
A cross-correlation that fails in the FFT logs an `@error` line and stops the call
by rethrowing its original error. An input the cross-correlation refuses throws
its own error without an `@error` line. Continuation does not proceed without CC.
"""
function driftcorrect(smld::SMLD, info::DriftInfo;
    dataset_mode::Symbol = info.dataset_mode,
    chunk_frames::Int = info.chunk_frames,
    n_chunks::Int = info.n_chunks,
    max_iterations::Int = 10,
    convergence_tol::Float64 = 0.001,
    maxn::Int = 100,
    verbose::Int = 0)

    t_start = time_ns()

    if dataset_mode ∉ (:registered, :continuous)
        throw(ArgumentError("Unknown dataset_mode: $dataset_mode. Use :registered or :continuous"))
    end
    if dataset_mode != :continuous && (chunk_frames > 0 || n_chunks > 0)
        throw(ArgumentError("Cannot continue a chunked model in $dataset_mode mode; chunking requires :continuous"))
    end

    # Deep copy to avoid modifying original
    smld_work = deepcopy(smld)
    model = deepcopy(info.model)

    chunk_info = nothing
    if dataset_mode == :continuous && (chunk_frames > 0 || n_chunks > 0)
        chunk_info = chunk_smld(smld_work; chunk_frames=chunk_frames, n_chunks=n_chunks)
        smld_work = chunk_info.smld
    end

    if !(model isa LegendrePolynomial)
        throw(ArgumentError("Cannot continue a $(typeof(model)) model; continuation requires LegendrePolynomial"))
    end
    if model.ndatasets != smld_work.n_datasets
        throw(ArgumentError("Cannot continue: input has $(smld_work.n_datasets) datasets after chunking, " *
                            "but the previous model has $(model.ndatasets); use the original dataset structure and chunking settings"))
    end
    if model.n_frames != smld_work.n_frames
        throw(ArgumentError("Cannot continue: input has $(smld_work.n_frames) frames per dataset after chunking, " *
                            "but the previous model uses $(model.n_frames); use the original frame structure and chunking settings"))
    end
    if model.intra[1].ndims != nDims(smld_work)
        throw(ArgumentError("Cannot continue: input and previous model have different spatial dimensions"))
    end

    # Continue with iterative refinement
    iter_result = _driftcorrect_iterate!(model, smld_work, dataset_mode, maxn, max_iterations,
                                          convergence_tol, verbose,
                                          info.iterations, copy(info.history);
                                          warned=fill(false, smld_work.n_datasets))

    # Apply corrections
    smld_corrected = _apply_final_corrections(smld, smld_work, model, chunk_info, dataset_mode)

    # Compute final entropy
    final_entropy = _compute_entropy(smld_corrected, maxn)

    # Compute residual drift diagnostic (intra + inter)
    _K = min(20, max(1, length(smld_corrected.emitters) ÷ max(1, smld_corrected.n_datasets) - 1))
    _diag_intra = position_frame_correlation(smld_corrected; K=_K, mode=:intra)
    _diag_inter = position_frame_correlation(smld_corrected; K=_K, mode=:inter)
    residual_corr = (
        intra_summary = _diag_intra.summary,
        intra_per_dataset = [(dataset=e.dataset, n_locs=e.n_locs, corr_x=e.corr_x, corr_y=e.corr_y, corr_z=e.corr_z) for e in _diag_intra.per_dataset],
        inter = (corr_x=_diag_inter.corr_x, corr_y=_diag_inter.corr_y, corr_z=_diag_inter.corr_z),
    )

    # Alignment support per dataset (per chunk when chunked); name those whose alignment rests on no evidence
    support = _support_and_warn(smld_corrected, smld_work, chunk_info !== nothing && chunk_info.n_chunks > 1)

    elapsed_s = (time_ns() - t_start) / 1e9

    new_info = DriftInfo(;
        model,
        elapsed_s,
        backend = :cpu,
        iterations = iter_result.iterations,
        converged = iter_result.converged,
        entropy = final_entropy,
        history = iter_result.history,
        roi_indices = info.roi_indices,  # Preserve ROI from original call
        residual_correlation = residual_corr,
        dataset_mode,
        chunk_frames,
        n_chunks,
        dataset_support = support
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
Each CC is computed on bin-count-capped histograms (`maxcount=_CC_MAXCOUNT`) to limit
how much any heavily populated bin counts, so a single bright spot no longer captures
the peak (an aggregate spread over many bins still can), and each CC shift (and the
pass-3 outlier re-alignment) replaces the incumbent unless the incumbent overlaps the
reference strictly better
(`_accept_if_better_overlap!`; ties go to the CC). As in registered mode, a CC with a
component over 5 μm (`_CC_MAX_SHIFT`) is not applied; where it overlaps clearly better than
the shift the dataset ends with, one warning per dataset per call names it as a likely
collection failure (the stage moved between acquisitions; `_warn_dropped_cc!`).
`dataset_mode` is not read by this tier, so the same warning rule applies with `:continuous`.
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

    # last CC shift dropped as absurd (> 5 μm) per dataset; checked against the final shift below
    dropped = Vector{Union{Nothing,Vector{Float64}}}(nothing, n_datasets)

    # Every CC shift below is overlap-guarded (`_accept_if_better_overlap!`): the
    # unnormalized histogram CC argmax can be captured by a hot spot present in some
    # datasets only and land microns away, so a CC shift replaces the incumbent only
    # where it puts more sites of the dataset near the reference (see `_cc_primary_inter!`).

    # Pass 1: Align each dataset to dataset 1 (rough alignment)
    # findshift(A, B) returns the shift of B relative to A
    # To correct B back to A's frame, correctdrift does: x - inter.dm
    # So inter.dm should equal the shift (positive)
    smld_ref = filter_by_dataset(smld, 1)
    tree_ref = _coord_tree(smld_ref, n_dims)
    for nn = 2:n_datasets
        smld_n = filter_by_dataset(smld, nn)
        (isempty(smld_ref.emitters) || isempty(smld_n.emitters)) && continue
        cc_shift = findshift(smld_ref, smld_n; histbinsize=_CC_HISTBIN, maxcount=_CC_MAXCOUNT)
        if _cc_too_far(cc_shift)
            dropped[nn] = Float64.(cc_shift[1:n_dims])
        else
            _accept_if_better_overlap!(model.inter[nn].dm, cc_shift, smld_n, tree_ref, n_dims)
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

        (isempty(ref_emitters) || isempty(smld_n.emitters)) && continue

        # Create reference SMLD from merged emitters
        smld_merged = typeof(smld)(
            ref_emitters,
            smld.camera,
            smld.n_frames,
            1,  # merged into single "dataset"
            copy(smld.metadata)
        )

        cc_shift = findshift(smld_merged, smld_n; histbinsize=_CC_HISTBIN, maxcount=_CC_MAXCOUNT)
        if _cc_too_far(cc_shift)
            dropped[nn] = Float64.(cc_shift[1:n_dims])
        else
            _accept_if_better_overlap!(model.inter[nn].dm, cc_shift, smld_n,
                                       _coord_tree(smld_merged, n_dims), n_dims)
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
                    isempty(smld_ref.emitters) && continue
                    smld_n = filter_by_dataset(smld, nn)
                    cc_shift = findshift_damped(smld_ref, smld_n;
                        histbinsize=_CC_HISTBIN, maxcount=_CC_MAXCOUNT,
                        prior_shift=median_dm,
                        prior_sigma=prior_sigma)
                    if _cc_too_far(cc_shift)
                        dropped[nn] = Float64.(cc_shift[1:n_dims])
                    else
                        _accept_if_better_overlap!(model.inter[nn].dm, cc_shift, smld_n, tree_ref, n_dims)
                    end
                    if verbose > 0
                        new_mag = sqrt(sum(model.inter[nn].dm.^2)) * 1000
                        @info("SMLMDriftCorrection: DS$nn re-aligned: $(round(new_mag, digits=1)) nm")
                    end
                end
            end
        end
    end

    _warn_dropped_cc!(model, smld, dropped, fill(false, n_datasets))

    return (iterations=0, converged=true, history=Float64[])
end

"""
Singlepass quality tier - entropy-based intra and inter correction.
Matches original algorithm: intra first, then inter vs DS1, then inter vs earlier.
"""
function _driftcorrect_singlepass!(model::LegendrePolynomial, smld::SMLD,
                                    dataset_mode::Symbol, maxn::Int, verbose::Int,
                                    shift_scale::Float64=1.0;
                                    skip_init::Bool=false,
                                    warned::Union{Nothing,Vector{Bool}}=nothing)
    if verbose > 0
        @info("SMLMDriftCorrection: singlepass mode")
    end

    # Step 1: Intra-dataset correction (same path for both modes)
    if model.intra[1].dm[1].degree > 0
        if verbose > 0
            @info("SMLMDriftCorrection: starting intra-dataset correction" *
                  (skip_init ? " (warm-started, skipping random init)" : ""))
        end
        Threads.@threads for nn = 1:smld.n_datasets
            findintra!(model.intra[nn], smld, nn, maxn; skip_init=skip_init)
        end
    else
        if verbose > 0
            @info("SMLMDriftCorrection: degree=0, skipping intra-dataset correction")
        end
    end

    # Step 2: Inter-dataset alignment
    # Step 2: Inter-dataset alignment — mode-specific.
    if dataset_mode == :registered
        # CC-primary: cross-correlation seeds each dataset (globally robust), entropy
        # refines (locally precise), and the overlap arbiter keeps the best of
        # {refined, CC, incumbent}. Subsumes
        # the old entropy-first pass1/pass2 + the post-hoc CC-rescue. Continuous keeps
        # endpoint-chaining below — CC-primary isn't built for temporal chunks whose
        # global consensus is smeared by across-chunk drift.
        if verbose > 0
            @info("SMLMDriftCorrection: CC-primary inter-dataset alignment")
        end
        _cc_primary_inter!(model, smld, maxn; n_passes=2, init=true, verbose=verbose,
                           warned=warned)
    else  # :continuous
        # CC-seeded inter shifts: measure each chunk boundary by cross-correlation
        # (endpoint-chaining kept only as a per-boundary fallback), then entropy-refine
        # and normalize so drift at (DS1, frame1)=0. See _ccseed_inter_continuous!.
        _ccseed_inter_continuous!(model, smld; verbose=verbose)
        reg_lambda = _estimate_continuous_lambda(model, smld.n_frames, verbose)
        warmstart_values = [copy(model.inter[nn].dm) for nn in 1:smld.n_datasets]
        precomputed = correctdrift(smld, model)
        # The seeds are chosen (cc_init=false: no fresh, unguarded CC inside the refine), and a
        # refine may not carry a chunk far from its seed without evidence.
        Threads.@threads for nn = 2:smld.n_datasets
            findinter!(model, smld, nn, [1], maxn;
                precomputed_corrected = precomputed,
                regularization_target = warmstart_values[nn], regularization_lambda = reg_lambda,
                cc_init = false)
        end
        for nn = 2:smld.n_datasets
            findinter!(model, smld, nn, collect(1:(nn-1)), maxn;
                regularization_target = warmstart_values[nn], regularization_lambda = reg_lambda,
                cc_init = false)
        end
        _hold_far_refines!(model, smld, warmstart_values)
        _normalize_continuous!(model)
    end

    return (iterations=1, converged=true, history=Float64[])
end

# Relative entropy-improvement threshold for declaring iterative convergence.
# The global merged-cloud entropy is nearly insensitive to a single dataset's inter
# shift (one dataset is ~1/N of the cloud, so a tens-of-nm shift moves the cost
# <1e-6). The inter parameters therefore sit in a near-flat cost basin: each
# iteration re-nudges them > convergence_tol while the entropy is unchanged, so a
# parameter-movement test never converges. Stop when the cost itself plateaus.
const _ENTROPY_REL_TOL = 1.0e-4

# Neighbor radius (μm) for the CC-primary overlap arbiter: per dataset, whichever of
# {entropy-refined, CC, incumbent} has the largest occupancy among the dataset's
# localizations within this distance of a consensus localization is kept. This makes
# each inter step never lower a dataset's consensus overlap even though the
# merged-cloud entropy is only ~1/N sensitive to one dataset's offset and so cannot
# reveal a wrong CC peak or a bad refine on its own.
const _CC_VERIFY_RADIUS = 0.030

# Histogram bin-count cap for the registered CC-primary, continuous chunk seed,
# and :fft cross-correlations
# (`findshift(...; maxcount)`): the CC then measures shared occupancy, so a bright
# aggregate localized thousands of times in one bin cannot capture the peak.
const _CC_MAXCOUNT = 4

# No tier applies a CC shift with a component over this (μm). Registered tiers and :fft
# warn only if a dropped CC overlaps clearly better than the kept shift (a likely
# collection failure); continuous mode uses endpoint chaining at that boundary without warning.
const _CC_MAX_SHIFT = 5.0
_cc_too_far(cc) = maximum(abs.(cc)) > _CC_MAX_SHIFT

"""
Iterative quality tier - full intra↔inter convergence loop.
"""
function _driftcorrect_iterative!(model::LegendrePolynomial, smld::SMLD,
                                   dataset_mode::Symbol, maxn::Int,
                                   max_iterations::Int, convergence_tol::Float64,
                                   verbose::Int, shift_scale::Float64=1.0;
                                   skip_init::Bool=false)
    if verbose > 0
        @info("SMLMDriftCorrection: iterative mode (max_iterations=$max_iterations, tol=$convergence_tol)")
    end

    # First run singlepass to initialize (preserves warm start when skip_init=true)
    # one stage-move warning per dataset across the singlepass init, iterations and retries
    warned = fill(false, smld.n_datasets)
    _driftcorrect_singlepass!(model, smld, dataset_mode, maxn, verbose, shift_scale;
                              skip_init=skip_init, warned=warned)

    # Compute initial entropy
    smld_corrected = correctdrift(smld, model)
    initial_entropy = _compute_entropy(smld_corrected, maxn)
    history = Float64[initial_entropy]

    if verbose > 0
        @info("SMLMDriftCorrection: initial entropy = $initial_entropy")
    end

    # Continue with iterative refinement
    result = _driftcorrect_iterate!(model, smld, dataset_mode, maxn, max_iterations - 1,
                                     convergence_tol, verbose, 1, history, shift_scale; warned=warned)

    # Automatic warm-start retry if not converged: the iterative loop can get
    # stuck in a local minimum where intra and inter fight each other. Restarting
    # from the current model often allows the loop to converge to a much better
    # optimum.
    max_retries = 3
    retry = 0
    while !result.converged && retry < max_retries
        retry += 1
        if verbose > 0
            @info("SMLMDriftCorrection: not converged after $(result.iterations) iterations — auto warm-start retry $retry/$max_retries")
        end
        result = _driftcorrect_iterate!(model, smld, dataset_mode, maxn, max_iterations,
                                         convergence_tol, verbose,
                                         result.iterations, copy(result.history), shift_scale;
                                         warned=warned)
    end

    return result
end

"""
Core iterative refinement loop - used by both :iterative and continuation.
"""
function _driftcorrect_iterate!(model::LegendrePolynomial, smld::SMLD,
                                 dataset_mode::Symbol, maxn::Int, max_iterations::Int,
                                 convergence_tol::Float64, verbose::Int,
                                 starting_iteration::Int, history::Vector{Float64},
                                 shift_scale::Float64=1.0;
                                 warned::Union{Nothing,Vector{Bool}}=nothing)

    n_datasets = smld.n_datasets
    n_dims = model.intra[1].ndims
    converged = false
    iteration = starting_iteration

    for iter = 1:max_iterations
        iteration += 1

        # Snapshot BOTH inter-shifts and intra drift vectors at [1, mid, end]
        # before the pass. Convergence check below uses the max of their deltas,
        # because the merged-cloud intra can move intra materially and inter-only
        # convergence would declare convergence prematurely.
        inter_old = [copy(model.inter[n].dm) for n in 1:n_datasets]
        intra_old_vecs = [Matrix{Float64}(undef, n_dims, 3) for _ in 1:n_datasets]
        for n in 1:n_datasets
            drift_vecs!(intra_old_vecs[n], model.intra[n], smld.n_frames)
        end

        # Re-run intra with inter applied (shifted coordinates).
        # skip_init=true: model already has coefficients from previous iteration/singlepass;
        # re-randomizing would discard the progress we're trying to refine.
        #
        # Merged-cloud intra: in iteration 2+ the inter model is populated, so every
        # dataset has a fully-corrected snapshot available. Pass the other datasets'
        # corrected coords as `ref_coords` — intra now fits against the same structural
        # scaffold that findinter! uses, which breaks the intra ↔ inter limit cycle
        # that blocks convergence on cells with small inter-shifts.
        #
        # Memory: we pass ONE set of full arrays (no per-dataset mask copies). Each
        # findintra! thread filters inline and allocates only the ref-filtered σ
        # arrays + data_combined it actually uses.
        smld_shifted = apply_inter_only(smld, model)
        smld_full = correctdrift(smld, model)
        x_all   = Float64[e.x   for e in smld_full.emitters]
        y_all   = Float64[e.y   for e in smld_full.emitters]
        σ_x_all = Float64[e.σ_x for e in smld_full.emitters]
        σ_y_all = Float64[e.σ_y for e in smld_full.emitters]
        ds_all  = Int[e.dataset for e in smld_full.emitters]
        z_all   = n_dims == 3 ? Float64[e.z   for e in smld_full.emitters] : Float64[]
        σ_z_all = n_dims == 3 ? Float64[e.σ_z for e in smld_full.emitters] : Float64[]

        # :continuous mode: compute soft endpoint prior for each chunk's intra fit.
        # Formulation is in TOTAL-DRIFT coordinates (matches findinter/warmstart
        # semantics). For chunk n, the total drift at frame f is `inter[n] + intra[n](f)`.
        # Continuity says drift at chunk n's first frame should equal drift at chunk
        # n-1's last frame. During the intra fit for chunk n, inter[n] is held at its
        # current value — so expressing the prior as "target for intra[n](1)" means
        # `target = total_end_{n-1} - inter[n]` where `total_end_{n-1} = inter[n-1] +
        # intra[n-1](N_frames)`. Analogous for the endpoint.
        #
        # λ scale: the θ-dependent entropy cost is `entropy_HD - out/N_n` where
        # entropy_HD is θ-independent and `out/N_n` is per-locus averaged. So the
        # θ-varying part is O(1) per evaluation, not O(N_n). The prior is also O(1)
        # per endpoint, so using `λ = 1/σ²` with σ from _estimate_continuous_lambda
        # (typical boundary-gap uncertainty, few nm) puts a ~1-unit penalty at the
        # σ level and a ~10-100× penalty at 10-100 nm deviation — comparable to
        # per-iteration entropy changes. We do NOT divide λ by N_n: the prior lives
        # at the same scale as the optimization-relevant entropy term.
        #
        # Without this prior, merged-cloud intra on sparse data can drift into
        # polynomials whose endpoints don't match the warmstart chain, producing
        # chunk-boundary blow-ups.
        boundary_priors = nothing
        if dataset_mode == :continuous
            λ_boundary = _estimate_continuous_lambda(model, smld.n_frames, verbose > 1 ? verbose : 0)
            start_targets = Vector{Union{Nothing, Vector{Float64}}}(undef, n_datasets)
            end_targets   = Vector{Union{Nothing, Vector{Float64}}}(undef, n_datasets)
            # The drift across an empty chunk is unknown, so no continuity prior crosses one:
            # an empty neighbor's placeholder model would otherwise pull this chunk's intra.
            has_locs = falses(n_datasets)
            for e in smld.emitters
                1 <= e.dataset <= n_datasets && (has_locs[e.dataset] = true)
            end
            for n in 1:n_datasets
                # Startpoint target: target value for intra[n](1).
                # Chunk 1 anchors origin — total drift at (DS=1, frame=1) = 0 after
                # normalization, so intra[1](1) ≈ -inter[1].
                if n == 1
                    start_targets[1] = [-model.inter[1].dm[d] for d in 1:n_dims]
                elseif !has_locs[n-1]
                    start_targets[n] = nothing
                else
                    # total_end_{n-1} = inter[n-1] + intra[n-1](N);  target for
                    # intra[n](1) is total_end_{n-1} - inter[n].
                    start_targets[n] = [model.inter[n-1].dm[d] +
                                        evaluate_at_frame(model.intra[n-1].dm[d], smld.n_frames) -
                                        model.inter[n].dm[d]
                                        for d in 1:n_dims]
                end
                # Endpoint target: target value for intra[n](N). No constraint for
                # the last chunk (nothing to chain into).
                if n == n_datasets || !has_locs[n+1]
                    end_targets[n] = nothing
                else
                    # total_start_{n+1} = inter[n+1] + intra[n+1](1);  target for
                    # intra[n](N) is total_start_{n+1} - inter[n].
                    end_targets[n] = [model.inter[n+1].dm[d] +
                                      evaluate_at_frame(model.intra[n+1].dm[d], 1) -
                                      model.inter[n].dm[d]
                                      for d in 1:n_dims]
                end
            end
            boundary_priors = (start_targets = start_targets,
                               end_targets = end_targets,
                               λ = λ_boundary)
        end

        Threads.@threads for nn = 1:n_datasets
            ref = if n_dims == 2
                (x_all = x_all, y_all = y_all,
                 σ_x_all = σ_x_all, σ_y_all = σ_y_all,
                 ds_all = ds_all, exclude_dataset = nn)
            else
                (x_all = x_all, y_all = y_all, z_all = z_all,
                 σ_x_all = σ_x_all, σ_y_all = σ_y_all, σ_z_all = σ_z_all,
                 ds_all = ds_all, exclude_dataset = nn)
            end
            prior = if boundary_priors === nothing
                nothing
            else
                (start_target = boundary_priors.start_targets[nn],
                 end_target   = boundary_priors.end_targets[nn],
                 λ            = boundary_priors.λ)
            end
            findintra!(model.intra[nn], smld_shifted, nn, maxn;
                       skip_init=true, ref_coords=ref, boundary_prior=prior)
        end

        # :continuous diagnostic: after the intra pass, report the residual
        # boundary gaps in nm — max across chunks and per-chunk list if verbose≥2.
        # A well-controlled run should show residuals of O(σ_endpoint) ≈ few nm.
        # Large sustained residuals on one boundary indicate the prior is being
        # overruled by the entropy cost.
        if dataset_mode == :continuous && verbose > 0
            max_gap_nm = 0.0
            worst_boundary = 0
            for n in 1:(n_datasets - 1)
                gap2 = 0.0
                for d in 1:n_dims
                    end_n = model.inter[n].dm[d] +
                            evaluate_at_frame(model.intra[n].dm[d], smld.n_frames)
                    start_np1 = model.inter[n+1].dm[d] +
                                evaluate_at_frame(model.intra[n+1].dm[d], 1)
                    gap2 += (start_np1 - end_n)^2
                end
                gap_nm = 1000 * sqrt(gap2)
                if gap_nm > max_gap_nm
                    max_gap_nm = gap_nm
                    worst_boundary = n
                end
                if verbose > 1
                    @info("SMLMDriftCorrection: boundary $(n)→$(n+1) gap = $(round(gap_nm, digits=2)) nm")
                end
            end
            @info("SMLMDriftCorrection: iter $iteration continuous-boundary max gap " *
                  "= $(round(max_gap_nm, digits=2)) nm at DS$(worst_boundary)→DS$(worst_boundary+1)")
        end

        # Update inter-shifts — mode-specific.
        if dataset_mode == :registered
            # CC-primary, one Jacobi pass per iteration (the loop provides the iteration):
            # CC seed + entropy refine + overlap arbiter, reference included + re-anchored.
            _cc_primary_inter!(model, smld, maxn; n_passes=1, init=false,
                               verbose = verbose > 1 ? verbose : 0, warned=warned)
        else  # :continuous
            # Endpoint-chained inter; do NOT re-chain from polynomial endpoints per
            # iteration (b463bd1) — refine continuously from the previous iteration;
            # continuity is enforced softly by the intra boundary prior in findintra!.
            reg_lambda = _estimate_continuous_lambda(model, smld.n_frames, verbose > 1 ? verbose : 0)
            warmstart_values = [copy(model.inter[nn].dm) for nn in 1:n_datasets]
            precomputed = correctdrift(smld, model)
            Threads.@threads for nn = 2:n_datasets
                findinter!(model, smld, nn, collect(setdiff(1:n_datasets, nn)), maxn;
                    precomputed_corrected = precomputed,
                    regularization_target = warmstart_values[nn], regularization_lambda = reg_lambda,
                    cc_init = false)
            end
            _hold_far_refines!(model, smld, warmstart_values)
            _normalize_continuous!(model)
        end

        # Compute entropy for this iteration
        smld_corrected = correctdrift(smld, model)
        current_entropy = _compute_entropy(smld_corrected, maxn)
        push!(history, current_entropy)

        # Entropy-plateau early-exit. The parameter-movement test below can fail to
        # settle in a flat entropy basin, so the loop runs to the iteration cap (and
        # through the warm-start retries, which only re-enter this same loop and so
        # cannot escape a flat basin) at ~0% cost gain. BOTH dataset modes can reach
        # such a basin:
        #   - registered: the global merged-cloud entropy is nearly insensitive to a
        #     single dataset's inter shift (one dataset is ~1/N of the cloud), so the
        #     inter parameters sit in a near-flat cost basin and intra wobbles as its
        #     scaffold follows.
        #   - continuous: the CC-seeded singlepass init already solves the alignment,
        #     so the iterate loop only wobbles params below the entropy scale yet above
        #     convergence_tol (a run can reach the converged run's exact
        #     entropy yet report converged=false at the iteration cap).
        # When the cost itself has stopped improving the optimization is done, so stop
        # on a relative-improvement plateau (both modes). The intra Legendre has no
        # P_0 term, so a flat-entropy param move is degenerate wobble, not a real
        # refinement being cut short.
        if length(history) >= 2
            prev_entropy = history[end-1]
            rel_impr = (prev_entropy - current_entropy) / max(abs(prev_entropy), eps())
            if rel_impr < _ENTROPY_REL_TOL
                converged = true
                if verbose > 0
                    @info("SMLMDriftCorrection: entropy plateau after $iteration iterations " *
                          "(relative improvement $(round(rel_impr, sigdigits=2)) < $_ENTROPY_REL_TOL)")
                end
                break
            end
        end

        # Convergence criterion now considers BOTH inter-shift movement and
        # intra drift-vector movement at test frames [1, mid, end]. Declaring
        # convergence on inter-only lets the merged-cloud intra sneak through
        # with material intra drift still in flight. Log the two components
        # separately so reports can show whether convergence is joint.
        inter_delta = _max_inter_change(inter_old, model.inter)
        intra_delta = 0.0
        scratch = Matrix{Float64}(undef, n_dims, 3)
        for n in 1:n_datasets
            drift_vecs!(scratch, model.intra[n], smld.n_frames)
            d = max_drift_vec_delta(scratch, intra_old_vecs[n])
            if d > intra_delta
                intra_delta = d
            end
        end
        max_change = max(inter_delta, intra_delta)

        if verbose > 0
            @info("SMLMDriftCorrection: iteration $iteration, entropy=$current_entropy, " *
                  "max_shift_change=$max_change (inter=$(round(inter_delta, sigdigits=3)) " *
                  "intra=$(round(intra_delta, sigdigits=3)))")
        end

        if max_change < convergence_tol
            converged = true
            if verbose > 0
                @info("SMLMDriftCorrection: converged after $iteration iterations " *
                      "(inter=$(round(inter_delta, sigdigits=3)), intra=$(round(intra_delta, sigdigits=3)))")
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
Warn when `auto_roi` leaves a dataset that has localizations with none in its region: that
dataset keeps its starting model (zero, a warm start's, or in continuous mode a shift chained
from its neighbor), which no localization of its own has fit. Returns the datasets left out.
"""
function _warn_datasets_outside_roi(full::SMLD, roi::SMLD)
    n = full.n_datasets
    in_full = falses(n); in_roi = falses(n)
    for e in full.emitters
        1 <= e.dataset <= n && (in_full[e.dataset] = true)
    end
    for e in roi.emitters
        1 <= e.dataset <= n && (in_roi[e.dataset] = true)
    end
    lost = findall(in_full .& .!in_roi)
    isempty(lost) || @warn("SMLMDriftCorrection: auto_roi kept no localizations of dataset(s) $(lost) in its " *
                           "region, so their drift is not fit to their own localizations; " *
                           "run with auto_roi=false")
    return lost
end

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
Fraction of a dataset's localizations (after applying `inter`) that fall within
`radius` (μm) of a consensus localization in `tree`. Used by the CC-primary
overlap arbiter as a model-free alignment-quality score.
"""
function _overlap_score(sd::SMLD, inter::Vector{Float64}, tree, radius::Float64, n_dims::Int)
    isempty(sd.emitters) && return 0.0
    cnt = 0
    pt = zeros(Float64, n_dims)
    for e in sd.emitters
        pt[1] = e.x - inter[1]
        pt[2] = e.y - inter[2]
        n_dims == 3 && (pt[3] = e.z - inter[3])
        _, dists = knn(tree, pt, 1)
        dists[1] < radius && (cnt += 1)
    end
    return cnt / length(sd.emitters)
end

"""
KDTree over an SMLD's localization coordinates (Float64), for `_overlap_score`.
"""
function _coord_tree(sd::SMLD, n_dims::Int)
    mat = if n_dims == 2
        permutedims([Float64[e.x for e in sd.emitters] Float64[e.y for e in sd.emitters]])
    else
        permutedims([Float64[e.x for e in sd.emitters] Float64[e.y for e in sd.emitters] Float64[e.z for e in sd.emitters]])
    end
    return KDTree(mat; leafsize=10)
end

"""
Overlap guard for a cross-correlation shift: overwrite `inter` with `cand` unless the
incumbent `inter` has a strictly larger occupancy (`_occupancy` of the localizations within
`_CC_VERIFY_RADIUS` of a reference localization in `tree`), or `cand` lies more than
`_EVIDENCE_RADIUS` from the incumbent without a significant lead (`_far_without_evidence`).
Within `_EVIDENCE_RADIUS` ties go to the CC: with the CC computed on capped histograms, a tie
(typically both zero, when a correct but bin-quantized CC lands just past the radius on sparse
precise data) carries no evidence for the incumbent. Returns whether `cand` was accepted.
"""
function _accept_if_better_overlap!(inter::Vector{Float64}, cand, sd::SMLD, tree, n_dims::Int)
    c = Float64.(cand[1:n_dims])
    in_cand = _overlap_inliers(sd, c, tree, _CC_VERIFY_RADIUS, n_dims)
    in_inc  = _overlap_inliers(sd, inter, tree, _CC_VERIFY_RADIUS, n_dims)
    w = _occupancy_weights(sd, _disagreeing(in_cand, in_inc))
    _occ_gt(_occupancy(w, in_inc), _occupancy(w, in_cand)) && return false   # reject only on evidence: ties go to the CC
    _far_without_evidence(sd, tree, c, inter, n_dims; raw_cc = true) && return false
    inter .= c
    return true
end

"Histogram bin (μm) of the registered and `:fft` inter cross-correlations (`findshift`)."
const _CC_HISTBIN = 0.05

"Histogram bin (μm) of the continuous-mode boundary cross-correlation (`_ccseed_inter_continuous!`)."
const _CC_CONT_HISTBIN = 0.02

"""
Distance (μm) beyond which a new inter shift replaces a dataset's incumbent only with a
significant lead in overlap (`_far_without_evidence`). The datasets of a registered
acquisition are offset from each other by much less than this on real data, while a
cross-correlation or refine with nothing to lock onto (sparse data) lands microns away.
Within this distance of the incumbent the tie rule still applies, so a dataset too sparse to
show a lead can move up to this far per pass, and further over the passes of a run (most in
`:iterative`); `info.dataset_support` names such a dataset. At 30 nm, correct 40-100 nm moves
on simulated sparse data were held at zero.
"""
const _EVIDENCE_RADIUS = 0.200

"""
Whether the localization flags of `sd` at shift `cand` lead those at `inc` significantly, inliers
within `r` of a localization in `tree`: weighted McNemar on effective sites (`_overlap_leads`,
`z = _TIE_Z`, weights `_occupancy_weights` of the localizations whose flags differ).
"""
function _leads_at(sd::SMLD, tree, cand, inc, n_dims::Int, r::Float64)
    ic = _overlap_inliers(sd, Float64.(cand), tree, r, n_dims)
    ii = _overlap_inliers(sd, Float64.(inc), tree, r, n_dims)
    w = _occupancy_weights(sd, _disagreeing(ic, ii))
    return _overlap_leads(ic, ii; z = _TIE_Z, w = w)
end

"""
Evidence rule for a large inter move: whether `cand` lies more than `_EVIDENCE_RADIUS` from the
incumbent `inc` without leading it significantly (`_leads_at`) at `_CC_VERIFY_RADIUS`, or, for a
raw cross-correlation shift (`raw_cc`), at `_CC_VERIFY_RADIUS` plus half a histogram bin
(`binsize`) diagonal, the precision a CC is measured to (a correct CC can sit up to that far from the truth,
just past `_CC_VERIFY_RADIUS`, where it and a far incumbent both cover nothing). A move needs
a lead of at least five effective sites when the incumbent covers none of them (`z > 2`); with
too little data to show one, the dataset keeps its incumbent and `_dataset_support` names it.
"""
function _far_without_evidence(sd::SMLD, tree, cand, inc, n_dims::Int; raw_cc::Bool,
                               binsize::Float64 = _CC_HISTBIN)
    sqrt(sum(abs2, cand .- inc)) > _EVIDENCE_RADIUS || return false
    _leads_at(sd, tree, cand, inc, n_dims, _CC_VERIFY_RADIUS) && return false
    raw_cc || return true
    rq = _CC_VERIFY_RADIUS + sqrt(n_dims) * binsize / 2
    return !_leads_at(sd, tree, cand, inc, n_dims, rq)
end

"""
Return a copy of dataset `d` with ONLY its intra-drift removed (inter shift left in),
so cross-correlation/overlap against the consensus recovers the total inter shift.
"""
function _intra_only_corrected(smld::SMLD, model::LegendrePolynomial, d::Int)
    n_dims = nDims(smld)
    sd = deepcopy(filter_by_dataset(smld, d))
    for e in sd.emitters
        e.x = correctdrift(e.x, e.frame, model.intra[d].dm[1])
        e.y = correctdrift(e.y, e.frame, model.intra[d].dm[2])
        n_dims == 3 && (e.z = correctdrift(e.z, e.frame, model.intra[d].dm[3]))
    end
    return sd
end

"""
    _eligible_candidates(sd, tree, refined, in_ref, cc, in_cc, inc, in_inc, seed, n_dims)

The registered arbiter's candidates after the evidence rule: a CC (judged as a raw CC) or a
refine more than `_EVIDENCE_RADIUS` from the incumbent `inc` without a significant lead
(`_far_without_evidence`) leaves before arbitration, so a veto cannot discard an admissible CC
after the arbiter picked a nearby refine (Codex review of d9577f6). The CC becomes `nothing`; the
refine falls back to its own `seed` (the incumbent or the CC, chosen under the same rule), so the
incumbent does not take the refine's place and its tie preference over an eligible CC (Codex
review of 797ad88: a bin-quantized CC covering nothing at 30 nm lost a zero-occupancy tie to the
incumbent). A refine that stayed on the raw CC keeps the CC's half-bin allowance.
Returns `(refined, in_ref, cc, in_cc)`.
"""
function _eligible_candidates(sd::SMLD, tree, refined, in_ref, cc, in_cc, inc, in_inc, seed, n_dims::Int)
    in_seed = cc !== nothing && seed == cc ? in_cc : in_inc
    if cc !== nothing && _far_without_evidence(sd, tree, cc, inc, n_dims; raw_cc = true)
        cc = nothing; in_cc = nothing
    end
    if _far_without_evidence(sd, tree, refined, inc, n_dims; raw_cc = cc !== nothing && refined == cc)
        refined, in_ref = seed == inc || cc === nothing ? (inc, in_inc) : (seed, in_seed)
    end
    return refined, in_ref, cc, in_cc
end

"""
    _hold_far_refines!(model, smld, seeds)

After a continuous inter refine (`findinter!` regularized toward `seeds`), put back any chunk
the refine carried more than `_EVIDENCE_RADIUS` from its seed without a significant overlap
lead over the seed (`_far_without_evidence`): the rule the boundary seed itself obeyed. Every
such move is judged against the other chunks at their seeds, the positions already accepted, so
no decision rests on another chunk's move that is itself put back (Codex review of 797ad88:
checked in order against the current model, a chunk kept a 1 μm move that only a later chunk's
reverted move supported). Moves within `_EVIDENCE_RADIUS` stay as refined.
"""
function _hold_far_refines!(model::LegendrePolynomial, smld::SMLD, seeds)
    n_dims = nDims(smld)
    n = smld.n_datasets
    moved = [nn for nn in 2:n if sqrt(sum(abs2, model.inter[nn].dm .- seeds[nn])) > _EVIDENCE_RADIUS]
    isempty(moved) && return model
    refined = [copy(model.inter[nn].dm) for nn in 1:n]
    for nn in 2:n
        model.inter[nn].dm .= seeds[nn]
    end
    at_seeds = correctdrift(smld, model)
    hold = falses(n)
    for nn in moved
        ref = filter_by_dataset(at_seeds, [x for x in 1:n if x != nn])
        cur = _intra_only_corrected(smld, model, nn)
        (isempty(ref.emitters) || isempty(cur.emitters)) && continue
        tree = _coord_tree(ref, n_dims)
        hold[nn] = _far_without_evidence(cur, tree, refined[nn], seeds[nn], n_dims; raw_cc = false)
    end
    for nn in 2:n
        model.inter[nn].dm .= hold[nn] ? seeds[nn] : refined[nn]
    end
    return model
end

"""
Warn, once per dataset per call (`warned`), where a CC shift dropped for a component over 5 μm
(`dropped[d]`, else `nothing`) would have overlapped clearly better (`_overlap_leads`) than
the shift the dataset ends with: a likely collection failure (the stage moved between
acquisitions). Used by registered `:singlepass` and `:iterative`, and by `:fft` regardless
of `dataset_mode`; continuous `:singlepass` and `:iterative` instead drop such a boundary
CC and fall back to endpoint chaining at that boundary without a warning. The dropped CC
was measured against an earlier consensus (DS1 for the cold start), so a fresh CC against
the final consensus is scored as well. The applied shift is not changed.
"""
function _warn_dropped_cc!(model::LegendrePolynomial, smld::SMLD,
                           dropped::Vector{Union{Nothing,Vector{Float64}}}, warned::Vector{Bool})
    n_dims = nDims(smld)
    n_datasets = smld.n_datasets
    any(!isnothing, dropped) || return
    snap = correctdrift(smld, model)
    for d in 1:n_datasets
        (dropped[d] === nothing || warned[d]) && continue
        ref = filter_by_dataset(snap, [x for x in 1:n_datasets if x != d])
        isempty(ref.emitters) && continue
        sd = _intra_only_corrected(smld, model, d)
        isempty(sd.emitters) && continue
        tree = _coord_tree(ref, n_dims)
        kept = copy(model.inter[d].dm)
        in_kept = _overlap_inliers(sd, kept, tree, _CC_VERIFY_RADIUS, n_dims)
        cands = [dropped[d]]
        fresh = findshift(ref, sd; histbinsize=_CC_HISTBIN, maxcount=_CC_MAXCOUNT)
        _cc_too_far(fresh) && push!(cands, Float64.(fresh[1:n_dims]))
        for c in cands
            in_c = _overlap_inliers(sd, c, tree, _CC_VERIFY_RADIUS, n_dims)
            _overlap_leads(in_c, in_kept) || continue
            warned[d] = true
            @warn("SMLMDriftCorrection: dataset $d: a cross-correlation shift of " *
                  "$(round.(c, digits=2)) μm (over 5 μm, not applied) puts " *
                  "$(round(100 * mean(in_c), digits=1))% of its localizations within " *
                  "$(round(1000 * _CC_VERIFY_RADIUS, digits=0)) nm of the other datasets, against " *
                  "$(round(100 * mean(in_kept), digits=1))% at the kept shift; the dataset was " *
                  "left at the kept shift (likely a collection failure: the stage moved between " *
                  "acquisitions; check the data)")
            break
        end
    end
    return
end

"""
Damping factor `(N-1)/N` for a registered pass over `N >= 3` datasets with
localizations, 1 for `N = 2`.

In a Jacobi pass every dataset with localizations (the reference included) is aligned
against one snapshot of the others while all of them move. A dataset's refine measures
its offset from the mean of the others, which move too, so applying the full measured step overcounts the relative
correction by `N/(N-1)`; scaling a small step by `(N-1)/N` removes that overcount (exact
in the small-offset regime). With two datasets that have localizations only the second
one moves (see `_cc_primary_inter!`), so nothing is overcounted.
"""
_jacobi_damping(n_datasets::Int) = n_datasets > 2 ? (n_datasets - 1) / n_datasets : 1.0

"""
CC-primary inter-dataset alignment (registered mode).

Cross-correlation is the *primary* per-dataset objective (globally robust — it finds
the right alignment basin even where the merged-cloud entropy is blind to a single
dataset's offset), and entropy is the *refinement* (locally precise). For each
dataset: cross-correlate against the corrected consensus of the others, seed from
the CC shift unless the incumbent shift overlaps strictly better,
entropy-refine from that seed (`findinter!`), then arbitrate {refined, CC, incumbent}
(`_arbitrate`): the largest occupancy (`_occupancy`: inliers, a site localized many times
counted once, by inverse local density weights) picks the basin, ties keeping the refined shift, and among
the candidates within `_CC_VERIFY_RADIUS` of that winner that it does not lead significantly at the
site level, the lowest entropy cost (the refine's objective) is kept; equal costs keep the incumbent. The overlap arbiter is
required because the merged-cloud entropy is only ~1/N sensitive to one dataset, so
neither a wrong CC peak nor a refine that drifted to a wrong local minimum shows up
in the entropy value. A raw-count histogram CC can be captured by a hot spot (a
bright aggregate localized thousands of times in one bin, present in some datasets
only): its peak then lands wherever the dataset's densest cluster lines up with the
hot spot, microns away. The CCs here are therefore computed on bin-count-capped
histograms (`findshift(...; maxcount=_CC_MAXCOUNT)`) to limit how much any heavily
populated bin counts, so a single bright spot no longer captures the peak (an
aggregate spread over many bins still can); the incumbent stays a
candidate as a backstop, so each step never lowers the dataset's consensus overlap
as scored. The refine starts from the chosen seed even when it is zero
(`findinter!(...; cc_init=false)`). A CC with a
component over 5 μm (`_CC_MAX_SHIFT`; `:fft` has the same rule) is not a candidate; if such
a dropped CC overlaps clearly better than the shift a dataset ends with (`_overlap_leads`),
one `@warn` names it as a likely collection failure (the stage moved between acquisitions),
once per dataset per call (`warned`; the iterative tier passes one vector across its
initialization, iterations, and retries). The warning does not change the applied shift.
The reference is included when at least three datasets have localizations, so a
misaligned reference is fixable; with two datasets that have localizations only
the second is aligned, against the first. The result is
re-anchored to inter[1]=0 (an image-invariant global shift). Where the arbiter's step for
a dataset moves it less than `_CC_VERIFY_RADIUS`, the step is damped by `_jacobi_damping`
using the number of datasets with localizations; a larger move (another basin) is
applied in full. `init=true` first does
a rough cross-correlation init against the first dataset with localizations (cold
start), accepted unless the incumbent (zero for registered data) overlaps it strictly
better. Jacobi: each pass uses one corrected snapshot. Subsumes the old post-hoc
CC-rescue.
"""
function _cc_primary_inter!(model::LegendrePolynomial, smld::SMLD, maxn::Int;
                             n_passes::Int=2, init::Bool=false, verbose::Int=0,
                             warned::Union{Nothing,Vector{Bool}}=nothing)
    n_dims = nDims(smld)
    n_datasets = smld.n_datasets
    counts = zeros(Int, n_datasets)
    for e in smld.emitters
        1 <= e.dataset <= n_datasets && (counts[e.dataset] += 1)
    end
    live = findall(>(0), counts)
    length(live) < 2 && return
    warned === nothing && (warned = fill(false, n_datasets))
    # last CC shift dropped as absurd (> 5 μm) per dataset; checked against the final shift below
    dropped = Vector{Union{Nothing,Vector{Float64}}}(nothing, n_datasets)

    # Cold start: align each live dataset to the first live one, kept unless the
    # incumbent overlaps the reference strictly better (ties go to the CC).
    if init
        r = live[1]
        ds1 = _intra_only_corrected(smld, model, r)
        if !isempty(ds1.emitters)
            tree1 = _coord_tree(ds1, n_dims)
            Threads.@threads for d in live[2:end]
                sd = _intra_only_corrected(smld, model, d)
                isempty(sd.emitters) && continue
                cc = findshift(ds1, sd; histbinsize=_CC_HISTBIN, maxcount=_CC_MAXCOUNT)
                if _cc_too_far(cc)
                    dropped[d] = Float64.(cc[1:n_dims])
                    continue
                end
                _accept_if_better_overlap!(model.inter[d].dm, cc, sd, tree1, n_dims)
            end
        end
    end

    # With two live datasets only their relative shift is defined: hold the first.
    aligned = length(live) == 2 ? live[2:2] : live
    damp = _jacobi_damping(length(live))
    for pass in 1:n_passes
        snap = correctdrift(smld, model)             # Jacobi snapshot (read-only across threads)
        Threads.@threads for d in aligned
            others = [x for x in 1:n_datasets if x != d]
            ref = filter_by_dataset(snap, others)
            isempty(ref.emitters) && continue
            sd = _intra_only_corrected(smld, model, d)
            isempty(sd.emitters) && continue

            tree = _coord_tree(ref, n_dims)
            inc = copy(model.inter[d].dm)
            in_inc = _overlap_inliers(sd, inc, tree, _CC_VERIFY_RADIUS, n_dims)

            # CC candidate; a CC with a component over 5 μm is not a candidate.
            cc = findshift(ref, sd; histbinsize=_CC_HISTBIN, maxcount=_CC_MAXCOUNT)
            cc_v = _cc_too_far(cc) ? nothing : Float64.(cc[1:n_dims])
            cc_v === nothing && (dropped[d] = Float64.(cc[1:n_dims]))
            in_cc = cc_v === nothing ? nothing : _overlap_inliers(sd, cc_v, tree, _CC_VERIFY_RADIUS, n_dims)

            # entropy-refine from the CC seed unless the incumbent overlaps strictly better (no L2 reg:
            # the seed is the anchor)
            seed = inc
            if cc_v !== nothing
                w = _occupancy_weights(sd, _disagreeing(in_cc, in_inc))
                seed = (_occ_gt(_occupancy(w, in_inc), _occupancy(w, in_cc)) ||
                        _far_without_evidence(sd, tree, cc_v, inc, n_dims; raw_cc = true)) ? inc : cc_v
            end
            model.inter[d].dm .= seed
            findinter!(model, smld, d, others, maxn;
                       precomputed_corrected = snap, regularization_lambda = 0.0,
                       cc_init = false)
            refined = copy(model.inter[d].dm)
            in_ref = _overlap_inliers(sd, refined, tree, _CC_VERIFY_RADIUS, n_dims)

            # a move beyond _EVIDENCE_RADIUS needs a significant lead (sparse data)
            refined, in_ref, cc_v, in_cc = _eligible_candidates(sd, tree, refined, in_ref, cc_v, in_cc,
                                                                inc, in_inc, seed, n_dims)
            # overlap arbiter over {refined, CC, incumbent}; near ties go to the lowest entropy cost
            w = _occupancy_weights(sd, cc_v === nothing ? _disagreeing(in_ref, in_inc) :
                                       _disagreeing(in_ref, in_cc, in_inc))
            best = _arbitrate(refined, in_ref, cc_v, in_cc, inc, in_inc;
                              score = f -> _occupancy(w, f), w = w,
                              cost = _arbiter_cost(model, smld, snap, d, others, maxn))
            step = best .- inc
            model.inter[d].dm .= sqrt(sum(abs2, step)) < _CC_VERIFY_RADIUS ? inc .+ damp .* step : best
        end
    end

    _warn_dropped_cc!(model, smld, dropped, warned)

    # Re-anchor to inter[1]=0 (global frame shift, image-invariant) if the reference moved.
    if any(abs.(model.inter[1].dm) .> 1e-12)
        δ = copy(model.inter[1].dm)
        for d in 1:n_datasets, k in 1:n_dims
            model.inter[d].dm[k] -= δ[k]
        end
    end
    return
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
Seed continuous-mode inter-shifts by cross-correlating against the last chunk with
localizations (FFT), with a per-boundary endpoint-chaining fallback.
The CC histograms are capped at `_CC_MAXCOUNT` localizations per bin to limit how
much any heavily populated bin counts, so a single bright spot no longer captures
the peak (an aggregate spread over many bins still can).

Continuous chunks share structure (same FOV, consecutive in time), so the inter
shift across each boundary can be *measured* by cross-correlating the intra-
corrected chunk against the already-placed last chunk with localizations, rather than
*extrapolated* from the polynomial endpoints. Per boundary (n-1)→n, both
candidates are formed: the CC-measured shift, and the endpoint-chained shift
`inter[n-1] + endpoint(n-1) - startpoint(n)`. Keep the CC unless its overlap
with the corrected last chunk with localizations is more than 0.05 of chunk n's
localizations below the endpoint candidate's overlap (`_overlap_score` within
`_CC_VERIFY_RADIUS`), or it lies more than `_EVIDENCE_RADIUS` from the endpoint candidate
without leading it significantly (`_far_without_evidence`, the evidence rule of registered
mode; a sparse chunk's CC has nothing to lock onto and landed microns away).

As in every tier, a CC with a component over 5 μm is dropped. Continuous mode then falls
back to endpoint chaining at that boundary without a warning.

Measuring avoids two endpoint-chaining failure modes: slow directional residual
accumulation, and a single bad boundary propagated down the cumulative chain when
a degree-d polynomial extrapolates poorly at t=±1. A wrong CC is still accepted
when both candidates overlap poorly (see Known limits in CLAUDE.md).
"""
function _ccseed_inter_continuous!(model::LegendrePolynomial, smld::SMLD; verbose::Int=0)
    n_dims = nDims(smld)
    n_datasets = smld.n_datasets
    n_datasets < 2 && return
    model.inter[1].dm .= 0.0

    # chunk d, intra removed, inter NOT applied (deepcopy: filter shares emitter refs)
    intra_only(d) = begin
        sd = deepcopy(filter_by_dataset(smld, d))
        for e in sd.emitters
            e.x = correctdrift(e.x, e.frame, model.intra[d].dm[1])
            e.y = correctdrift(e.y, e.frame, model.intra[d].dm[2])
            n_dims == 3 && (e.z = correctdrift(e.z, e.frame, model.intra[d].dm[3]))
        end
        sd
    end

    n_cc = 0; n_ep = 0
    prev = intra_only(1)
    last = 1
    for nn in 2:n_datasets
        cur = intra_only(nn)
        # Reference = last chunk with localizations placed in the corrected frame.
        ref = deepcopy(prev)
        for e in ref.emitters
            e.x -= model.inter[last].dm[1]
            e.y -= model.inter[last].dm[2]
            n_dims == 3 && (e.z -= model.inter[last].dm[3])
        end

        # endpoint-chained candidate (always available, even where CC can't lock on)
        ep = endpoint_drift(model.intra[nn-1], smld.n_frames) .- startpoint_drift(model.intra[nn])
        inter_ep = Float64[model.inter[nn-1].dm[k] + Float64(ep[k]) for k in 1:n_dims]

        if isempty(ref.emitters) || isempty(cur.emitters)
            model.inter[nn].dm .= inter_ep; n_ep += 1
            if !isempty(cur.emitters)
                prev = cur; last = nn
            end
            continue
        end

        refmat = n_dims == 2 ?
            permutedims([Float64[e.x for e in ref.emitters] Float64[e.y for e in ref.emitters]]) :
            permutedims([Float64[e.x for e in ref.emitters] Float64[e.y for e in ref.emitters] Float64[e.z for e in ref.emitters]])
        tree = KDTree(refmat; leafsize=10)
        ov_ep = _overlap_score(cur, inter_ep, tree, _CC_VERIFY_RADIUS, n_dims)

        # CC-measured candidate: rigid shift of cur onto the corrected ref = full inter[n].
        # Fine histogram (20 nm) — coarse bins lose the overlap contest to the smooth
        # endpoint prediction at clean boundaries even though CC is globally better.
        cc = findshift(ref, cur; histbinsize=_CC_CONT_HISTBIN, maxcount=_CC_MAXCOUNT)
        if !_cc_too_far(cc)
            inter_cc = Float64[Float64(cc[k]) for k in 1:n_dims]
            ov_cc = _overlap_score(cur, inter_cc, tree, _CC_VERIFY_RADIUS, n_dims)
            # Prefer the CC measurement: take endpoint only when CC overlap is clearly
            # worse (a real lock-on failure), not on a marginal edge — endpoint errors
            # chain down the sequence, so CC wins globally even when slightly behind at a
            # single boundary. A CC far from the endpoint prediction needs a significant lead.
            if ov_cc >= ov_ep - 0.05 &&
               !_far_without_evidence(cur, tree, inter_cc, inter_ep, n_dims; raw_cc = true,
                                      binsize = _CC_CONT_HISTBIN)
                model.inter[nn].dm .= inter_cc; n_cc += 1
            else
                model.inter[nn].dm .= inter_ep; n_ep += 1
            end
        else
            model.inter[nn].dm .= inter_ep; n_ep += 1
        end
        prev = cur; last = nn
    end
    verbose > 0 && @info("SMLMDriftCorrection: continuous CC-seed — $n_cc/$(n_datasets-1) " *
                         "boundaries from cross-correlation, $n_ep from endpoint fallback")
    return
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
