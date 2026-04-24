# Intra+Inter drift correction functions

function InterShift(ndims::Int)
    return InterShift(ndims, zeros(ndims))
end

function inter2theta(s::InterShift)
    θ=s.dm
end

function theta2inter!(s::InterShift,θ)
    s.dm.=θ
end

"""
Apply drift to simulated data.
"""
function applydrift(x::AbstractFloat, s::InterShift, dim::Int)
    return x + s.dm[dim]
end

"""
Apply drift correction to drifted data.
"""
function correctdrift(x::AbstractFloat, s::InterShift, dim::Int)
    return x - s.dm[dim]
end

"""
Apply x-, y- and z-drift to the data in the smld structure.
"""
function applydrift!(smld::SMLD, dm::AbstractIntraInter)
    n_dims = nDims(smld)

    for nn in eachindex(smld.emitters)
        smld.emitters[nn].x = applydrift(smld.emitters[nn].x, smld.emitters[nn].frame, dm.intra[smld.emitters[nn].dataset].dm[1])
        smld.emitters[nn].x = applydrift(smld.emitters[nn].x, dm.inter[smld.emitters[nn].dataset], 1)

        smld.emitters[nn].y = applydrift(smld.emitters[nn].y, smld.emitters[nn].frame, dm.intra[smld.emitters[nn].dataset].dm[2])
        smld.emitters[nn].y = applydrift(smld.emitters[nn].y, dm.inter[smld.emitters[nn].dataset], 2)

        if n_dims == 3
            smld.emitters[nn].z = applydrift(smld.emitters[nn].z, smld.emitters[nn].frame, dm.intra[smld.emitters[nn].dataset].dm[3])
            smld.emitters[nn].z = applydrift(smld.emitters[nn].z, dm.inter[smld.emitters[nn].dataset], 3)
        end
    end
end

"""
applydrift(smld, driftmodel) -> smld

Applies a drift model to SMLM data (for simulation/testing).
"""
function applydrift(smld::SMLD, driftmodel::AbstractIntraInter)
    smld_shifted = deepcopy(smld)
    applydrift!(smld_shifted, driftmodel)
    return smld_shifted
end

function correctdrift!(smld::SMLD, dm::AbstractIntraInter)
    n_dims = nDims(smld)

    for nn in eachindex(smld.emitters)
        smld.emitters[nn].x = correctdrift(smld.emitters[nn].x, smld.emitters[nn].frame, dm.intra[smld.emitters[nn].dataset].dm[1])
        smld.emitters[nn].x = correctdrift(smld.emitters[nn].x, dm.inter[smld.emitters[nn].dataset], 1)

        smld.emitters[nn].y = correctdrift(smld.emitters[nn].y, smld.emitters[nn].frame, dm.intra[smld.emitters[nn].dataset].dm[2])
        smld.emitters[nn].y = correctdrift(smld.emitters[nn].y, dm.inter[smld.emitters[nn].dataset], 2)
        if n_dims == 3
            smld.emitters[nn].z = correctdrift(smld.emitters[nn].z, smld.emitters[nn].frame, dm.intra[smld.emitters[nn].dataset].dm[3])
            smld.emitters[nn].z = correctdrift(smld.emitters[nn].z, dm.inter[smld.emitters[nn].dataset], 3)
        end
    end
end

function correctdrift(smld::SMLD, driftmodel::AbstractIntraInter)
    smld_shifted = deepcopy(smld)
    correctdrift!(smld_shifted, driftmodel)
    return smld_shifted
end

function correctdrift!(smld::SMLD, shift::Vector{Float64})
    n_dims = nDims(smld)

    for nn in eachindex(smld.emitters)
        smld.emitters[nn].x -= shift[1]
        smld.emitters[nn].y -= shift[2]
        if n_dims == 3
            smld.emitters[nn].z -= shift[3]
        end
    end
end

"""
    findintra!(intra, smld, dataset, maxn; skip_init=false, ref_coords=nothing)

Find and correct intra-dataset drift using entropy minimization with
adaptive KDTree neighbor rebuilding.

Each `optimize(...)` call sees a frozen KDTree, so the objective is deterministic
for Nelder-Mead. Rebuilds happen in an outer loop, triggered only when the drift
polynomial has moved by more than `rebuild_threshold` (μm) since the last rebuild.
Rebuild gating uses drift **vectors** at test frames `[1, mid, end]` (see
`drift_vecs!` / `max_drift_vec_delta` in costfuns.jl), so direction/sign changes
at the same magnitude still trigger a rebuild.

Two modes:
- **Per-dataset** (default): Neighbors come from this dataset only. Correct for
  iteration 1 when inter is not yet populated.
- **Merged-cloud** (when `ref_coords` is passed): Neighbors come from the combined
  cloud of this dataset + all other (already-corrected) datasets. Used for
  iteration 2+ to anchor intra against the same structural scaffold that
  `findinter!` uses — stops intra ↔ inter from chasing each other in a limit
  cycle when drift is small.

# Keyword Arguments
- `skip_init=false`: skip random re-init (warm-started externally)
- `ref_coords=nothing`: `NamedTuple` with full-array handoff (one allocation
  per caller, no per-dataset copies).
    - 2D: `(x_all, y_all, σ_x_all, σ_y_all, ds_all, exclude_dataset)`
    - 3D: `(x_all, y_all, z_all, σ_x_all, σ_y_all, σ_z_all, ds_all, exclude_dataset)`
  All coordinate arrays are fully-corrected in the common frame; `ds_all` is
  the dataset id per element; `exclude_dataset` is the dataset to *exclude*
  from the reference (the one being optimized).
"""
function findintra!(intra::AbstractIntraDrift,
    smld::SMLD,
    dataset::Int,
    maxn::Int;
    skip_init::Bool = false,
    ref_coords::Union{Nothing, NamedTuple} = nothing)

    idx = [e.dataset for e in smld.emitters] .== dataset
    emitters = smld.emitters[idx]
    N = length(emitters)

    # Guard against degenerate per-dataset subsets (empty, or too few points for KNN)
    if N < 2
        return
    end

    # Extract vectors directly
    x = Float64[e.x for e in emitters]
    y = Float64[e.y for e in emitters]
    σ_x = Float64[e.σ_x for e in emitters]
    σ_y = Float64[e.σ_y for e in emitters]
    framenum = Int[e.frame for e in emitters]

    if intra.ndims == 3
        z = Float64[e.z for e in emitters]
        σ_z = Float64[e.σ_z for e in emitters]
    end

    # Initialize with small random values (unless warmstarted externally)
    nframes = smld.n_frames
    if !skip_init
        initialize_random!(intra, 0.01, nframes)
    end

    # Convert to parameter vector for optimization
    θ0 = Float64.(intra2theta(intra))

    # Pre-allocate work arrays
    x_work = similar(x)
    y_work = similar(y)
    z_work = intra.ndims == 3 ? similar(z) : Float64[]

    rebuild_threshold = 0.1  # μm (100 nm) — rebuild when drift changes significantly

    if ref_coords === nothing
        # -------- per-dataset adaptive path (iteration 1 / no ref) --------
        k = min(maxn, N - 1)
        state = NeighborState(N, k, rebuild_threshold, intra.ndims)

        if intra.ndims == 2
            build_neighbors!(state, x, y)
            drift_vecs!(state.last_drift_vecs, intra, nframes)  # record drift at build time
            myfun = θ -> costfun_entropy_intra_2D_adaptive(θ, x, y, σ_x, σ_y, framenum, maxn, intra,
                                                           state, nframes;
                                                           x_work=x_work, y_work=y_work)
        else
            build_neighbors!(state, x, y, z)
            drift_vecs!(state.last_drift_vecs, intra, nframes)
            myfun = θ -> costfun_entropy_intra_3D_adaptive(θ, x, y, z, σ_x, σ_y, σ_z, framenum, maxn, intra,
                                                           state, nframes;
                                                           x_work=x_work, y_work=y_work, z_work=z_work)
        end

        state.allow_rebuild = false
        opt = Optim.Options(iterations=10000, f_abstol=1e-2, x_abstol=1e-4, show_trace=false)
        max_outer = 3
        local res
        for _ in 1:max_outer
            res = optimize(myfun, θ0, opt)
            theta2intra!(intra, res.minimizer)
            θ0 = res.minimizer

            drift_vecs!(state.current_drift_vecs, intra, nframes)
            if max_drift_vec_delta(state.current_drift_vecs, state.last_drift_vecs) < state.rebuild_threshold
                break
            end

            @inbounds for i in 1:N
                x_work[i] = correctdrift(x[i], framenum[i], intra.dm[1])
                y_work[i] = correctdrift(y[i], framenum[i], intra.dm[2])
            end
            if intra.ndims == 2
                build_neighbors!(state, x_work, y_work)
            else
                @inbounds for i in 1:N
                    z_work[i] = correctdrift(z[i], framenum[i], intra.dm[3])
                end
                build_neighbors!(state, x_work, y_work, z_work)
            end
            state.last_drift_vecs .= state.current_drift_vecs
        end
        theta2intra!(intra, res.minimizer)
        return
    end

    # ---------------- merged-cloud path (iteration 2+) ----------------
    x_all = ref_coords.x_all
    y_all = ref_coords.y_all
    σ_x_all = ref_coords.σ_x_all
    σ_y_all = ref_coords.σ_y_all
    ds_all = ref_coords.ds_all
    exclude_ds = ref_coords.exclude_dataset
    if intra.ndims == 3
        z_all = ref_coords.z_all
        σ_z_all = ref_coords.σ_z_all
    end

    # Count reference size once.
    N_ref = 0
    @inbounds for i in eachindex(ds_all)
        if ds_all[i] != exclude_ds
            N_ref += 1
        end
    end
    if N_ref == 0
        # No reference available — fall back to per-dataset.
        return findintra!(intra, smld, dataset, maxn; skip_init=true)
    end

    # Allocate once: filtered σ_ref arrays (needed for per-pair divergence),
    # and data_combined. Ref columns of data_combined are filled ONCE here;
    # the cost function only rewrites the probe columns (1:N) per evaluation.
    σ_x_ref = Vector{Float64}(undef, N_ref)
    σ_y_ref = Vector{Float64}(undef, N_ref)
    σ_z_ref = intra.ndims == 3 ? Vector{Float64}(undef, N_ref) : Float64[]
    data_combined = Matrix{Float64}(undef, intra.ndims, N + N_ref)

    let ref_idx = 0
        @inbounds for i in eachindex(ds_all)
            if ds_all[i] != exclude_ds
                ref_idx += 1
                data_combined[1, N + ref_idx] = x_all[i]
                data_combined[2, N + ref_idx] = y_all[i]
                σ_x_ref[ref_idx] = σ_x_all[i]
                σ_y_ref[ref_idx] = σ_y_all[i]
                if intra.ndims == 3
                    data_combined[3, N + ref_idx] = z_all[i]
                    σ_z_ref[ref_idx] = σ_z_all[i]
                end
            end
        end
    end

    k = min(maxn, N + N_ref - 1)
    state = NeighborState(N, k, rebuild_threshold, intra.ndims)
    # last_drift_vecs initialised to +Inf so first cost eval always rebuilds.

    if intra.ndims == 2
        myfun = θ -> costfun_entropy_intra_2D_merged(θ,
            x, y, σ_x, σ_y, framenum,
            σ_x_ref, σ_y_ref,
            maxn, intra, nframes,
            data_combined, state;
            x_work=x_work, y_work=y_work)
    else
        myfun = θ -> costfun_entropy_intra_3D_merged(θ,
            x, y, z, σ_x, σ_y, σ_z, framenum,
            σ_x_ref, σ_y_ref, σ_z_ref,
            maxn, intra, nframes,
            data_combined, state;
            x_work=x_work, y_work=y_work, z_work=z_work)
    end

    # Trigger the initial KDTree build while allow_rebuild=true, then freeze.
    state.allow_rebuild = true
    myfun(θ0)
    state.allow_rebuild = false

    opt = Optim.Options(iterations=10000, f_abstol=1e-2, x_abstol=1e-4, show_trace=false)
    max_outer = 3
    local res
    for _ in 1:max_outer
        res = optimize(myfun, θ0, opt)
        theta2intra!(intra, res.minimizer)
        θ0 = res.minimizer

        drift_vecs!(state.current_drift_vecs, intra, nframes)
        if max_drift_vec_delta(state.current_drift_vecs, state.last_drift_vecs) < state.rebuild_threshold
            break
        end

        # Force one rebuild at current θ0, then refreeze for the next optimize pass.
        state.allow_rebuild = true
        myfun(θ0)
        state.allow_rebuild = false
    end
    theta2intra!(intra, res.minimizer)
end

"""
    filter_by_dataset(smld, datasets)

Filter SMLD to include only emitters from specified dataset(s).
"""
function filter_by_dataset(smld::SMLD, dataset::Int)
    idx = [e.dataset == dataset for e in smld.emitters]
    return filter_emitters(smld, idx)
end

function filter_by_dataset(smld::SMLD, datasets::Vector{Int})
    idx = [e.dataset in datasets for e in smld.emitters]
    return filter_emitters(smld, idx)
end

"""
    findinter!(dm, smld_uncorrected, dataset_n, ref_datasets, maxn; kwargs...)

Find and correct inter-dataset drift using entropy minimization.

Aligns `dataset_n` to the reference datasets by minimizing the entropy
of the combined point cloud. Uses cross-correlation for initial guess,
then refines with entropy optimization.

# Arguments
- `dm`: drift model (modified in place)
- `smld_uncorrected`: original SMLD data (not corrected)
- `dataset_n`: dataset index to shift/align
- `ref_datasets`: vector of reference dataset indices
- `maxn`: maximum neighbors for entropy calculation

# Keyword Arguments
- `precomputed_corrected`: pre-corrected SMLD (skips `correctdrift` call). Use when
  calling from threaded loops where `correctdrift` can be computed once outside the loop.
- `regularization_target`: target shift to regularize towards (default: nothing)
- `regularization_lambda`: regularization strength (default: 0.0)
  Cost becomes: entropy + λ*||shift - target||²
"""
function findinter!(dm::AbstractIntraInter,
    smld_uncorrected::SMLD,
    dataset_n::Int,
    ref_datasets::Vector{Int},
    maxn::Int;
    precomputed_corrected::Union{Nothing, SMLD} = nothing,
    regularization_target::Union{Nothing, Vector{Float64}} = nothing,
    regularization_lambda::Float64 = 0.0)

    n_dims = nDims(smld_uncorrected)

    # Get UNCORRECTED coords for dataset_n
    idx_n = [e.dataset == dataset_n for e in smld_uncorrected.emitters]
    emitters_n = smld_uncorrected.emitters[idx_n]

    # Apply ONLY intra-drift correction (not inter) to dataset_n
    # This way the optimizer finds the TOTAL inter-shift needed.
    x_n = Float64[correctdrift(e.x, e.frame, dm.intra[dataset_n].dm[1]) for e in emitters_n]
    y_n = Float64[correctdrift(e.y, e.frame, dm.intra[dataset_n].dm[2]) for e in emitters_n]
    σ_x_n = Float64[e.σ_x for e in emitters_n]
    σ_y_n = Float64[e.σ_y for e in emitters_n]
    if n_dims == 3
        z_n = Float64[correctdrift(e.z, e.frame, dm.intra[dataset_n].dm[3]) for e in emitters_n]
        σ_z_n = Float64[e.σ_z for e in emitters_n]
    end

    # Correct reference datasets fully (intra + inter) for comparison
    smld_corrected = precomputed_corrected !== nothing ? precomputed_corrected : correctdrift(smld_uncorrected, dm)

    # Extract CORRECTED coords from reference datasets
    idx_ref = [e.dataset in ref_datasets for e in smld_corrected.emitters]
    emitters_ref = smld_corrected.emitters[idx_ref]

    # Guard against empty dataset_n or empty reference set (e.g., filtered-out
    # dataset, or a ref_datasets list with no surviving emitters). Without this
    # guard the KDTree build would throw on 0 points.
    if length(emitters_n) == 0 || length(emitters_ref) == 0
        return 0.0
    end

    x_ref = Float64[e.x for e in emitters_ref]
    y_ref = Float64[e.y for e in emitters_ref]
    σ_x_ref = Float64[e.σ_x for e in emitters_ref]
    σ_y_ref = Float64[e.σ_y for e in emitters_ref]
    if n_dims == 3
        z_ref = Float64[e.z for e in emitters_ref]
        σ_z_ref = Float64[e.σ_z for e in emitters_ref]
    end

    inter = dm.inter[dataset_n]

    # Initial guess: use current inter value if non-zero, otherwise try CC
    # This allows the second pass to refine from the first pass result.
    if any(abs.(inter.dm) .> 1e-10)
        # Use current inter as starting point (preserves first pass result)
        θ0 = Float64.(inter.dm)
    else
        # First pass: try cross-correlation for initial guess
        # Note: For CC we need both datasets in the same reference frame
        smld_ref = filter_by_dataset(smld_corrected, ref_datasets)

        # Create intra-only corrected SMLD for dataset_n (to match what optimizer uses)
        # This is a temporary copy for CC only
        smld_n_intra_only = deepcopy(filter_by_dataset(smld_uncorrected, dataset_n))
        for i in eachindex(smld_n_intra_only.emitters)
            e = smld_n_intra_only.emitters[i]
            smld_n_intra_only.emitters[i].x = correctdrift(e.x, e.frame, dm.intra[dataset_n].dm[1])
            smld_n_intra_only.emitters[i].y = correctdrift(e.y, e.frame, dm.intra[dataset_n].dm[2])
            if n_dims == 3
                smld_n_intra_only.emitters[i].z = correctdrift(e.z, e.frame, dm.intra[dataset_n].dm[3])
            end
        end

        θ0 = zeros(Float64, n_dims)
        try
            cc_shift = findshift(smld_ref, smld_n_intra_only; histbinsize=0.05)  # 50nm bins
            # findshift(A, B) returns the shift of B relative to A
            # If B is shifted by +Δ relative to A, findshift returns +Δ
            # To correct B back to A, inter.dm = Δ (correctdrift subtracts it)
            # Sanity check: shift should be < 5 μm typically
            if maximum(abs.(cc_shift)) < 5.0
                θ0 = Float64.(cc_shift)
            end
        catch
            # Keep zero initialization
        end
    end

    # Pre-allocate work arrays
    N_n = length(x_n)
    N_ref = length(x_ref)
    x_work = similar(x_n)
    y_work = similar(y_n)

    # Adaptive neighbor state - only rebuild KDTree when shift changes significantly
    k = min(maxn, N_n + N_ref - 1)
    rebuild_threshold = 0.1  # μm (100nm) - same as intra-dataset

    # Build cost function with optional regularization
    if n_dims == 2
        # Pre-allocate combined data matrix for KDTree (avoids allocation per iteration)
        data_combined = Matrix{Float64}(undef, 2, N_n + N_ref)
        state = InterNeighborState(N_n, k, rebuild_threshold)
        entropy_cost = θ -> costfun_entropy_inter_2D_merged(θ,
            x_n, y_n, σ_x_n, σ_y_n,
            x_ref, y_ref, σ_x_ref, σ_y_ref,
            maxn, inter;
            x_work=x_work, y_work=y_work, data_combined=data_combined, state=state)
    else # 3D
        z_work = similar(z_n)
        data_combined = Matrix{Float64}(undef, 3, N_n + N_ref)
        state = InterNeighborState3D(N_n, k, rebuild_threshold)
        entropy_cost = θ -> costfun_entropy_inter_3D_merged(θ,
            x_n, y_n, z_n, σ_x_n, σ_y_n, σ_z_n,
            x_ref, y_ref, z_ref, σ_x_ref, σ_y_ref, σ_z_ref,
            maxn, inter;
            x_work=x_work, y_work=y_work, z_work=z_work, data_combined=data_combined, state=state)
    end

    # Add regularization if specified: cost = entropy + λ*||θ - target||²
    if regularization_target !== nothing && regularization_lambda > 0.0
        myfun = θ -> entropy_cost(θ) + regularization_lambda * sum((θ .- regularization_target).^2)
    else
        myfun = entropy_cost
    end

    # Optimize with gradient-based method for better convergence.
    # BFGS handles flat entropy landscapes better than Nelder-Mead (2x slower but ~4x more accurate).
    #
    # Determinism: each optimize() call sees a frozen KDTree. The outer loop rebuilds
    # neighbors only between optimize() calls, after the shift has moved beyond
    # rebuild_threshold. This keeps BFGS's gradient approximation consistent — the
    # function is not mutating underneath the optimizer.
    opt = Optim.Options(iterations=10000, g_abstol=1e-8, show_trace=false)
    # Trigger the initial KDTree build at θ0 while allow_rebuild=true, then freeze.
    state.allow_rebuild = true
    entropy_cost(θ0)
    state.allow_rebuild = false

    max_outer = 3
    local res
    for _ in 1:max_outer
        res = optimize(myfun, θ0, BFGS())
        theta2inter!(inter, res.minimizer)
        θ0 = res.minimizer

        # Check shift change from last rebuild
        Δ2 = 0.0
        for dim in 1:n_dims
            Δ2 += (θ0[dim] - state.last_shift[dim])^2
        end
        if sqrt(Δ2) < state.rebuild_threshold
            break
        end

        # Force one rebuild at current θ0, then freeze for the next optimize pass
        state.allow_rebuild = true
        entropy_cost(θ0)
        state.allow_rebuild = false
    end
    theta2inter!(inter, res.minimizer)
    return res.minimum
end
