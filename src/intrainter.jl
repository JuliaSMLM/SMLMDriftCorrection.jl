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
    findintra!(intra, smld, dataset, maxn)

Find and correct intra-dataset drift using entropy minimization with
adaptive KDTree neighbor rebuilding.

Uses KL divergence entropy cost function with adaptive neighbor tracking.
Only rebuilds neighbors when drift changes by more than 0.5 μm.
"""
function findintra!(intra::AbstractIntraDrift,
    smld::SMLD,
    dataset::Int,
    maxn::Int)

    idx = [e.dataset for e in smld.emitters] .== dataset
    emitters = smld.emitters[idx]
    N = length(emitters)

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

    # Initialize with small random values
    nframes = smld.n_frames
    initialize_random!(intra, 0.01, nframes)

    # Convert to parameter vector for optimization
    θ0 = Float64.(intra2theta(intra))

    # Pre-allocate work arrays
    x_work = similar(x)
    y_work = similar(y)

    # Adaptive entropy cost function
    k = min(maxn, N - 1)
    rebuild_threshold = 0.5  # μm - entropy is robust, rebuild rarely
    state = NeighborState(N, k, rebuild_threshold)

    if intra.ndims == 2
        build_neighbors!(state, x, y)
        myfun = θ -> costfun_entropy_intra_2D_adaptive(θ, x, y, σ_x, σ_y, framenum, maxn, intra,
                                                       state, nframes;
                                                       divmethod="KL", x_work=x_work, y_work=y_work)
    else # 3D
        z_work = similar(z)
        build_neighbors!(state, x, y, z)
        myfun = θ -> costfun_entropy_intra_3D_adaptive(θ, x, y, z, σ_x, σ_y, σ_z, framenum, maxn, intra,
                                                       state, nframes;
                                                       divmethod="KL", x_work=x_work, y_work=y_work, z_work=z_work)
    end

    # Optimize with convergence tolerances
    opt = Optim.Options(iterations=10000, f_abstol=1e-2, x_abstol=1e-4, show_trace=false)
    res = optimize(myfun, θ0, opt)
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
    findinter!(dm, smld_uncorrected, dataset_n, ref_datasets, maxn)

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
"""
function findinter!(dm::AbstractIntraInter,
    smld_uncorrected::SMLD,
    dataset_n::Int,
    ref_datasets::Vector{Int},
    maxn::Int)

    n_dims = nDims(smld_uncorrected)

    # Get uncorrected coords for dataset_n (these will be shifted)
    idx_n = [e.dataset == dataset_n for e in smld_uncorrected.emitters]
    emitters_n = smld_uncorrected.emitters[idx_n]

    x_n = Float64[e.x for e in emitters_n]
    y_n = Float64[e.y for e in emitters_n]
    σ_x_n = Float64[e.σ_x for e in emitters_n]
    σ_y_n = Float64[e.σ_y for e in emitters_n]
    if n_dims == 3
        z_n = Float64[e.z for e in emitters_n]
        σ_z_n = Float64[e.σ_z for e in emitters_n]
    end

    # Correct everything with current model to get reference positions
    smld_corrected = correctdrift(smld_uncorrected, dm)

    # Extract CORRECTED coords from reference datasets
    idx_ref = [e.dataset in ref_datasets for e in smld_corrected.emitters]
    emitters_ref = smld_corrected.emitters[idx_ref]

    x_ref = Float64[e.x for e in emitters_ref]
    y_ref = Float64[e.y for e in emitters_ref]
    σ_x_ref = Float64[e.σ_x for e in emitters_ref]
    σ_y_ref = Float64[e.σ_y for e in emitters_ref]
    if n_dims == 3
        z_ref = Float64[e.z for e in emitters_ref]
        σ_z_ref = Float64[e.σ_z for e in emitters_ref]
    end

    # Get cross-correlation initial guess (coarse estimate)
    # Note: This works best when datasets image the same structure (real data).
    # For simulated data with independent random emitters, CC may not help.
    smld_n_corrected = filter_by_dataset(smld_corrected, dataset_n)
    smld_ref = filter_by_dataset(smld_corrected, ref_datasets)

    inter = dm.inter[dataset_n]

    # Try cross-correlation, but use zero init if result is unreasonable
    θ0 = zeros(Float64, n_dims)
    try
        cc_shift = findshift(smld_ref, smld_n_corrected; histbinsize=0.05)  # 50nm bins
        # findshift(A, B) returns -(B - A), so we need θ = -cc_shift
        # Sanity check: shift should be < 5 μm typically
        if maximum(abs.(cc_shift)) < 5.0
            θ0 = Float64.(-cc_shift)
        end
    catch
        # Keep zero initialization
    end

    # Pre-allocate work arrays
    N_n = length(x_n)
    N_ref = length(x_ref)
    x_work = similar(x_n)
    y_work = similar(y_n)

    # Adaptive neighbor state - only rebuild KDTree when shift changes significantly
    k = min(maxn, N_n + N_ref - 1)
    rebuild_threshold = 0.5  # μm - same as intra-dataset

    if n_dims == 2
        # Pre-allocate combined data matrix for KDTree (avoids allocation per iteration)
        data_combined = Matrix{Float64}(undef, 2, N_n + N_ref)
        state = InterNeighborState(N_n, k, rebuild_threshold)
        myfun = θ -> costfun_entropy_inter_2D_merged(θ,
            x_n, y_n, σ_x_n, σ_y_n,
            x_ref, y_ref, σ_x_ref, σ_y_ref,
            maxn, inter;
            divmethod="KL", x_work=x_work, y_work=y_work, data_combined=data_combined, state=state)
    else # 3D
        z_work = similar(z_n)
        data_combined = Matrix{Float64}(undef, 3, N_n + N_ref)
        state = InterNeighborState3D(N_n, k, rebuild_threshold)
        myfun = θ -> costfun_entropy_inter_3D_merged(θ,
            x_n, y_n, z_n, σ_x_n, σ_y_n, σ_z_n,
            x_ref, y_ref, z_ref, σ_x_ref, σ_y_ref, σ_z_ref,
            maxn, inter;
            divmethod="KL", x_work=x_work, y_work=y_work, z_work=z_work, data_combined=data_combined, state=state)
    end

    # Optimize with convergence tolerances
    opt = Optim.Options(iterations=10000, f_abstol=1e-2, x_abstol=1e-4, show_trace=false)
    res = optimize(myfun, θ0, opt)
    theta2inter!(inter, res.minimizer)
    return res.minimum
end
