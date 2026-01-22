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
    findinter!(dm, smld_uncorrected, dataset1, dataset2, maxn)

Find and correct inter-dataset drift using entropy minimization.
"""
function findinter!(dm::AbstractIntraInter,
    smld_uncorrected::SMLD,
    dataset1::Int,
    dataset2::Vector{Int},
    maxn::Int)

    n_dims = nDims(smld_uncorrected)

    # Get uncorrected coords for dataset1
    idx1 = [e.dataset for e in smld_uncorrected.emitters] .== dataset1
    emitters1 = smld_uncorrected.emitters[idx1]

    x1 = Float64[e.x for e in emitters1]
    y1 = Float64[e.y for e in emitters1]
    σ_x1 = Float64[e.σ_x for e in emitters1]
    σ_y1 = Float64[e.σ_y for e in emitters1]
    if n_dims == 3
        z1 = Float64[e.z for e in emitters1]
        σ_z1 = Float64[e.σ_z for e in emitters1]
    end

    # Correct everything with current model
    smld = correctdrift(smld_uncorrected, dm)

    # Use current model as starting point
    inter = dm.inter[dataset1]
    θ0 = Float64.(inter2theta(inter))

    # Pre-allocate work arrays
    x_work = similar(x1)
    y_work = similar(y1)

    if n_dims == 2
        myfun = θ -> costfun_entropy_inter_2D(θ, x1, y1, σ_x1, σ_y1, maxn, inter;
                                               divmethod="KL", x_work=x_work, y_work=y_work)
    else # 3D
        z_work = similar(z1)
        myfun = θ -> costfun_entropy_inter_3D(θ, x1, y1, z1, σ_x1, σ_y1, σ_z1, maxn, inter;
                                               divmethod="KL", x_work=x_work, y_work=y_work, z_work=z_work)
    end

    # Optimize with convergence tolerances
    opt = Optim.Options(iterations=10000, f_abstol=1e-2, x_abstol=1e-4, show_trace=false)
    res = optimize(myfun, θ0, opt)
    theta2inter!(inter, res.minimizer)
    return res.minimum
end
