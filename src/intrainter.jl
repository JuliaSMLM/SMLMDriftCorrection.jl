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
applydrift(smld::SMLMData.Emitter2DFit, driftmodel::AbstractIntraInter) ->
  SMLMData.Emitter2DFit

Applies a drift model to the Single-Molecule Localization Microscopy (SMLM)
data and returns the drift-corrected data.
"""
function applydrift(smld::SMLD, driftmodel::AbstractIntraInter)
    smld_shifted = deepcopy(smld)
    applydrift!(smld_shifted::SMLD, driftmodel::AbstractIntraInter)
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
Find and correct intra-dataset drift.
Optimized version with pre-allocated work arrays and adaptive neighbor rebuilding.

# Fields:
- intra:            intra-dataset structure
- cost_fun:         cost function: {"Kdtree", "Entropy", "SymEntropy", "Bhattacharyya", "Mahalanobis"}
- smld:             data structure containing coordinate data
- dataset:          dataset number to operate on
- d_cutoff:         cutoff distance (for Kdtree)
- maxn:             maximum number of neighbors considered (for Entropy)

# Performance Note:
For Kdtree cost function, uses adaptive neighbor rebuilding which only rebuilds
the KDTree when drift magnitude changes significantly (threshold = 2 × d_cutoff).
This gives ~100x speedup compared to rebuilding every iteration.
"""
function findintra!(intra::AbstractIntraDrift,
    cost_fun::String,
    smld::SMLD,
    dataset::Int,
    d_cutoff::AbstractFloat,
    maxn::Int)

    idx = [e.dataset for e in smld.emitters] .== dataset
    emitters = smld.emitters[idx]
    N = length(emitters)

    # Extract vectors directly (avoid matrix construction)
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
    rscale = 0.01
    nframes = smld.n_frames
    initialize_random!(intra, rscale, nframes)

    # Convert intra drift parameters to a single vector for optimization
    θ0 = Float64.(intra2theta(intra))

    # Pre-allocate work arrays (reused across all cost function calls)
    x_work = similar(x)
    y_work = similar(y)

    # Select cost function based on dimensionality and method
    if intra.ndims == 2
        if cost_fun == "Kdtree"
            # Use adaptive neighbor approach - rebuild threshold is 2x d_cutoff
            # This means we only rebuild when drift changes enough to significantly
            # affect neighbor relationships
            k = min(4, N - 1)
            rebuild_threshold = 2.0 * Float64(d_cutoff)
            state = NeighborState(N, k, rebuild_threshold)

            # Build initial neighbors from uncorrected coordinates
            build_neighbors!(state, x, y)

            myfun = θ -> costfun_kdtree_intra_2D_adaptive(θ, x, y, framenum, Float64(d_cutoff), intra,
                                                          state, nframes;
                                                          x_work=x_work, y_work=y_work)
        elseif cost_fun == "Entropy"
            myfun = θ -> costfun_entropy_intra_2D(θ, x, y, σ_x, σ_y, framenum, maxn, intra;
                                                   divmethod="KL", x_work=x_work, y_work=y_work)
        elseif cost_fun == "SymEntropy"
            myfun = θ -> costfun_entropy_intra_2D(θ, x, y, σ_x, σ_y, framenum, maxn, intra;
                                                   divmethod="Symmetric", x_work=x_work, y_work=y_work)
        elseif cost_fun == "Bhattacharyya"
            myfun = θ -> costfun_entropy_intra_2D(θ, x, y, σ_x, σ_y, framenum, maxn, intra;
                                                   divmethod="Bhattacharyya", x_work=x_work, y_work=y_work)
        elseif cost_fun == "Mahalanobis"
            myfun = θ -> costfun_entropy_intra_2D(θ, x, y, σ_x, σ_y, framenum, maxn, intra;
                                                   divmethod="Mahalanobis", x_work=x_work, y_work=y_work)
        else
            error("cost_fun not recognized: $cost_fun")
        end
    else # 3D
        z_work = similar(z)
        if cost_fun == "Kdtree"
            k = min(4, N - 1)
            rebuild_threshold = 2.0 * Float64(d_cutoff)
            state = NeighborState(N, k, rebuild_threshold)

            build_neighbors!(state, x, y, z)

            myfun = θ -> costfun_kdtree_intra_3D_adaptive(θ, x, y, z, framenum, Float64(d_cutoff), intra,
                                                          state, nframes;
                                                          x_work=x_work, y_work=y_work, z_work=z_work)
        elseif cost_fun == "Entropy"
            myfun = θ -> costfun_entropy_intra_3D(θ, x, y, z, σ_x, σ_y, σ_z, framenum, maxn, intra;
                                                   divmethod="KL", x_work=x_work, y_work=y_work, z_work=z_work)
        elseif cost_fun == "SymEntropy"
            myfun = θ -> costfun_entropy_intra_3D(θ, x, y, z, σ_x, σ_y, σ_z, framenum, maxn, intra;
                                                   divmethod="Symmetric", x_work=x_work, y_work=y_work, z_work=z_work)
        elseif cost_fun == "Bhattacharyya"
            myfun = θ -> costfun_entropy_intra_3D(θ, x, y, z, σ_x, σ_y, σ_z, framenum, maxn, intra;
                                                   divmethod="Bhattacharyya", x_work=x_work, y_work=y_work, z_work=z_work)
        elseif cost_fun == "Mahalanobis"
            myfun = θ -> costfun_entropy_intra_3D(θ, x, y, z, σ_x, σ_y, σ_z, framenum, maxn, intra;
                                                   divmethod="Mahalanobis", x_work=x_work, y_work=y_work, z_work=z_work)
        else
            error("cost_fun not recognized: $cost_fun")
        end
    end

    opt = Optim.Options(iterations = 10000, show_trace = false)
    res = optimize(myfun, θ0, opt)
    θ_found = res.minimizer

    theta2intra!(intra, θ_found)
end

"""
Find and correct inter-dataset drift.
Optimized version with pre-allocated work arrays.

# Fields:
- dm:               drift model structure
- cost_fun:         cost function: {"Kdtree", "Entropy", ...}
- smld_uncorrected: data structure containing uncorrected coordinate data
- dataset1:         dataset number for the dataset to correct
- dataset2:         reference dataset numbers
- d_cutoff:         cutoff distance (for Kdtree)
- maxn:             maximum number of neighbors considered (for Entropy)
- histbinsize:      histogram bin size for optional cross-correlation correction
"""
function findinter!(dm::AbstractIntraInter,
    cost_fun::String,
    smld_uncorrected::SMLD,
    dataset1::Int,
    dataset2::Vector{Int},
    d_cutoff::AbstractFloat,
    maxn::Int,
    histbinsize::AbstractFloat)

    n_dims = nDims(smld_uncorrected)

    # Get uncorrected coords for dataset1
    idx1 = [e.dataset for e in smld_uncorrected.emitters] .== dataset1
    idx1_find = findall(idx1)
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

    # Get corrected coords for reference datasets (build sum image)
    idx2 = zeros(Bool, length(smld.emitters))
    for nn in dataset2
        idx2 .= idx2 .| ([e.dataset for e in smld.emitters] .== nn)
    end
    emitters2 = smld.emitters[idx2]

    x2 = Float64[e.x for e in emitters2]
    y2 = Float64[e.y for e in emitters2]
    if n_dims == 3
        z2 = Float64[e.z for e in emitters2]
    end

    # Optional cross-correlation correction
    if histbinsize > 0.0
        smld1 = filter_emitters(smld, idx1_find)
        smld2 = filter_emitters(smld, idx2)
        shift = findshift(smld1, smld2; histbinsize=histbinsize)
        correctdrift!(smld1, shift)
        for nn in eachindex(idx1_find)
            smld.emitters[idx1_find[nn]].x = smld1.emitters[nn].x
            smld.emitters[idx1_find[nn]].y = smld1.emitters[nn].y
            if n_dims == 3
                smld.emitters[idx1_find[nn]].z = smld1.emitters[nn].z
            end
        end
        if cost_fun == "None"
            theta2inter!(dm.inter[dataset1], shift)
            return 0.0
        end
    end

    # Build static KDTree from reference data
    if n_dims == 2
        data_ref = Matrix{Float64}(undef, 2, length(x2))
        @inbounds for i in eachindex(x2)
            data_ref[1, i] = x2[i]
            data_ref[2, i] = y2[i]
        end
    else
        data_ref = Matrix{Float64}(undef, 3, length(x2))
        @inbounds for i in eachindex(x2)
            data_ref[1, i] = x2[i]
            data_ref[2, i] = y2[i]
            data_ref[3, i] = z2[i]
        end
    end
    kdtree = KDTree(data_ref; leafsize=10)

    # Use current model as starting point
    inter = dm.inter[dataset1]
    θ0 = Float64.(inter2theta(inter))

    # Pre-allocate work arrays
    x_work = similar(x1)
    y_work = similar(y1)

    if n_dims == 2
        if cost_fun == "Kdtree"
            myfun = θ -> costfun_kdtree_inter_2D(θ, x1, y1, kdtree, Float64(d_cutoff), inter;
                                                  x_work=x_work, y_work=y_work)
        elseif cost_fun == "Entropy"
            myfun = θ -> costfun_entropy_inter_2D(θ, x1, y1, σ_x1, σ_y1, maxn, inter;
                                                   divmethod="KL", x_work=x_work, y_work=y_work)
        elseif cost_fun == "SymEntropy"
            myfun = θ -> costfun_entropy_inter_2D(θ, x1, y1, σ_x1, σ_y1, maxn, inter;
                                                   divmethod="Symmetric", x_work=x_work, y_work=y_work)
        elseif cost_fun == "Bhattacharyya"
            myfun = θ -> costfun_entropy_inter_2D(θ, x1, y1, σ_x1, σ_y1, maxn, inter;
                                                   divmethod="Bhattacharyya", x_work=x_work, y_work=y_work)
        elseif cost_fun == "Mahalanobis"
            myfun = θ -> costfun_entropy_inter_2D(θ, x1, y1, σ_x1, σ_y1, maxn, inter;
                                                   divmethod="Mahalanobis", x_work=x_work, y_work=y_work)
        else
            error("cost_fun not recognized: $cost_fun")
        end
    else # 3D
        z_work = similar(z1)
        if cost_fun == "Kdtree"
            myfun = θ -> costfun_kdtree_inter_3D(θ, x1, y1, z1, kdtree, Float64(d_cutoff), inter;
                                                  x_work=x_work, y_work=y_work, z_work=z_work)
        elseif cost_fun == "Entropy"
            myfun = θ -> costfun_entropy_inter_3D(θ, x1, y1, z1, σ_x1, σ_y1, σ_z1, maxn, inter;
                                                   divmethod="KL", x_work=x_work, y_work=y_work, z_work=z_work)
        elseif cost_fun == "SymEntropy"
            myfun = θ -> costfun_entropy_inter_3D(θ, x1, y1, z1, σ_x1, σ_y1, σ_z1, maxn, inter;
                                                   divmethod="Symmetric", x_work=x_work, y_work=y_work, z_work=z_work)
        elseif cost_fun == "Bhattacharyya"
            myfun = θ -> costfun_entropy_inter_3D(θ, x1, y1, z1, σ_x1, σ_y1, σ_z1, maxn, inter;
                                                   divmethod="Bhattacharyya", x_work=x_work, y_work=y_work, z_work=z_work)
        elseif cost_fun == "Mahalanobis"
            myfun = θ -> costfun_entropy_inter_3D(θ, x1, y1, z1, σ_x1, σ_y1, σ_z1, maxn, inter;
                                                   divmethod="Mahalanobis", x_work=x_work, y_work=y_work, z_work=z_work)
        else
            error("cost_fun not recognized: $cost_fun")
        end
    end

    opt = Optim.Options(iterations = 10000, show_trace = false)
    res = optimize(myfun, θ0, opt)
    θ_found = res.minimizer

    theta2inter!(inter, θ_found)
    return res.minimum
end

"""
Find and correct inter-dataset drift (simplified interface).
"""
function findinter!(dm::AbstractIntraInter,
    cost_fun::String,
    smld_uncorrected::SMLD,
    dataset1::Int,
    d_cutoff::AbstractFloat,
    maxn::Int,
    histbinsize::AbstractFloat)
    refdatasets = Int.(1:smld_uncorrected.n_datasets)
    deleteat!(refdatasets, dataset1)
    return findinter!(dm, cost_fun, smld_uncorrected, dataset1, refdatasets,
                      d_cutoff, maxn, histbinsize)
end
