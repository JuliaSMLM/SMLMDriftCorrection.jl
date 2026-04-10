# Alignment of independent SMLDs (shift or affine transform)

"""
    align_smld(smlds; kwargs...) -> (Vector{<:SMLD}, AlignInfo)
    align_smld(smlds, config::AlignConfig) -> (Vector{<:SMLD}, AlignInfo)

Align a vector of independent SMLD structures to a common reference (the first).
Each SMLD is assumed to be an independent acquisition of the same FOV/structure.

# Arguments
- `smlds`: Vector of SMLD structures (>= 2 required)

# Keyword Arguments
- `method=:entropy`: `:entropy` (CC initial guess + entropy refinement) or `:fft` (CC only)
- `transform=:shift`: `:shift` (translation only) or `:affine` (rotation + scale + translation)
- `maxn=100`: Maximum neighbors for entropy calculation
- `histbinsize=0.05`: Histogram bin size (μm) for cross-correlation
- `verbose=0`: Verbosity: 0=quiet, 1=info

# Returns
Tuple `(aligned_smlds, info::AlignInfo)` where:
- `aligned_smlds`: Vector of aligned SMLDs (first is unchanged)
- `info.shifts[i]`: translation component for `smlds[i]` (`shifts[1]` = zeros)
- `info.transforms[i]`: full transform (ShiftTransform or AffineTransform2D/3D)

# Example
```julia
# Shift-only (default)
(aligned, info) = align_smld(smlds)
info.shifts  # [zeros(2), [dx2, dy2], [dx3, dy3], ...]

# Affine (rotation + scale + translation)
(aligned, info) = align_smld(smlds; transform=:affine)
info.transforms[2]  # AffineTransform2D(θ, scale, tx, ty)
```
"""
function align_smld(smlds::Vector{<:SMLD}; kwargs...)
    config = AlignConfig(; kwargs...)
    return align_smld(smlds, config)
end

function align_smld(smlds::Vector{<:SMLD}, config::AlignConfig)
    t0 = time()

    N = length(smlds)
    N >= 2 || error("align_smld requires at least 2 SMLDs, got $N")

    # Validate all have emitters
    for i in 1:N
        isempty(smlds[i].emitters) && error("align_smld: smlds[$i] has no emitters")
    end

    # Validate consistent dimensionality
    ndims_ref = nDims(smlds[1])
    for i in 2:N
        nDims(smlds[i]) == ndims_ref || error("align_smld: inconsistent dimensionality between smlds[1] ($ndims_ref D) and smlds[$i] ($(nDims(smlds[i])) D)")
    end

    config.method in (:entropy, :fft) || error("align_smld: unknown method $(config.method), expected :entropy or :fft")
    config.transform in (:shift, :affine) || error("align_smld: unknown transform $(config.transform), expected :shift or :affine")
    config.transform == :affine && config.method == :fft && error("align_smld: transform=:affine requires method=:entropy (FFT can only recover translation)")

    if config.transform == :shift
        return _align_shift(smlds, N, ndims_ref, config, t0)
    else
        return _align_affine(smlds, N, ndims_ref, config, t0)
    end
end

# ============================================================================
# Shift-only alignment (original behavior)
# ============================================================================

function _align_shift(smlds, N, ndims_ref, config, t0)
    shifts = Vector{Vector{Float64}}(undef, N)
    shifts[1] = zeros(ndims_ref)

    if config.method == :fft
        _align_fft!(shifts, smlds, ndims_ref, config)
    else
        _align_entropy!(shifts, smlds, ndims_ref, config)
    end

    # Apply shifts
    aligned = Vector{typeof(smlds[1])}(undef, N)
    aligned[1] = smlds[1]
    for i in 2:N
        aligned[i] = deepcopy(smlds[i])
        correctdrift!(aligned[i], shifts[i])
    end

    elapsed = time() - t0
    if config.verbose >= 1
        println("align_smld: aligned $N SMLDs ($(config.method), shift) in $(round(elapsed; digits=2))s")
        for i in 2:N
            println("  smlds[$i]: shift = $(round.(shifts[i]; digits=4))")
        end
    end

    transforms = AbstractAlignTransform[ShiftTransform(s) for s in shifts]
    info = AlignInfo(shifts, transforms, elapsed, config.method, config.transform, :cpu)
    return (aligned, info)
end

# ============================================================================
# Affine alignment (rotation + scale + translation)
# ============================================================================

function _align_affine(smlds, N, ndims_ref, config, t0)
    shifts = Vector{Vector{Float64}}(undef, N)
    shifts[1] = zeros(ndims_ref)
    transforms = Vector{AbstractAlignTransform}(undef, N)
    if ndims_ref == 2
        transforms[1] = AffineTransform2D(0.0, 1.0, 0.0, 0.0)
    else
        transforms[1] = AffineTransform3D(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    end

    _align_entropy_affine!(shifts, transforms, smlds, ndims_ref, config)

    # Apply transforms
    aligned = Vector{typeof(smlds[1])}(undef, N)
    aligned[1] = smlds[1]
    for i in 2:N
        aligned[i] = deepcopy(smlds[i])
        apply_affine_transform!(aligned[i], transforms[i])
    end

    elapsed = time() - t0
    if config.verbose >= 1
        println("align_smld: aligned $N SMLDs (entropy, affine) in $(round(elapsed; digits=2))s")
        for i in 2:N
            t = transforms[i]
            if t isa AffineTransform2D
                println("  smlds[$i]: θ=$(round(rad2deg(t.θ); digits=3))°, s=$(round(t.scale; digits=5)), shift=$(round.([t.tx, t.ty]; digits=4))")
            else
                t3 = t::AffineTransform3D
                println("  smlds[$i]: θ=$(round.(rad2deg.([t3.θx, t3.θy, t3.θz]); digits=3))°, s=$(round(t3.scale; digits=5)), shift=$(round.([t3.tx, t3.ty, t3.tz]; digits=4))")
            end
        end
    end

    info = AlignInfo(shifts, transforms, elapsed, config.method, config.transform, :cpu)
    return (aligned, info)
end

# --- FFT method: cross-correlation only ---

function _align_fft!(shifts, smlds, ndims_ref, config)
    N = length(smlds)
    Threads.@threads for i in 2:N
        shifts[i] = findshift(smlds[1], smlds[i]; histbinsize=config.histbinsize)
    end
end

# --- Entropy method (shift): CC initial guess + regularized entropy refinement ---

function _align_entropy!(shifts, smlds, ndims_ref, config)
    N = length(smlds)

    # Extract reference coords once (shared across threads, read-only)
    ref_emitters = smlds[1].emitters
    x_ref = Float64[e.x for e in ref_emitters]
    y_ref = Float64[e.y for e in ref_emitters]
    σ_x_ref = Float64[e.σ_x for e in ref_emitters]
    σ_y_ref = Float64[e.σ_y for e in ref_emitters]
    if ndims_ref == 3
        z_ref = Float64[e.z for e in ref_emitters]
        σ_z_ref = Float64[e.σ_z for e in ref_emitters]
    end
    N_ref = length(ref_emitters)

    Threads.@threads for i in 2:N
        # Extract target coords
        emitters_i = smlds[i].emitters
        x_i = Float64[e.x for e in emitters_i]
        y_i = Float64[e.y for e in emitters_i]
        σ_x_i = Float64[e.σ_x for e in emitters_i]
        σ_y_i = Float64[e.σ_y for e in emitters_i]
        N_i = length(emitters_i)

        if ndims_ref == 3
            z_i = Float64[e.z for e in emitters_i]
            σ_z_i = Float64[e.σ_z for e in emitters_i]
        end

        # CC initial guess
        θ_cc = zeros(Float64, ndims_ref)
        try
            cc_result = Float64.(findshift(smlds[1], smlds[i]; histbinsize=config.histbinsize))
            if maximum(abs.(cc_result)) < 5.0
                θ_cc = cc_result
            end
        catch
            # Keep zero initialization
        end

        # Build entropy cost function (same pattern as findinter!)
        inter = InterShift(ndims_ref)
        x_work = similar(x_i)
        y_work = similar(y_i)
        k = min(config.maxn, N_i + N_ref - 1)
        rebuild_threshold = 0.1  # 100nm

        if ndims_ref == 2
            data_combined = Matrix{Float64}(undef, 2, N_i + N_ref)
            state = InterNeighborState(N_i, k, rebuild_threshold)
            entropy_cost = θ -> costfun_entropy_inter_2D_merged(θ,
                x_i, y_i, σ_x_i, σ_y_i,
                x_ref, y_ref, σ_x_ref, σ_y_ref,
                config.maxn, inter;
                x_work=x_work, y_work=y_work,
                data_combined=data_combined, state=state)
        else  # 3D
            z_work = similar(z_i)
            data_combined = Matrix{Float64}(undef, 3, N_i + N_ref)
            state = InterNeighborState3D(N_i, k, rebuild_threshold)
            entropy_cost = θ -> costfun_entropy_inter_3D_merged(θ,
                x_i, y_i, z_i, σ_x_i, σ_y_i, σ_z_i,
                x_ref, y_ref, z_ref, σ_x_ref, σ_y_ref, σ_z_ref,
                config.maxn, inter;
                x_work=x_work, y_work=y_work, z_work=z_work,
                data_combined=data_combined, state=state)
        end

        # Regularize toward CC result: cost = entropy + λ*||θ - θ_cc||²
        # Allows entropy to refine the CC shift but prevents wandering to
        # spurious local minima. Same pattern as findinter! regularization.
        # λ chosen so that a 0.5 μm deviation from CC adds ~0.25 to cost,
        # which is significant relative to typical entropy differences (~0.1-1.0).
        λ = 1.0
        myfun = θ -> entropy_cost(θ) + λ * sum((θ .- θ_cc).^2)

        opt = Optim.Options(iterations=10000, g_abstol=1e-8, show_trace=false)
        res = optimize(myfun, θ_cc, BFGS(), opt)
        shifts[i] = Float64.(res.minimizer)
    end
end

# --- Entropy method (affine): CC initial guess for translation + entropy for all params ---

function _align_entropy_affine!(shifts, transforms, smlds, ndims_ref, config)
    N = length(smlds)

    # Extract reference coords once (shared across threads, read-only)
    ref_emitters = smlds[1].emitters
    x_ref = Float64[e.x for e in ref_emitters]
    y_ref = Float64[e.y for e in ref_emitters]
    σ_x_ref = Float64[e.σ_x for e in ref_emitters]
    σ_y_ref = Float64[e.σ_y for e in ref_emitters]
    if ndims_ref == 3
        z_ref = Float64[e.z for e in ref_emitters]
        σ_z_ref = Float64[e.σ_z for e in ref_emitters]
    end
    N_ref = length(ref_emitters)

    Threads.@threads for i in 2:N
        # Extract target coords
        emitters_i = smlds[i].emitters
        x_i = Float64[e.x for e in emitters_i]
        y_i = Float64[e.y for e in emitters_i]
        σ_x_i = Float64[e.σ_x for e in emitters_i]
        σ_y_i = Float64[e.σ_y for e in emitters_i]
        N_i = length(emitters_i)

        if ndims_ref == 3
            z_i = Float64[e.z for e in emitters_i]
            σ_z_i = Float64[e.σ_z for e in emitters_i]
        end

        # CC initial guess for translation component
        θ_cc_shift = zeros(Float64, ndims_ref)
        try
            cc_result = Float64.(findshift(smlds[1], smlds[i]; histbinsize=config.histbinsize))
            if maximum(abs.(cc_result)) < 5.0
                θ_cc_shift = cc_result
            end
        catch
            # Keep zero initialization
        end

        # Build affine parameter vector: identity transform + CC translation
        x_work = similar(x_i)
        y_work = similar(y_i)
        k = min(config.maxn, N_i + N_ref - 1)
        rebuild_threshold = 0.1  # 100nm, same as shift case

        # Estimate FOV radius for rebuild threshold scaling
        fov_radius = max(maximum(x_i) - minimum(x_i), maximum(y_i) - minimum(y_i)) / 2

        if ndims_ref == 2
            # θ = [θ_rot, scale, tx, ty]
            θ0 = Float64[0.0, 1.0, θ_cc_shift[1], θ_cc_shift[2]]
            last_params = fill(Inf, 4)  # force initial rebuild

            data_combined = Matrix{Float64}(undef, 2, N_i + N_ref)
            state = InterNeighborState(N_i, k, rebuild_threshold)
            entropy_cost = θ -> costfun_entropy_affine_2D_merged(θ,
                x_i, y_i, σ_x_i, σ_y_i,
                x_ref, y_ref, σ_x_ref, σ_y_ref,
                config.maxn;
                x_work=x_work, y_work=y_work,
                data_combined=data_combined, state=state,
                last_params=last_params, fov_radius=fov_radius)

            # Mild regularization toward identity for rotation/scale.
            # Translation is NOT regularized toward CC — CC shift is contaminated
            # by rotation when the data is rotated (CC sees centroid displacement
            # = t + (sR - I)*centroid, not just t).
            λ_rot = 1.0
            λ_scale = 1.0
            myfun = θ -> begin
                cost = entropy_cost(θ)
                cost += λ_rot * θ[1]^2                    # rotation toward 0
                cost += λ_scale * (θ[2] - 1.0)^2          # scale toward 1
                return cost
            end

            opt = Optim.Options(iterations=10000, g_abstol=1e-8, show_trace=false)
            res = optimize(myfun, θ0, BFGS(), opt)
            θ_opt = Float64.(res.minimizer)

            shifts[i] = [θ_opt[3], θ_opt[4]]
            transforms[i] = AffineTransform2D(θ_opt[1], θ_opt[2], θ_opt[3], θ_opt[4])
        else  # 3D
            # θ = [θx, θy, θz, scale, tx, ty, tz]
            θ0 = Float64[0.0, 0.0, 0.0, 1.0, θ_cc_shift[1], θ_cc_shift[2], θ_cc_shift[3]]

            z_work = similar(z_i)
            last_params = fill(Inf, 7)  # force initial rebuild

            data_combined = Matrix{Float64}(undef, 3, N_i + N_ref)
            state = InterNeighborState3D(N_i, k, rebuild_threshold)
            entropy_cost = θ -> costfun_entropy_affine_3D_merged(θ,
                x_i, y_i, z_i, σ_x_i, σ_y_i, σ_z_i,
                x_ref, y_ref, z_ref, σ_x_ref, σ_y_ref, σ_z_ref,
                config.maxn;
                x_work=x_work, y_work=y_work, z_work=z_work,
                data_combined=data_combined, state=state,
                last_params=last_params, fov_radius=fov_radius)

            λ_rot = 1.0
            λ_scale = 1.0
            myfun = θ -> begin
                cost = entropy_cost(θ)
                cost += λ_rot * (θ[1]^2 + θ[2]^2 + θ[3]^2)              # rotation toward 0
                cost += λ_scale * (θ[4] - 1.0)^2                         # scale toward 1
                return cost
            end

            opt = Optim.Options(iterations=10000, g_abstol=1e-8, show_trace=false)
            res = optimize(myfun, θ0, BFGS(), opt)
            θ_opt = Float64.(res.minimizer)

            shifts[i] = [θ_opt[5], θ_opt[6], θ_opt[7]]
            transforms[i] = AffineTransform3D(θ_opt[1], θ_opt[2], θ_opt[3], θ_opt[4], θ_opt[5], θ_opt[6], θ_opt[7])
        end
    end
end
