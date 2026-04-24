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
- `info.transforms[i]`: diagnostic transform (ShiftTransform for `:shift`; a similarity
  fit `AffineTransform2D`/`AffineTransform3D` for `:affine`).
- `info.diagnostic`: `nothing` for `:shift`; for `:affine` a NamedTuple with
  `affine_coeffs[i] = (a, b, c, d, e, f)` and `global_shifts[i]` — the **exact**
  per-dataset correction applied (summed across internal passes). The similarity
  stored in `info.transforms[i]` is only a best-fit approximation of this general
  affine.

# Example
```julia
# Shift-only (default)
(aligned, info) = align_smld(smlds)
info.shifts  # [zeros(2), [dx2, dy2], [dx3, dy3], ...]

# Affine (rotation + scale + translation)
(aligned, info) = align_smld(smlds; transform=:affine)
info.transforms[2]        # AffineTransform2D(θ, scale, tx, ty) — similarity approximation
info.diagnostic.affine_coeffs[2]  # (a, b, c, d, e, f) — actual correction applied
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

    if config.transform == :shift
        return _align_shift(smlds, N, ndims_ref, config, t0)
    else  # :affine — always uses shift-field FFT method
        return _align_affine_fft(smlds, N, ndims_ref, config, t0)
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
    info = AlignInfo(shifts, transforms, elapsed, config.method, config.transform, :cpu, nothing)
    return (aligned, info)
end

# ============================================================================
# FFT affine: shift-field method (sub-region CC + affine fit)
# ============================================================================

function _align_affine_fft(smlds, N, ndims_ref, config, t0)
    ndims_ref == 2 || error("align_smld: transform=:affine with method=:fft only supports 2D")

    shifts = Vector{Vector{Float64}}(undef, N)
    shifts[1] = zeros(ndims_ref)
    transforms = Vector{AbstractAlignTransform}(undef, N)
    transforms[1] = AffineTransform2D(0.0, 1.0, 0.0, 0.0)
    aligned = Vector{typeof(smlds[1])}(undef, N)
    aligned[1] = smlds[1]

    # Record the exact affine correction actually applied per dataset, so callers
    # can recover the full general affine (transforms[i] stores only a similarity fit).
    affine_coeffs = Vector{NTuple{6, Float64}}(undef, N)
    global_shifts = Vector{Vector{Float64}}(undef, N)
    affine_coeffs[1] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    global_shifts[1] = zeros(ndims_ref)

    for i in 2:N
        # Pass 1: global shift + affine from shift field
        result = find_affine_shift_field(smlds[1], smlds[i];
                    histbinsize=config.histbinsize, min_locs=200)
        a, b, c, d, e, f = result.a, result.b, result.c, result.d, result.e, result.f

        aligned[i] = deepcopy(smlds[i])
        correctdrift!(aligned[i], result.global_shift)
        for nn in eachindex(aligned[i].emitters)
            x = aligned[i].emitters[nn].x; y = aligned[i].emitters[nn].y
            aligned[i].emitters[nn].x = x - (a*x + b*y + c)
            aligned[i].emitters[nn].y = y - (d*x + e*y + f)
        end

        if config.verbose >= 1
            θ1 = atan((d - b) / 2)
            println("  smlds[$i] pass1: θ=$(round(rad2deg(θ1); digits=3))°, sx=$(round(1+a; digits=5)), sy=$(round(1+e; digits=5)), tiles=$(result.n_tiles), rms=$(round(result.rms_residual*1000; digits=1))nm")
        end

        # Pass 2: refine on corrected data (residual should be small)
        result2 = find_affine_shift_field(smlds[1], aligned[i];
                    histbinsize=config.histbinsize, min_locs=200)
        a2, b2, c2, d2, e2, f2 = result2.a, result2.b, result2.c, result2.d, result2.e, result2.f

        # Apply global shift residual + affine residual
        correctdrift!(aligned[i], result2.global_shift)
        for nn in eachindex(aligned[i].emitters)
            x = aligned[i].emitters[nn].x; y = aligned[i].emitters[nn].y
            aligned[i].emitters[nn].x = x - (a2*x + b2*y + c2)
            aligned[i].emitters[nn].y = y - (d2*x + e2*y + f2)
        end

        if config.verbose >= 1
            println("  smlds[$i] pass2: rms=$(round(result2.rms_residual*1000; digits=1))nm")
        end

        # Record the exact correction applied (sum of two passes)
        total_a = a + a2; total_b = b + b2; total_c = c + c2
        total_d = d + d2; total_e = e + e2; total_f = f + f2
        total_global = result.global_shift .+ result2.global_shift
        affine_coeffs[i] = (total_a, total_b, total_c, total_d, total_e, total_f)
        global_shifts[i] = collect(total_global)

        # Total shift at centroid (diagnostic similarity fit — independent x/y scale
        # and shear from the general affine are not captured here).
        cx = mean(em.x for em in smlds[i].emitters)
        cy = mean(em.y for em in smlds[i].emitters)
        shifts[i] = [total_global[1] + a*cx + b*cy + c, total_global[2] + d*cx + e*cy + f]

        θ_fit = atan((total_d - total_b) / 2)
        s_fit = sqrt(abs((1 + total_a) * (1 + total_e)))
        transforms[i] = AffineTransform2D(θ_fit, s_fit, shifts[i][1], shifts[i][2])
    end

    elapsed = time() - t0
    config.verbose >= 1 && println("align_smld: aligned $N SMLDs (fft, affine) in $(round(elapsed; digits=2))s")

    diag = (affine_coeffs = affine_coeffs, global_shifts = global_shifts)
    info = AlignInfo(shifts, transforms, elapsed, config.method, config.transform, :cpu, diag)
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
        # Deterministic objective for BFGS: freeze state during optimize, drive
        # KDTree rebuilds from an outer loop (mirrors findinter!).
        state.allow_rebuild = true
        entropy_cost(θ_cc)
        state.allow_rebuild = false

        θ0 = copy(θ_cc)
        local res
        for _ in 1:3
            res = optimize(myfun, θ0, BFGS(), opt)
            θ0 = Float64.(res.minimizer)
            Δ2 = 0.0
            for dim in 1:ndims_ref
                Δ2 += (θ0[dim] - state.last_shift[dim])^2
            end
            sqrt(Δ2) < state.rebuild_threshold && break
            state.allow_rebuild = true
            entropy_cost(θ0)
            state.allow_rebuild = false
        end
        shifts[i] = Float64.(res.minimizer)
    end
end

