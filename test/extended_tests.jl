# extended_tests.jl — Local / extended test suite.
#
# Contains the full integration coverage: SMLMSim-driven multi-method comparisons,
# RMSD checks against known drift, multi-iteration runs, auto-ROI integration,
# and the stochastic position_frame_correlation smoke. Some tests are
# noise-sensitive and have proven flaky on Julia 1.10 thread schedules. Run
# locally with:
#
#     julia --project test/extended_tests.jl
#
# For CI / `Pkg.test()`, see test/runtests.jl which runs only the lightweight
# interface + compat + deterministic-unit tests.

using SMLMDriftCorrection
DC = SMLMDriftCorrection
using SMLMSim
using Test
using Random

@testset "SMLMDriftCorrection.jl" begin
    # Use fixed seed for reproducible tests
    Random.seed!(42)

    # Realistic simulation parameters:
    # - Smaller FOV (64x64 = 6.4 μm) for faster tests
    # - Higher k_on (0.02) for ~3-5 blinks per molecule
    # - 3 datasets (enough for inter-dataset testing)
    # - ~1000+ localizations per dataset for good statistics
    params_2d = StaticSMLMConfig(
        10.0,     # density (ρ): emitters per μm² (gives ~400 molecules)
        0.13,     # σ_psf: PSF width in μm (130nm)
        30,       # minphotons: lower threshold to keep more localizations
        3,        # ndatasets: 3 datasets for inter testing
        1000,     # nframes: frames per dataset
        50.0,     # framerate: frames per second
        2,        # ndims: 2D
        [0.0, 1.0]  # zrange: z-range (not used for 2D)
    )
    (smld_noisy, _sim_info) = simulate(
        params_2d;
        pattern=Nmer2D(n=6, d=0.2),
        molecule=GenericFluor(; photons=5000.0, k_on=0.02, k_off=50.0),
        camera=IdealCamera(1:64, 1:64, 0.1)  # 64x64 = 6.4 μm FOV
    )

    # make a 3D Nmer dataset
    params_3d = StaticSMLMConfig(
        10.0,     # density (ρ): emitters per μm²
        0.13,     # σ_psf: PSF width in μm (130nm)
        30,       # minphotons
        3,        # ndatasets
        1000,     # nframes: frames per dataset
        50.0,     # framerate: frames per second
        3,        # ndims: 3D
        [-0.5, 0.5]  # zrange: ±0.5 μm for 3D
    )
    (smld_noisy3, _sim_info3) = simulate(
        params_3d;
        pattern=Nmer3D(n=6, d=0.2),
        molecule=GenericFluor(; photons=5000.0, k_on=0.02, k_off=50.0),
        camera=IdealCamera(1:64, 1:64, 0.1)
    )

    # --- entropy 2D ---
    x = [e.x for e in smld_noisy.emitters]
    y = [e.y for e in smld_noisy.emitters]
    σ_x = [e.σ_x for e in smld_noisy.emitters]
    σ_y = [e.σ_y for e in smld_noisy.emitters]
    N = length(smld_noisy.emitters)
    ent_HD = DC.entropy_HD(σ_x, σ_y)
    ub_ent = DC.ub_entropy(x, y, σ_x, σ_y)
    println("2D: N = $N, entropy_HD = $ent_HD, ub_entropy = $ub_ent")
    @test ent_HD < ub_ent

    # --- entropy 3D ---
    x3 = [e.x for e in smld_noisy3.emitters]
    y3 = [e.y for e in smld_noisy3.emitters]
    z3 = [e.z for e in smld_noisy3.emitters]
    σ_x3 = [e.σ_x for e in smld_noisy3.emitters]
    σ_y3 = [e.σ_y for e in smld_noisy3.emitters]
    σ_z3 = [e.σ_z for e in smld_noisy3.emitters]
    N3 = length(smld_noisy3.emitters)
    ent_HD3 = DC.entropy_HD(σ_x3, σ_y3, σ_z3)
    ub_ent3 = DC.ub_entropy(x3, y3, z3, σ_x3, σ_y3, σ_z3)
    println("3D: N = $N3, entropy_HD = $ent_HD3, ub_entropy = $ub_ent3")
    @test ent_HD3 < ub_ent3

    # --- findshift 2D ---
    println("findshift 2D identity: N = $(length(smld_noisy.emitters))")
    smld_shift = DC.findshift(smld_noisy, smld_noisy; histbinsize=0.10)
    @test isapprox(smld_shift, [0.0, 0.0])

    println("findshift 2D shift: N = $(length(smld_noisy.emitters))")
    shift_imposed = [-4.3, 2.8]
    smldn = deepcopy(smld_noisy)
    for nn = 1:length(smldn.emitters)
        # Apply shift: smldn = smld_noisy + shift_imposed
        smldn.emitters[nn].x += shift_imposed[1]
        smldn.emitters[nn].y += shift_imposed[2]
        smldn.emitters[nn].x = max.(0, min.(smldn.emitters[nn].x, 256))
        smldn.emitters[nn].y = max.(0, min.(smldn.emitters[nn].y, 256))
    end
    # findshift(ref, target) returns the shift of target relative to ref
    smldn_shift = DC.findshift(smld_noisy, smldn; histbinsize=0.10)
    @test isapprox(smldn_shift, shift_imposed, atol = 0.10)

    # --- findshift 3D ---
    println("findshift 3D identity: N3 = $(length(smld_noisy3.emitters))")
    smld_shift3 = DC.findshift(smld_noisy3, smld_noisy3; histbinsize=0.10)
    @test isapprox(smld_shift3, [0.0, 0.0, 0.0])

    println("findshift 3D shift: N3 = $(length(smld_noisy3.emitters))")
    shift_imposed3 = [-4.3, 2.8, 0.2]
    smldn3 = deepcopy(smld_noisy3)
    for nn = 1:length(smldn3.emitters)
        # Apply shift: smldn3 = smld_noisy3 + shift_imposed3
        smldn3.emitters[nn].x += shift_imposed3[1]
        smldn3.emitters[nn].y += shift_imposed3[2]
        smldn3.emitters[nn].z += shift_imposed3[3]
        smldn3.emitters[nn].x = max.(0, min.(smldn3.emitters[nn].x, 256))
        smldn3.emitters[nn].y = max.(0, min.(smldn3.emitters[nn].y, 256))
        smldn3.emitters[nn].z = max.(0, min.(smldn3.emitters[nn].z, 256))
    end
    # findshift(ref, target) returns the shift of target relative to ref
    smldn_shift3 = DC.findshift(smld_noisy3, smldn3; histbinsize=0.10)
    @test isapprox(smldn_shift3, shift_imposed3, atol = 0.10)

    # ========== 2D ==========

    # --- Test correctdrift (LegendrePolynomial) ---
    N = length(smld_noisy.emitters)
    # Create drift model with inter[1] = 0 (DS1 is reference, no global offset)
    Random.seed!(123)
    driftmodel = DC.LegendrePolynomial(smld_noisy; degree=2, initialize="random", rscale=0.1)
    driftmodel.inter[1].dm .= 0.0  # DS1 has no inter shift (reference)
    smld_drift = DC.applydrift(smld_noisy, driftmodel)
    smld_DC = DC.correctdrift(smld_drift, driftmodel)

    smld_noisy_x = [e.x for e in smld_noisy.emitters]
    smld_noisy_y = [e.y for e in smld_noisy.emitters]
    smld_DC_x = [e.x for e in smld_DC.emitters]
    smld_DC_y = [e.y for e in smld_DC.emitters]
    rmsd = sqrt(sum((smld_DC_x .- smld_noisy_x).^2 .+
                    (smld_DC_y .- smld_noisy_y).^2) ./ N)
    print("rmsd 2D [correctdrift] = $rmsd\n")
    @test isapprox(rmsd, 0.0; atol=1e-10)

    # --- Test DriftInfo tuple pattern ---
    @testset "DriftInfo tuple pattern" begin
        (smld_corrected, info) = DC.driftcorrect(smld_drift)
        @test smld_corrected isa DC.SMLD
        @test info isa DC.DriftInfo
        @test info isa DC.AbstractSMLMInfo
        @test info.model isa DC.LegendrePolynomial
        @test info.elapsed_s > 0
        @test info.backend == :cpu
        @test info.iterations >= 1
        @test info.converged == true
        @test info.entropy isa Float64
        @test info.history isa Vector{Float64}
    end

    # --- Test DriftConfig ---
    @testset "DriftConfig" begin
        config = DC.DriftConfig(; quality=:singlepass, degree=2, verbose=0)
        @test config isa DC.AbstractSMLMConfig
        @test config.quality == :singlepass
        @test config.degree == 2
        (smld_cfg, info_cfg) = DC.driftcorrect(smld_drift, config)
        @test smld_cfg isa DC.SMLD
        @test info_cfg isa DC.DriftInfo
        @test info_cfg.converged == true
    end

    # --- Test driftcorrect (default = singlepass) ---
    (smld_corrected, info) = DC.driftcorrect(smld_drift)
    smld_DC_x = [e.x for e in smld_corrected.emitters]
    smld_DC_y = [e.y for e in smld_corrected.emitters]
    rmsd = sqrt(sum((smld_DC_x .- smld_noisy_x).^2 .+
                    (smld_DC_y .- smld_noisy_y).^2) ./ N)
    print("rmsd 2D (singlepass) = $rmsd\n")
    @test rmsd < 0.300  # 300 nm (thread-dependent variance)
    @test info.iterations == 1

    # --- Test quality=:fft ---
    @testset "FFT quality tier" begin
        (smld_fft, info_fft) = DC.driftcorrect(smld_drift; quality=:fft)
        @test info_fft isa DC.DriftInfo
        @test info_fft.iterations == 0
        @test info_fft.converged == true
        @test info_fft.elapsed_s > 0
        # FFT is less accurate but should still be reasonable
        smld_DC_x = [e.x for e in smld_fft.emitters]
        smld_DC_y = [e.y for e in smld_fft.emitters]
        rmsd_fft = sqrt(sum((smld_DC_x .- smld_noisy_x).^2 .+
                            (smld_DC_y .- smld_noisy_y).^2) ./ N)
        print("rmsd 2D (fft) = $rmsd_fft\n")
        # FFT should at least be in the ballpark (< 15 μm)
        # Note: FFT is less accurate than entropy-based methods
        @test rmsd_fft < 0.500  # 500 nm - FFT is less accurate
    end

    # --- Test quality=:iterative ---
    @testset "Iterative quality tier" begin
        (smld_iter, info_iter) = DC.driftcorrect(smld_drift; quality=:iterative, max_iterations=3, verbose=1)
        @test info_iter isa DC.DriftInfo
        @test info_iter.iterations >= 1
        @test length(info_iter.history) >= 1
        @test info_iter.elapsed_s > 0
        smld_DC_x = [e.x for e in smld_iter.emitters]
        smld_DC_y = [e.y for e in smld_iter.emitters]
        rmsd_iter = sqrt(sum((smld_DC_x .- smld_noisy_x).^2 .+
                             (smld_DC_y .- smld_noisy_y).^2) ./ N)
        print("rmsd 2D (iterative) = $rmsd_iter\n")
        @test rmsd_iter < 0.150  # 150 nm (stochastic, allow margin)
    end

    # --- Test warm start ---
    @testset "Warm start" begin
        (smld1, info1) = DC.driftcorrect(smld_drift; quality=:singlepass)
        (smld2, info2) = DC.driftcorrect(smld_drift; warm_start=info1.model)
        @test info2 isa DC.DriftInfo
        @test info2.elapsed_s > 0
        print("Warm start: entropy $(info1.entropy) -> $(info2.entropy)\n")
    end

    # --- Test continuation (dispatch on DriftInfo) ---
    @testset "DriftInfo continuation" begin
        (smld1, info1) = DC.driftcorrect(smld_drift; quality=:singlepass)
        (smld2, info2) = DC.driftcorrect(smld1, info1; max_iterations=2)
        @test info2 isa DC.DriftInfo
        @test info2.iterations > info1.iterations
        print("Continuation: $(info1.iterations) -> $(info2.iterations) iterations\n")
    end

    # --- Test driftcorrect with verbose ---
    (smld_corrected, info) = DC.driftcorrect(smld_drift; maxn=100, verbose=1)
    smld_DC_x = [e.x for e in smld_corrected.emitters]
    smld_DC_y = [e.y for e in smld_corrected.emitters]
    rmsd = sqrt(sum((smld_DC_x .- smld_noisy_x).^2 .+
                    (smld_DC_y .- smld_noisy_y).^2) ./ N)
    print("rmsd 2D (maxn=100) = $rmsd\n")
    @test rmsd < 0.300  # 300 nm (singlepass, thread-dependent RNG variance)

    # --- Test driftcorrect with different degree ---
    # Note: Using degree=3 on degree=2 drift can overfit, so tolerance is relaxed
    (smld_corrected, info) = DC.driftcorrect(smld_drift; degree=3)
    smld_DC_x = [e.x for e in smld_corrected.emitters]
    smld_DC_y = [e.y for e in smld_corrected.emitters]
    rmsd = sqrt(sum((smld_DC_x .- smld_noisy_x).^2 .+
                    (smld_DC_y .- smld_noisy_y).^2) ./ N)
    print("rmsd 2D (degree=3) = $rmsd\n")
    @test rmsd < 1.0  # 1 μm - higher degree can overfit

    # ========== 3D ==========

    # --- Test correctdrift (LegendrePolynomial) ---
    N = length(smld_noisy3.emitters)
    Random.seed!(124)
    driftmodel3 = DC.LegendrePolynomial(smld_noisy3; degree=2, initialize="random", rscale=0.1)
    driftmodel3.inter[1].dm .= 0.0  # DS1 has no inter shift (reference)
    smld_drift3 = DC.applydrift(smld_noisy3, driftmodel3)
    smld_DC = DC.correctdrift(smld_drift3, driftmodel3)

    smld_noisy3_x = [e.x for e in smld_noisy3.emitters]
    smld_noisy3_y = [e.y for e in smld_noisy3.emitters]
    smld_noisy3_z = [e.z for e in smld_noisy3.emitters]
    smld_DC_x = [e.x for e in smld_DC.emitters]
    smld_DC_y = [e.y for e in smld_DC.emitters]
    smld_DC_z = [e.z for e in smld_DC.emitters]
    rmsd = sqrt(sum((smld_DC_x .- smld_noisy3_x).^2 .+
                    (smld_DC_y .- smld_noisy3_y).^2 .+
                    (smld_DC_z .- smld_noisy3_z).^2) ./ N)
    print("rmsd 3D [correctdrift] = $rmsd\n")
    @test isapprox(rmsd, 0.0; atol=1e-10)

    # --- Test driftcorrect (default) ---
    (smld_corrected, info) = DC.driftcorrect(smld_drift3)
    @test info isa DC.DriftInfo
    smld_DC_x = [e.x for e in smld_corrected.emitters]
    smld_DC_y = [e.y for e in smld_corrected.emitters]
    smld_DC_z = [e.z for e in smld_corrected.emitters]
    rmsd = sqrt(sum((smld_DC_x .- smld_noisy3_x).^2 .+
                    (smld_DC_y .- smld_noisy3_y).^2 .+
                    (smld_DC_z .- smld_noisy3_z).^2) ./ N)
    print("rmsd 3D (default) = $rmsd\n")
    @test isapprox(rmsd, 0.0; atol = 0.100)  # 100 nm for 3D

    # --- Test driftcorrect with verbose ---
    (smld_corrected, info) = DC.driftcorrect(smld_drift3; maxn=100, verbose=1)
    smld_DC_x = [e.x for e in smld_corrected.emitters]
    smld_DC_y = [e.y for e in smld_corrected.emitters]
    smld_DC_z = [e.z for e in smld_corrected.emitters]
    rmsd = sqrt(sum((smld_DC_x .- smld_noisy3_x).^2 .+
                    (smld_DC_y .- smld_noisy3_y).^2 .+
                    (smld_DC_z .- smld_noisy3_z).^2) ./ N)
    print("rmsd 3D (maxn=100) = $rmsd\n")
    @test isapprox(rmsd, 0.0; atol = 0.100)  # 100 nm for 3D

    # --- Test 3D quality tiers ---
    @testset "3D quality tiers" begin
        # FFT
        (smld_fft, info_fft) = DC.driftcorrect(smld_drift3; quality=:fft)
        @test info_fft.iterations == 0
        @test info_fft.elapsed_s > 0

        # Iterative
        (smld_iter, info_iter) = DC.driftcorrect(smld_drift3; quality=:iterative, max_iterations=2)
        @test info_iter.iterations >= 1
        @test info_iter.elapsed_s > 0
    end

    # ========== ROI Selection ==========
    @testset "ROI selection functions" begin
        # Test calculate_n_locs_required scaling
        @testset "calculate_n_locs_required" begin
            # Default parameters
            n_req = DC.calculate_n_locs_required(1000)
            @test n_req > 0
            @test n_req isa Int

            # Higher degree needs more data
            n_req_d2 = DC.calculate_n_locs_required(1000; degree=2)
            n_req_d3 = DC.calculate_n_locs_required(1000; degree=3)
            @test n_req_d3 > n_req_d2

            # Tighter target needs more data
            n_req_tight = DC.calculate_n_locs_required(1000; σ_target=0.0005)
            n_req_loose = DC.calculate_n_locs_required(1000; σ_target=0.002)
            @test n_req_tight > n_req_loose

            # More frames with same density needs more locs (lower λ_window)
            n_req_1k = DC.calculate_n_locs_required(1000)
            n_req_5k = DC.calculate_n_locs_required(5000)
            @test n_req_5k > n_req_1k
        end

        # Test find_dense_roi (returns contiguous region with >= n_target locs)
        @testset "find_dense_roi" begin
            n_target = 500
            indices = DC.find_dense_roi(smld_noisy, n_target)
            @test length(indices) >= n_target  # at least n_target
            @test all(1 .<= indices .<= length(smld_noisy.emitters))
            @test length(unique(indices)) == length(indices)  # no duplicates

            # Test edge case: request more than available
            n_total = length(smld_noisy.emitters)
            indices_all = DC.find_dense_roi(smld_noisy, n_total + 100)
            @test length(indices_all) == n_total

            # Test 3D
            indices_3d = DC.find_dense_roi(smld_noisy3, n_target)
            @test length(indices_3d) >= n_target
        end
    end

    # ========== align_smld ==========
    @testset "align_smld" begin
        # --- Type checks ---
        @testset "Type hierarchy" begin
            @test DC.AlignConfig <: DC.AbstractSMLMConfig
            @test DC.AlignInfo <: DC.AbstractSMLMInfo
            @test DC.ShiftTransform <: DC.AbstractAlignTransform
            @test DC.AffineTransform2D <: DC.AbstractAlignTransform
            @test DC.AffineTransform3D <: DC.AbstractAlignTransform
            config = DC.AlignConfig(method=:fft, maxn=50)
            @test config.method == :fft
            @test config.maxn == 50
            @test config.transform == :shift
            config_aff = DC.AlignConfig(transform=:affine)
            @test config_aff.transform == :affine
        end

        # --- Build shifted SMLDs from smld_noisy (2D) ---
        # Use dataset=1 subset as independent "acquisitions"
        smld_base = DC.filter_emitters(smld_noisy,
            [e.dataset == 1 for e in smld_noisy.emitters])

        true_shifts_2d = [[0.0, 0.0], [0.3, -0.2], [-0.15, 0.4]]
        smlds_2d = Vector{typeof(smld_base)}(undef, 3)
        smlds_2d[1] = smld_base
        for k in 2:3
            s = deepcopy(smld_base)
            for nn in eachindex(s.emitters)
                s.emitters[nn].x += true_shifts_2d[k][1]
                s.emitters[nn].y += true_shifts_2d[k][2]
            end
            smlds_2d[k] = s
        end

        # --- Entropy 2D ---
        @testset "Entropy 2D" begin
            (aligned, info) = DC.align_smld(smlds_2d; method=:entropy, verbose=1)
            @test length(aligned) == 3
            @test info isa DC.AlignInfo
            @test info.method == :entropy
            @test info.transform == :shift
            @test info.backend == :cpu
            @test info.elapsed_s > 0
            @test info.shifts[1] == [0.0, 0.0]
            @test info.transforms[1] isa DC.ShiftTransform
            for k in 2:3
                @test isapprox(info.shifts[k], true_shifts_2d[k]; atol=0.050)
                @test info.transforms[k] isa DC.ShiftTransform
            end
        end

        # --- FFT 2D ---
        @testset "FFT 2D" begin
            (aligned, info) = DC.align_smld(smlds_2d; method=:fft)
            @test info.method == :fft
            for k in 2:3
                @test isapprox(info.shifts[k], true_shifts_2d[k]; atol=0.150)
            end
        end

        # --- Build shifted SMLDs from smld_noisy3 (3D) ---
        smld_base3 = DC.filter_emitters(smld_noisy3,
            [e.dataset == 1 for e in smld_noisy3.emitters])

        true_shifts_3d = [[0.0, 0.0, 0.0], [0.2, -0.15, 0.1], [-0.1, 0.25, -0.05]]
        smlds_3d = Vector{typeof(smld_base3)}(undef, 3)
        smlds_3d[1] = smld_base3
        for k in 2:3
            s = deepcopy(smld_base3)
            for nn in eachindex(s.emitters)
                s.emitters[nn].x += true_shifts_3d[k][1]
                s.emitters[nn].y += true_shifts_3d[k][2]
                s.emitters[nn].z += true_shifts_3d[k][3]
            end
            smlds_3d[k] = s
        end

        # --- Entropy 3D ---
        @testset "Entropy 3D" begin
            (aligned, info) = DC.align_smld(smlds_3d; method=:entropy, verbose=1)
            @test info.method == :entropy
            @test info.shifts[1] == [0.0, 0.0, 0.0]
            for k in 2:3
                @test isapprox(info.shifts[k], true_shifts_3d[k]; atol=0.100)
            end
        end

        # --- FFT 3D ---
        @testset "FFT 3D" begin
            (aligned, info) = DC.align_smld(smlds_3d; method=:fft)
            @test info.method == :fft
            for k in 2:3
                @test isapprox(info.shifts[k], true_shifts_3d[k]; atol=0.150)
            end
        end

        # --- Config dispatch ---
        @testset "Config dispatch" begin
            config = DC.AlignConfig(method=:fft, histbinsize=0.05)
            (aligned, info) = DC.align_smld(smlds_2d, config)
            @test info isa DC.AlignInfo
            @test info.method == :fft
        end

        # --- Edge cases ---
        @testset "Edge cases" begin
            # 2 SMLDs works
            (aligned2, info2) = DC.align_smld(smlds_2d[1:2]; method=:fft)
            @test length(aligned2) == 2
            @test length(info2.shifts) == 2

            # 1 SMLD errors
            @test_throws ErrorException DC.align_smld([smlds_2d[1]])

            # Bad method errors
            @test_throws ErrorException DC.align_smld(smlds_2d; method=:bad)

            # Bad transform errors
            @test_throws ErrorException DC.align_smld(smlds_2d; transform=:bad)

        end

        # --- Affine 2D: known rotation + scale + shift (shift-field method) ---
        @testset "Affine 2D" begin
            # Apply known affine: 3° rotation, 1.02 scale, (0.3, -0.2) shift
            true_θ = deg2rad(3.0)
            true_s = 1.02
            true_tx = 0.3
            true_ty = -0.2
            cosθ = cos(true_θ)
            sinθ = sin(true_θ)

            smld_affine = deepcopy(smld_base)
            for nn in eachindex(smld_affine.emitters)
                x = smld_affine.emitters[nn].x
                y = smld_affine.emitters[nn].y
                smld_affine.emitters[nn].x = true_s * (cosθ * x - sinθ * y) + true_tx
                smld_affine.emitters[nn].y = true_s * (sinθ * x + cosθ * y) + true_ty
            end

            smlds_aff = [smld_base, smld_affine]
            (aligned, info) = DC.align_smld(smlds_aff; transform=:affine, verbose=1)

            @test info isa DC.AlignInfo
            @test info.transform == :affine
            @test info.transforms[1] isa DC.AffineTransform2D
            @test info.transforms[2] isa DC.AffineTransform2D

            # Verify correction quality: measure residual shift between aligned data
            # The affine correction should bring the data back close to smld_base
            x_base = [e.x for e in smld_base.emitters]
            y_base = [e.y for e in smld_base.emitters]
            x_aligned = [e.x for e in aligned[2].emitters]
            y_aligned = [e.y for e in aligned[2].emitters]
            rmsd = sqrt(sum((x_aligned .- x_base).^2 .+ (y_aligned .- y_base).^2) / length(x_base))
            @test rmsd < 0.050  # 50nm RMSD tolerance
        end

        # --- Affine 2D: shift-only data recovers near-identity ---
        @testset "Affine 2D identity" begin
            (aligned, info) = DC.align_smld(smlds_2d; transform=:affine)
            @test info.transform == :affine
            # With shift-only data, affine should recover the shifts
            for k in 2:3
                @test isapprox(info.shifts[k], true_shifts_2d[k]; atol=0.15)
            end
        end
    end

    # ========== Auto ROI Integration ==========
    @testset "Auto ROI integration" begin
        # Test that auto_roi=true produces reasonable results
        # Use smaller tolerance since we have more localizations than needed
        (smld_roi, info_roi) = DC.driftcorrect(smld_drift; auto_roi=true, verbose=1)
        @test info_roi isa DC.DriftInfo
        @test info_roi.elapsed_s > 0

        # Compare with auto_roi=false (should be similar accuracy, different speed)
        (smld_no_roi, info_no_roi) = DC.driftcorrect(smld_drift; auto_roi=false)

        # Both should correct drift reasonably well
        smld_roi_x = [e.x for e in smld_roi.emitters]
        smld_roi_y = [e.y for e in smld_roi.emitters]
        smld_noisy_x = [e.x for e in smld_noisy.emitters]
        smld_noisy_y = [e.y for e in smld_noisy.emitters]
        rmsd_roi = sqrt(sum((smld_roi_x .- smld_noisy_x).^2 .+
                            (smld_roi_y .- smld_noisy_y).^2) ./ length(smld_noisy.emitters))
        print("rmsd 2D (auto_roi=true) = $rmsd_roi\n")
        @test rmsd_roi < 0.100  # 100 nm tolerance

        # Test with custom ROI parameters
        (smld_custom, info_custom) = DC.driftcorrect(smld_drift;
            auto_roi=true, σ_loc=0.015, σ_target=0.002, roi_safety_factor=1.5)
        @test info_custom isa DC.DriftInfo

        # Test auto_roi with 3D data
        (smld_roi3, info_roi3) = DC.driftcorrect(smld_drift3; auto_roi=true)
        @test info_roi3 isa DC.DriftInfo
    end

    # ========== Vector-delta rebuild gating ==========
    # Scalar max_drift_magnitude would miss a same-magnitude direction flip
    # (e.g. +0.05 μm at the midpoint rotating to -0.05 μm). The vector-based
    # gate must fire on that change.
    @testset "Rebuild gating on same-magnitude direction change" begin
        # Two drift-vector matrices with identical column magnitudes but
        # flipped directions at the middle test frame. Delta norm is 0.11 μm
        # at that column, exceeding the 0.1 μm threshold.
        last    = [ 0.0  0.05  0.0;
                    0.0  0.00  0.0]
        current = [ 0.0 -0.06  0.0;
                    0.0  0.00  0.0]
        Δ = DC.max_drift_vec_delta(current, last)
        @test Δ ≈ 0.11 atol=1e-12
        @test Δ > 0.1  # triggers rebuild at default threshold

        # Identical vectors → zero delta, no rebuild
        @test DC.max_drift_vec_delta(last, last) == 0.0

        # Translation (same direction, larger magnitude) still detected
        current2 = [ 0.0  0.20  0.0;
                     0.0  0.00  0.0]
        @test DC.max_drift_vec_delta(current2, last) ≈ 0.15 atol=1e-12

        # NeighborState constructor: last_drift_vecs initialised to +Inf so
        # the first evaluation's delta is +Inf and forces a rebuild regardless
        # of how small the actual drift is.
        st = DC.NeighborState(10, 5, 0.1, 2)
        cur = [0.0 0.0 0.0; 0.0 0.0 0.0]
        @test DC.max_drift_vec_delta(cur, st.last_drift_vecs) == Inf
    end

    # ========== Edge-case guards ==========
    @testset "Edge-case guards" begin
        # normalize_frame with n_frames=1 should not throw DivideError / DomainError
        @test DC.normalize_frame(1, 1) == 0.0
        # Legendre polynomial evaluation at a single-frame collapse is finite
        poly_1f = DC.LegendrePoly1D(2, 1)
        poly_1f.coefficients .= [0.5, 0.25]
        @test isfinite(DC.evaluate_at_frame(poly_1f, 1))

        # compute_chunk_params: n_chunks > n_frames should be capped, not 0-sized
        p1 = DC.compute_chunk_params(10; n_chunks=20)
        @test p1.n_chunks <= 10
        @test p1.frames_per_chunk >= 1

        # compute_chunk_params: chunk_frames > n_frames should collapse to 1 chunk
        p2 = DC.compute_chunk_params(10; chunk_frames=100)
        @test p2.n_chunks == 1
        @test p2.frames_per_chunk == 10

        # compute_chunk_params: single-frame input should not divide by zero
        p3 = DC.compute_chunk_params(1)
        @test p3.n_chunks == 1
        @test p3.frames_per_chunk >= 1

        # findshift_damped: σ_px floor prevents collapse when prior_sigma ≈ 0
        smld_small = DC.filter_emitters(smld_noisy,
            [e.dataset == 1 for e in smld_noisy.emitters])
        shift_zero_sigma = DC.findshift_damped(smld_small, smld_small;
            histbinsize=0.10, prior_shift=[0.0, 0.0], prior_sigma=0.0)
        @test all(isfinite, shift_zero_sigma)

        # findshift_damped: known shift + non-zero prior recovers approximately
        shift_imposed_damped = [0.5, -0.3]
        smld_shifted_d = deepcopy(smld_small)
        for nn in eachindex(smld_shifted_d.emitters)
            smld_shifted_d.emitters[nn].x += shift_imposed_damped[1]
            smld_shifted_d.emitters[nn].y += shift_imposed_damped[2]
        end
        # Prior centered near the true shift → damping does not reject the true peak
        shift_damped = DC.findshift_damped(smld_small, smld_shifted_d;
            histbinsize=0.10, prior_shift=shift_imposed_damped, prior_sigma=1.0)
        @test isapprox(shift_damped, shift_imposed_damped; atol=0.15)

        # filter_emitters to an empty mask + driftcorrect-style helpers must not crash
        empty_mask = falses(length(smld_small.emitters))
        smld_empty = DC.filter_emitters(smld_small, empty_mask)
        @test isempty(smld_empty.emitters)

        # findintra! on a dataset with zero emitters is a no-op
        model_check = DC.LegendrePolynomial(smld_small; degree=2)
        # Build a model scoped to a bogus dataset index so the per-dataset filter is empty
        @test_nowarn DC.findintra!(model_check.intra[1], smld_empty, 1, 50)
    end

    # ========== position_frame_correlation ==========
    # Smoke test for the diagnostic function. Wrapped in isdefined so this
    # testset is skipped gracefully if the function hasn't been merged yet.
    if isdefined(SMLMDriftCorrection, :position_frame_correlation)
        @testset "position_frame_correlation" begin
            # Build a small synthetic SMLD specifically for this test
            # (3 datasets, 500 frames, density ~10) to keep things fast.
            Random.seed!(7)
            params_pfc = StaticSMLMConfig(
                30.0,    # density (higher for enough locs)
                0.13,    # σ_psf
                30,      # minphotons
                2,       # ndatasets
                2000,    # nframes
                50.0,    # framerate
                2,       # ndims (2D)
                [0.0, 1.0]
            )
            (smld_pfc, _) = simulate(
                params_pfc;
                pattern=Nmer2D(n=6, d=0.2),
                molecule=GenericFluor(; photons=5000.0, k_on=0.05, k_off=50.0),
                camera=IdealCamera(1:64, 1:64, 0.1)
            )

            # Apply a known random drift large enough to dominate noise
            Random.seed!(321)
            drift_pfc = DC.LegendrePolynomial(smld_pfc; degree=2,
                                              initialize="random", rscale=0.3)
            drift_pfc.inter[1].dm .= 0.0
            smld_drifted_pfc = DC.applydrift(smld_pfc, drift_pfc)

            # Drift-correct
            (smld_corr_pfc, _info_pfc) = DC.driftcorrect(smld_drifted_pfc)

            # --- Intra mode ---
            res_drifted = SMLMDriftCorrection.position_frame_correlation(
                smld_drifted_pfc; K=20, mode=:intra)
            res_corrected = SMLMDriftCorrection.position_frame_correlation(
                smld_corr_pfc; K=20, mode=:intra)

            # Shape checks
            @test res_drifted.mode == :intra
            @test res_corrected.mode == :intra
            @test res_drifted.K == 20
            @test length(res_drifted.per_dataset) == 2
            @test length(res_corrected.per_dataset) == 2

            for entry in res_drifted.per_dataset
                @test 0.0 <= abs(entry.corr_x) <= 1.0
                @test 0.0 <= abs(entry.corr_y) <= 1.0
                @test entry.corr_z === nothing  # 2D
                @test entry.residuals_z === nothing
                @test length(entry.residuals_x) == entry.n_locs
                @test length(entry.residuals_y) == entry.n_locs
                @test length(entry.frames) == entry.n_locs
            end

            # Summary checks
            @test 0.0 <= res_drifted.summary.mean_abs_corr_x <= 1.0
            @test 0.0 <= res_drifted.summary.mean_abs_corr_y <= 1.0
            @test res_drifted.summary.mean_abs_corr_z === nothing
            @test 0.0 <= res_corrected.summary.mean_abs_corr_x <= 1.0
            @test 0.0 <= res_corrected.summary.mean_abs_corr_y <= 1.0

            # Hypothesis: drifted data should show stronger combined
            # position-frame correlation than corrected data. Check the sum
            # of |corr_x| and |corr_y| to avoid per-axis noise flipping the sign.
            drifted_total = res_drifted.summary.mean_abs_corr_x + res_drifted.summary.mean_abs_corr_y
            corrected_total = res_corrected.summary.mean_abs_corr_x + res_corrected.summary.mean_abs_corr_y
            print("position_frame_correlation: |corr_x|+|corr_y| drifted=$drifted_total corrected=$corrected_total\n")
            @test drifted_total > corrected_total

            # --- Inter mode ---
            res_inter = SMLMDriftCorrection.position_frame_correlation(
                smld_drifted_pfc; K=20, mode=:inter)
            @test res_inter.mode == :inter
            @test res_inter.K == 20
            @test 0.0 <= abs(res_inter.corr_x) <= 1.0
            @test 0.0 <= abs(res_inter.corr_y) <= 1.0
            @test res_inter.corr_z === nothing
            @test res_inter.residuals_z === nothing
            @test length(res_inter.residuals_x) == length(res_inter.dataset_indices)
            @test length(res_inter.residuals_y) == length(res_inter.dataset_indices)
        end
    end
end
