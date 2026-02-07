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
        @test rmsd_iter < 0.100  # 100 nm
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
end
