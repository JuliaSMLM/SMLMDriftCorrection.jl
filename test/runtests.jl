using SMLMDriftCorrection
DC = SMLMDriftCorrection
using SMLMSim
using Test

@testset "SMLMDriftCorrection.jl" begin
    # Simulation parameters use physical units
    # smld structures are in units of pixels and frames
    params_2d = StaticSMLMParams(
        2.0,      # density (ρ): emitters per μm²
        0.13,     # σ_psf: PSF width in μm (130nm)
        50,       # minphotons: minimum photons for detection
        10,       # ndatasets: number of independent datasets
        1000,     # nframes: frames per dataset
        50.0,     # framerate: frames per second
        2,        # ndims: 2D
        [0.0, 1.0]  # zrange: z-range (not used for 2D)
    )
    smld_true, smld_model, smld_noisy = simulate(
        params_2d;
        pattern=Nmer2D(n=6, d=0.2),
        molecule=GenericFluor(; photons=5000.0, k_on=0.001, k_off=50.0),
        camera=IdealCamera(1:256, 1:256, 0.1)
    )

    # make a 3D Nmer dataset
    params_3d = StaticSMLMParams(
        2.0,      # density (ρ): emitters per μm²
        0.13,     # σ_psf: PSF width in μm (130nm)
        50,       # minphotons: minimum photons for detection
        10,       # ndatasets: number of independent datasets
        1000,     # nframes: frames per dataset
        50.0,     # framerate: frames per second
        3,        # ndims: 3D
        [-1.0, 1.0]  # zrange: z-range for 3D
    )
    smld_true3, smld_model3, smld_noisy3 = simulate(
        params_3d;
        pattern=Nmer3D(n=6, d=0.2),
        molecule=GenericFluor(; photons=5000.0, k_on=0.001, k_off=50.0),
        camera=IdealCamera(1:256, 1:256, 0.1)
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
        smldn.emitters[nn].x -= shift_imposed[1]
        smldn.emitters[nn].y -= shift_imposed[2]
        smldn.emitters[nn].x = max.(0, min.(smldn.emitters[nn].x, 256))
        smldn.emitters[nn].y = max.(0, min.(smldn.emitters[nn].y, 256))
    end
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
        smldn3.emitters[nn].x -= shift_imposed3[1]
        smldn3.emitters[nn].y -= shift_imposed3[2]
        smldn3.emitters[nn].z -= shift_imposed3[3]
        smldn3.emitters[nn].x = max.(0, min.(smldn3.emitters[nn].x, 256))
        smldn3.emitters[nn].y = max.(0, min.(smldn3.emitters[nn].y, 256))
        smldn3.emitters[nn].z = max.(0, min.(smldn3.emitters[nn].z, 256))
    end
    smldn_shift3 = DC.findshift(smld_noisy3, smldn3; histbinsize=0.10)
    @test isapprox(smldn_shift3, shift_imposed3, atol = 0.10)

    # ========== 2D ==========

    # --- Test correctdrift (LegendrePolynomial) ---
    N = length(smld_noisy.emitters)
    driftmodel = DC.LegendrePolynomial(smld_noisy; degree=2, initialize="random")
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
        @test info.model isa DC.LegendrePolynomial
        @test info.elapsed_ns > 0
        @test info.backend == :cpu
        @test info.iterations >= 1
        @test info.converged == true
        @test info.entropy isa Float64
        @test info.history isa Vector{Float64}
    end

    # --- Test driftcorrect (default = singlepass) ---
    (smld_corrected, info) = DC.driftcorrect(smld_drift)
    smld_DC_x = [e.x for e in smld_corrected.emitters]
    smld_DC_y = [e.y for e in smld_corrected.emitters]
    rmsd = sqrt(sum((smld_DC_x .- smld_noisy_x).^2 .+
                    (smld_DC_y .- smld_noisy_y).^2) ./ N)
    print("rmsd 2D (singlepass) = $rmsd\n")
    @test isapprox(rmsd, 0.0; atol = 5.0)
    @test info.iterations == 1

    # --- Test quality=:fft ---
    @testset "FFT quality tier" begin
        (smld_fft, info_fft) = DC.driftcorrect(smld_drift; quality=:fft)
        @test info_fft isa DC.DriftInfo
        @test info_fft.iterations == 0
        @test info_fft.converged == true
        @test info_fft.elapsed_ns > 0
        # FFT is less accurate but should still be reasonable
        smld_DC_x = [e.x for e in smld_fft.emitters]
        smld_DC_y = [e.y for e in smld_fft.emitters]
        rmsd_fft = sqrt(sum((smld_DC_x .- smld_noisy_x).^2 .+
                            (smld_DC_y .- smld_noisy_y).^2) ./ N)
        print("rmsd 2D (fft) = $rmsd_fft\n")
        # FFT should at least be in the ballpark (< 15 μm)
        # Note: FFT is less accurate than entropy-based methods
        @test rmsd_fft < 15.0
    end

    # --- Test quality=:iterative ---
    @testset "Iterative quality tier" begin
        (smld_iter, info_iter) = DC.driftcorrect(smld_drift; quality=:iterative, max_iterations=3, verbose=1)
        @test info_iter isa DC.DriftInfo
        @test info_iter.iterations >= 1
        @test length(info_iter.history) >= 1
        @test info_iter.elapsed_ns > 0
        smld_DC_x = [e.x for e in smld_iter.emitters]
        smld_DC_y = [e.y for e in smld_iter.emitters]
        rmsd_iter = sqrt(sum((smld_DC_x .- smld_noisy_x).^2 .+
                             (smld_DC_y .- smld_noisy_y).^2) ./ N)
        print("rmsd 2D (iterative) = $rmsd_iter\n")
        @test rmsd_iter < 5.0
    end

    # --- Test warm start ---
    @testset "Warm start" begin
        (smld1, info1) = DC.driftcorrect(smld_drift; quality=:singlepass)
        (smld2, info2) = DC.driftcorrect(smld_drift; warm_start=info1.model)
        @test info2 isa DC.DriftInfo
        @test info2.elapsed_ns > 0
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
    @test isapprox(rmsd, 0.0; atol = 5.0)

    # --- Test driftcorrect with different degree ---
    (smld_corrected, info) = DC.driftcorrect(smld_drift; degree=3)
    smld_DC_x = [e.x for e in smld_corrected.emitters]
    smld_DC_y = [e.y for e in smld_corrected.emitters]
    rmsd = sqrt(sum((smld_DC_x .- smld_noisy_x).^2 .+
                    (smld_DC_y .- smld_noisy_y).^2) ./ N)
    print("rmsd 2D (degree=3) = $rmsd\n")
    @test isapprox(rmsd, 0.0; atol = 5.0)

    # ========== 3D ==========

    # --- Test correctdrift (LegendrePolynomial) ---
    N = length(smld_noisy3.emitters)
    driftmodel3 = DC.LegendrePolynomial(smld_noisy3; degree=2, initialize="random")
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
    @test isapprox(rmsd, 0.0; atol = 10.0)

    # --- Test driftcorrect with verbose ---
    (smld_corrected, info) = DC.driftcorrect(smld_drift3; maxn=100, verbose=1)
    smld_DC_x = [e.x for e in smld_corrected.emitters]
    smld_DC_y = [e.y for e in smld_corrected.emitters]
    smld_DC_z = [e.z for e in smld_corrected.emitters]
    rmsd = sqrt(sum((smld_DC_x .- smld_noisy3_x).^2 .+
                    (smld_DC_y .- smld_noisy3_y).^2 .+
                    (smld_DC_z .- smld_noisy3_z).^2) ./ N)
    print("rmsd 3D (maxn=100) = $rmsd\n")
    @test isapprox(rmsd, 0.0; atol = 10.0)

    # --- Test 3D quality tiers ---
    @testset "3D quality tiers" begin
        # FFT
        (smld_fft, info_fft) = DC.driftcorrect(smld_drift3; quality=:fft)
        @test info_fft.iterations == 0
        @test info_fft.elapsed_ns > 0

        # Iterative
        (smld_iter, info_iter) = DC.driftcorrect(smld_drift3; quality=:iterative, max_iterations=2)
        @test info_iter.iterations >= 1
        @test info_iter.elapsed_ns > 0
    end
end
