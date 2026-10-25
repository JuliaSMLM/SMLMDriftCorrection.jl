# driftcorrect and align_smld interface smoke on tiny multi-dataset data (Long: runs the pipeline).
using Test, SMLMDriftCorrection, SMLMSim, Random

const DC = SMLMDriftCorrection

Random.seed!(42)

# -----------------------------------------------------------------------
# Tiny simulated SMLD (1 dataset, 200 frames, density 5/μm² on a 32×32px
# camera @ 100 nm/pixel = 3.2 μm FOV). Used by smoke + interface tests.
# Kept deliberately small so any thread/RNG sensitivity stays bounded and
# nothing here makes a noisy "drift correction reduces correlation"-style
# assertion.
# -----------------------------------------------------------------------
params_small = StaticSMLMConfig(
    5.0,    # density (emitters/μm²)
    0.13,   # σ_psf
    30,     # minphotons
    1,      # ndatasets
    200,    # nframes
    50.0,   # framerate
    2,      # ndims
    [0.0, 1.0]
)
(smld_small, _) = simulate(
    params_small;
    pattern  = Nmer2D(n = 4, d = 0.2),
    molecule = GenericFluor(; photons = 5000.0, k_on = 0.05, k_off = 50.0),
    camera   = IdealCamera(1:32, 1:32, 0.1)
)

# -----------------------------------------------------------------------
# driftcorrect interface smoke — verify return type / DriftInfo fields.
# Uses tiny multi-dataset data so inter alignment runs. Does NOT assert
# noisy RMSD values that would be sensitive to thread/RNG variance.
# -----------------------------------------------------------------------
@testset "driftcorrect interface smoke" begin
    params_3ds = StaticSMLMConfig(5.0, 0.13, 30, 3, 200, 50.0, 2, [0.0, 1.0])
    (smld_test, _) = simulate(params_3ds;
        pattern  = Nmer2D(n = 4, d = 0.2),
        molecule = GenericFluor(; photons = 5000.0, k_on = 0.05, k_off = 50.0),
        camera   = IdealCamera(1:32, 1:32, 0.1))
    Random.seed!(123)
    model = DC.LegendrePolynomial(smld_test; degree = 2,
                                   initialize = "random", rscale = 0.1)
    model.inter[1].dm .= 0.0
    smld_drifted = DC.applydrift(smld_test, model)

    # Singlepass — default
    (smld_c, info) = DC.driftcorrect(smld_drifted)
    @test smld_c isa DC.SMLD
    @test info isa DC.DriftInfo
    @test info isa DC.AbstractSMLMInfo
    @test info.model isa DC.LegendrePolynomial
    @test info.elapsed_s > 0
    @test info.backend == :cpu
    @test info.iterations == 1                       # singlepass
    @test info.converged == true
    @test info.entropy isa Float64
    @test info.history isa Vector{Float64}

    # Config dispatch
    cfg = DC.DriftConfig(; quality = :singlepass, degree = 2)
    (smld_c2, info_c2) = DC.driftcorrect(smld_drifted, cfg)
    @test smld_c2 isa DC.SMLD
    @test info_c2 isa DC.DriftInfo
    @test info_c2.iterations == 1

    # FFT — quick path, no optimization, deterministic
    (smld_fft, info_fft) = DC.driftcorrect(smld_drifted; quality = :fft)
    @test info_fft.iterations == 0
    @test info_fft.converged == true

    # Warm-start path returns valid info
    (smld_w, info_w) = DC.driftcorrect(smld_drifted; warm_start = info.model)
    @test info_w isa DC.DriftInfo
    @test info_w.elapsed_s > 0
end

# -----------------------------------------------------------------------
# align_smld smoke — known shift recovery on small data.
# -----------------------------------------------------------------------
@testset "align_smld smoke" begin
    # Use the dataset=1 subset as an "independent acquisition" baseline
    # (smld_small is single-dataset already)
    smld_base = smld_small
    true_shift = [0.3, -0.2]
    smld_shifted = deepcopy(smld_base)
    for e in smld_shifted.emitters
        e.x += true_shift[1]
        e.y += true_shift[2]
    end
    (aligned, info) = DC.align_smld([smld_base, smld_shifted]; method = :fft)
    @test info isa DC.AlignInfo
    @test length(aligned) == 2
    @test info.shifts[1] == [0.0, 0.0]
    @test info.transforms[1] isa DC.ShiftTransform
    # FFT recovery within 150 nm — generous tolerance for tiny data
    @test isapprox(info.shifts[2], true_shift; atol = 0.150)
end
