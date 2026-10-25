# Continuous mode smoke: chunking and the CC-seed inter path.
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
# Continuous mode smoke — exercises chunking + the CC-seed inter path
# (_ccseed_inter_continuous!). Tiny data, so assert only that it runs and
# returns valid output; full-sim drift recovery lives in extended_tests.jl.
# -----------------------------------------------------------------------
@testset "continuous mode smoke" begin
    Random.seed!(7)
    cmodel = DC.LegendrePolynomial(smld_small; degree = 2,
                                    initialize = "random", rscale = 0.05)
    cmodel.inter[1].dm .= 0.0
    smld_cdrift = DC.applydrift(smld_small, cmodel)

    (sc, cinfo) = DC.driftcorrect(smld_cdrift; dataset_mode = :continuous,
                                   n_chunks = 2, degree = 2)
    @test sc isa DC.SMLD
    @test cinfo isa DC.DriftInfo
    @test cinfo.model isa DC.LegendrePolynomial
    @test length(sc.emitters) == length(smld_cdrift.emitters)
    @test cinfo.elapsed_s > 0

    # _ccseed_inter_continuous! runs without error on a 2-chunk model and
    # produces one inter shift per chunk.
    ci = DC.chunk_smld(smld_small; n_chunks = 2)
    m2 = DC.LegendrePolynomial(ci.smld; degree = 2)
    @test_nowarn DC._ccseed_inter_continuous!(m2, ci.smld)
    @test length(m2.inter) == ci.smld.n_datasets
end
