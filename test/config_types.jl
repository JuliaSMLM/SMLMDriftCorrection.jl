# Config and info types, and the interface's argument checks.
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
# Type hierarchy + config dispatch
# -----------------------------------------------------------------------
@testset "Type hierarchy + config" begin
    @test DC.DriftConfig <: DC.AbstractSMLMConfig
    @test DC.DriftInfo  <: DC.AbstractSMLMInfo
    @test DC.AlignConfig <: DC.AbstractSMLMConfig
    @test DC.AlignInfo  <: DC.AbstractSMLMInfo
    @test DC.ShiftTransform     <: DC.AbstractAlignTransform
    @test DC.AffineTransform2D  <: DC.AbstractAlignTransform
    @test DC.AffineTransform3D  <: DC.AbstractAlignTransform

    cfg = DC.DriftConfig(; quality = :singlepass, degree = 2, verbose = 0)
    @test cfg.quality == :singlepass
    @test cfg.degree == 2

    ac = DC.AlignConfig(method = :fft, transform = :affine, maxn = 50)
    @test ac.method == :fft
    @test ac.transform == :affine
    @test ac.maxn == 50

    # Bad arg validation
    @test_throws Exception DC.driftcorrect(smld_small; quality = :nope)
    @test_throws Exception DC.driftcorrect(smld_small; dataset_mode = :nope)
    @test_throws ErrorException DC.align_smld([smld_small])
    @test_throws ErrorException DC.align_smld([smld_small, smld_small]; method = :bad)
    @test_throws ErrorException DC.align_smld([smld_small, smld_small]; transform = :bad)
end
