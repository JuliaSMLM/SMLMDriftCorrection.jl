# Edge-case guards on small inputs.
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
# Edge-case guards
# -----------------------------------------------------------------------
@testset "Edge-case guards" begin
    # findshift_damped with σ_px ≈ 0 → σ_px floor must prevent collapse
    sh = DC.findshift_damped(smld_small, smld_small;
            histbinsize = 0.10, prior_shift = [0.0, 0.0], prior_sigma = 0.0)
    @test all(isfinite, sh)

    # findshift_damped: known shift + prior centered on truth recovers
    true_sh = [0.5, -0.3]
    smld_shifted = deepcopy(smld_small)
    for e in smld_shifted.emitters
        e.x += true_sh[1]; e.y += true_sh[2]
    end
    sh = DC.findshift_damped(smld_small, smld_shifted;
            histbinsize = 0.10, prior_shift = true_sh, prior_sigma = 1.0)
    @test isapprox(sh, true_sh; atol = 0.15)

    # filter_emitters to empty mask
    empty_mask = falses(length(smld_small.emitters))
    smld_empty = DC.filter_emitters(smld_small, empty_mask)
    @test isempty(smld_empty.emitters)

    # findintra! on empty per-dataset subset is a no-op
    model = DC.LegendrePolynomial(smld_small; degree = 2)
    @test_nowarn DC.findintra!(model.intra[1], smld_empty, 1, 50)

    # find_dense_roi: requesting more than available returns all locs
    n_total = length(smld_small.emitters)
    idx_all = DC.find_dense_roi(smld_small, n_total + 100)
    @test length(idx_all) == n_total
end
