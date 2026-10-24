# Single-function units: frame normalization, Legendre evaluation, chunk parameters, rebuild gating, applydrift/correctdrift round trip.
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
# Pure-function units — deterministic, no driftcorrect, no SMLMSim
# -----------------------------------------------------------------------
@testset "Pure-function units" begin
    # normalize_frame
    @test DC.normalize_frame(1, 1)   == 0.0    # n_frames=1 collapse safe
    @test DC.normalize_frame(1, 100) == -1.0
    @test DC.normalize_frame(100, 100) == +1.0
    @test DC.normalize_frame(50, 99)  == 0.0   # midpoint

    # LegendrePoly1D evaluate_at_frame stays finite at endpoints
    poly = DC.LegendrePoly1D(2, 100)
    poly.coefficients .= [0.5, 0.25]
    @test isfinite(DC.evaluate_at_frame(poly, 1))
    @test isfinite(DC.evaluate_at_frame(poly, 50))
    @test isfinite(DC.evaluate_at_frame(poly, 100))
    # n_frames=1 case
    poly_1f = DC.LegendrePoly1D(2, 1)
    poly_1f.coefficients .= [0.5, 0.25]
    @test isfinite(DC.evaluate_at_frame(poly_1f, 1))

    # compute_chunk_params guards
    p1 = DC.compute_chunk_params(10; n_chunks = 20)
    @test p1.n_chunks <= 10 && p1.frames_per_chunk >= 1
    p2 = DC.compute_chunk_params(10; chunk_frames = 100)
    @test p2.n_chunks == 1 && p2.frames_per_chunk == 10
    @test DC.compute_chunk_params(1).n_chunks == 1
    @test DC.compute_chunk_params(1).frames_per_chunk >= 1

    # calculate_n_locs_required scaling (pure function on Float64s)
    n_req_d2 = DC.calculate_n_locs_required(1000; degree = 2)
    n_req_d3 = DC.calculate_n_locs_required(1000; degree = 3)
    @test n_req_d2 isa Int && n_req_d2 > 0
    @test n_req_d3 > n_req_d2
    @test DC.calculate_n_locs_required(1000; σ_target = 0.0005) >
          DC.calculate_n_locs_required(1000; σ_target = 0.002)
    @test DC.calculate_n_locs_required(5000) >
          DC.calculate_n_locs_required(1000)
end

# -----------------------------------------------------------------------
# Vector-delta rebuild gating (pure unit, no SMLD)
# -----------------------------------------------------------------------
@testset "Rebuild gating: vector delta" begin
    # Same magnitude, opposite sign at the middle test frame — scalar
    # max_drift_magnitude would miss this; vector delta must catch it.
    last    = [0.0  0.05  0.0;
               0.0  0.00  0.0]
    current = [0.0 -0.06  0.0;
               0.0  0.00  0.0]
    @test DC.max_drift_vec_delta(current, last) ≈ 0.11 atol = 1e-12
    @test DC.max_drift_vec_delta(last, last) == 0.0

    # Translation (same direction, larger magnitude) still detected
    current2 = [0.0  0.20  0.0;
                0.0  0.00  0.0]
    @test DC.max_drift_vec_delta(current2, last) ≈ 0.15 atol = 1e-12

    # NeighborState.last_drift_vecs initialised to +Inf so the first cost
    # eval always rebuilds regardless of how small actual drift is.
    st = DC.NeighborState(10, 5, 0.1, 2)
    cur = [0.0 0.0 0.0; 0.0 0.0 0.0]
    @test DC.max_drift_vec_delta(cur, st.last_drift_vecs) == Inf
end

# -----------------------------------------------------------------------
# correctdrift roundtrip (2D) — exact recovery of a known drift, no
# optimization involved. Deterministic.
# -----------------------------------------------------------------------
@testset "correctdrift roundtrip (2D)" begin
    Random.seed!(123)
    N = length(smld_small.emitters)
    model = DC.LegendrePolynomial(smld_small; degree = 2,
                                   initialize = "random", rscale = 0.1)
    model.inter[1].dm .= 0.0     # DS1 reference
    smld_drifted = DC.applydrift(smld_small, model)
    smld_recovered = DC.correctdrift(smld_drifted, model)
    rmsd = sqrt(sum((e1.x - e2.x)^2 + (e1.y - e2.y)^2
                    for (e1, e2) in zip(smld_recovered.emitters, smld_small.emitters)) / N)
    @test isapprox(rmsd, 0.0; atol = 1e-10)
end
