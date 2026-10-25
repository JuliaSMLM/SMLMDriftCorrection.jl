# gridphase_replica seed 2: a flat entropy cost keeps the incumbent.
using Test, SMLMDriftCorrection, SMLMSim, Random

const DC = SMLMDriftCorrection

# Flat entropy cost (the dense dataset's own neighbors dominate): the incumbent stays
# (battery gridphase_replica seed 2: 8.6 nm with refined ahead of the incumbent).
@testset "flat entropy cost keeps the incumbent (gridphase_replica seed 2)" begin
    cam = IdealCamera(1:64, 1:64, 0.1)
    site(x, y, n) = [Emitter2DFit{Float64}(x, y, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0; σ_xy = 0.0, track_id = i) for i in 1:n]
    cfg1 = StaticSMLMConfig(1.0, 0.13, 50, 1, 300, 50.0, 2, [0.0, 1.0])
    rng = Random.MersenneTwister(2)
    pos = [(0.6 + 5.0 * rand(rng), 0.6 + 5.0 * rand(rng)) for _ in 1:60]
    ref = reduce(vcat, [site(p[1], p[2], 4) for p in pos])
    for (i, e) in enumerate(ref); e.track_id = i; end
    Random.seed!(2)
    (r, _) = simulate(cfg1; starting_conditions = ref,
        molecule = GenericFluor(; photons = 5000.0, k_on = 2.0, k_off = 50.0), camera = cam)
    rep_em = [Emitter2DFit{Float64}(p[1] + 1.013 - 0.025 + 0.001 * randn(rng), p[2] + 1.013 - 0.025 + 0.001 * randn(rng), 5000.0, 0.0, 0.002, 0.002, 0.0, 0.0; σ_xy = 0.0, frame = 1, dataset = 1, track_id = 0, id = 0) for p in pos]
    tgt = [Emitter2DFit{Float64}(p[1] + 1.013, p[2] + 1.013, 5000.0, 0.0, 0.002, 0.002, 0.0, 0.0; σ_xy = 0.0, frame = 1, dataset = 2, track_id = 0, id = 0) for p in pos]
    smld = BasicSMLD(vcat(r.emitters, rep_em, tgt), cam, 300, 2, Dict{String,Any}())
    m = DC.LegendrePolynomial(smld; degree = 2)
    DC._cc_primary_inter!(m, smld, 100; n_passes = 2, init = true)
    @test hypot((m.inter[2].dm .- m.inter[1].dm .- [1.013, 1.013])...) < 0.002
end
