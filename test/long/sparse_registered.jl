# Sparse registered datasets (the 2026-11-02 review, M1): 60 datasets of 300 frames on a 12.8 μm field, SMLMSim
# hexamers whose emitters each blink about once in four datasets (k_on 0.05), true inter shifts drawn with sd 30 nm
# per axis, no intra drift. In 0.2.10 the default tier's degree-2 intra polynomials bent to pull unrelated
# localizations together (no background in the entropy cost) and left datasets microns off, and `:fft` accepted
# cross-correlations with nothing to lock onto (ties went to the CC), throwing single datasets microns away.
using Test, SMLMDriftCorrection, SMLMSim, Random, Statistics

const DC = SMLMDriftCorrection

function sparse_fixture(density, seed)
    Random.seed!(seed)
    cam = IdealCamera(1:128, 1:128, 0.1)
    (s, _) = simulate(StaticSMLMConfig(density, 0.13, 50, 60, 300, 50.0, 2, [0.0, 1.0]);
                      pattern = Nmer2D(n = 6, d = 0.2),
                      molecule = GenericFluor(; photons = 5000.0, k_on = 0.05, k_off = 50.0), camera = cam)
    rng = MersenneTwister(100 + seed)
    truth = [0.030 .* randn(rng, 2) for _ in 1:60]
    em = deepcopy(s.emitters)
    for e in em
        e.x += truth[e.dataset][1]
        e.y += truth[e.dataset][2]
    end
    return BasicSMLD(em, cam, 300, 60, Dict{String,Any}()), s.emitters
end

# Per dataset: the mean of (corrected - true position) after removing the median over datasets (a global shift is
# image-invariant), and the rms of (corrected - true) about that mean (a runaway intra polynomial smears a dataset).
function dataset_errors(sc, ref)
    ds = [e.dataset for e in ref]
    dx = [sc.emitters[i].x - ref[i].x for i in eachindex(ref)]
    dy = [sc.emitters[i].y - ref[i].y for i in eachindex(ref)]
    live = [d for d in 1:60 if any(==(d), ds)]
    m = Dict(d => (mean(dx[ds .== d]), mean(dy[ds .== d])) for d in live)
    gx = median(first(m[d]) for d in live)
    gy = median(last(m[d]) for d in live)
    err = [hypot(m[d][1] - gx, m[d][2] - gy) for d in live]
    spread = [sqrt(mean((dx[ds .== d] .- m[d][1]) .^ 2 .+ (dy[ds .== d] .- m[d][2]) .^ 2)) for d in live]
    return err, spread
end

@testset "sparse registered: default tier, intra polynomials do not run away" begin
    # ~100 localizations per dataset; 0.2.10: a dataset 1.9 um off, 20 of 60 over 200 nm, spread up to 1.5 um
    # (here 52 nm and 91 nm)
    (smld, ref) = sparse_fixture(0.42, 3)
    Random.seed!(3)
    (sc, _) = driftcorrect(smld)
    err, spread = dataset_errors(sc, ref)
    @test maximum(err) < 0.300
    @test maximum(spread) < 0.300
end

@testset "sparse registered: :fft does not throw a dataset" begin
    # ~19 localizations per dataset; 0.2.10: one dataset 4.0 um off (here 88 nm)
    (smld, ref) = sparse_fixture(0.075, 2)
    Random.seed!(2)
    (sc, _) = driftcorrect(smld; quality = :fft)
    err, _ = dataset_errors(sc, ref)
    @test maximum(err) < 0.300
end
