# Sparse continuous data (M1, continuous flavor): 274 localizations over 800 frames (SMLMSim tetramers, emitters
# blinking about once each) with a known degree-2 drift. Unchunked, 0.2.10's intra polynomial bent toward unrelated
# localizations (no background in the entropy cost); chunked into 4, chunk boundaries took cross-correlations with
# nothing to lock onto and two chunks landed 2.2-2.4 um off, with nothing in the result to say so.
using Test, SMLMDriftCorrection, SMLMSim, Random, Statistics

const DC = SMLMDriftCorrection

function sparse_continuous()
    cam = IdealCamera(1:32, 1:32, 0.1)
    Random.seed!(7)
    (truth, _) = simulate(StaticSMLMConfig(8.0, 0.13, 30, 1, 800, 50.0, 2, [0.0, 1.0]);
                          pattern = Nmer2D(n = 4, d = 0.2),
                          molecule = GenericFluor(; photons = 5000.0, k_on = 0.05, k_off = 50.0), camera = cam)
    drift = DC.LegendrePolynomial(truth; degree = 2, initialize = "random", rscale = 0.1)
    return DC.applydrift(truth, drift), truth
end

# mean (x, y) error and rms spread of (corrected - true) in each 200-frame block
function block_errors(out, truth)
    dx = [a.x - t.x for (a, t) in zip(out.emitters, truth.emitters)]
    dy = [a.y - t.y for (a, t) in zip(out.emitters, truth.emitters)]
    fr = [e.frame for e in truth.emitters]
    blocks = [(fr .> (c - 1) * 200) .& (fr .<= c * 200) for c in 1:4]
    means = [(mean(dx[m]), mean(dy[m])) for m in blocks]
    spread = [sqrt(mean((dx[m] .- mean(dx[m])) .^ 2 .+ (dy[m] .- mean(dy[m])) .^ 2)) for m in blocks]
    return means, spread
end
largest_gap(means) = maximum(hypot((a .- b)...) for a in means for b in means)

@testset "sparse continuous, unchunked: the drift is recovered" begin
    # 0.2.10: block means 123-250 nm apart in x, spread up to 50 nm (here 15 nm and 4 nm). Coarse to fine
    # alone left 91 nm and 34 nm: its wide floor lets unrelated pairs pull; the direct start wins here.
    (smld, truth) = sparse_continuous()
    Random.seed!(1)
    (out, _) = driftcorrect(smld; dataset_mode = :continuous, degree = 2)
    means, spread = block_errors(out, truth)
    @test largest_gap(means) < 0.050
    @test maximum(spread) < 0.020
end

@testset "sparse continuous, 4 chunks: no chunk thrown, the unsupported chunks named" begin
    # 0.2.10: chunks 3 and 4 2.2-2.4 um off the others, no warning (here 0.40-0.82 um apart over seeds 1-3, all four
    # named: about 70 localizations per chunk do not establish where a chunk sits relative to the others)
    (smld, truth) = sparse_continuous()
    Random.seed!(1)
    local out, info
    @test_logs (:warn, r"chunk\(s\) \[") match_mode = :any begin
        (out, info) = driftcorrect(smld; dataset_mode = :continuous, degree = 2, n_chunks = 4)
    end
    means, _ = block_errors(out, truth)
    @test largest_gap(means) < 1.0
    @test length(info.dataset_support) == 4
    @test all(s -> !s.supported, info.dataset_support)
end

# Codex review 6 (P2), hand-built: zero drift, two chunks. The four shared singles give the correct boundary CC peak a
# score of 4; shifted by (1, 0) um the current chunk's singles at (7, 6) and (7.8, 6.4) land on the reference's
# four-localization groups and score 8. The endpoint prediction overlaps 4 of 100 localizations, the wrong CC 2 of 100,
# and 0.2.10's 0.05 overlap rule accepted the CC 1 um off. A move that large now needs a significant lead.
@testset "continuous boundary: a far CC without a lead is not taken (Codex review 6, P2)" begin
    S = [(1.0, 1.0), (1.4, 1.8), (2.2, 2.6), (3.6, 4.2)]
    ref = vcat(S, fill((6.0, 6.0), 4), fill((6.8, 6.4), 4))
    cur = vcat(S, [(7.0, 6.0), (7.8, 6.4)], [(0.5 + 0.08k, 8.0) for k in 0:93])
    σ = 0.014
    em = [Emitter2DFit{Float64}(p[1], p[2], 1000.0, 10.0, σ, σ, 0.0, 0.0; σ_xy = 0.0, frame = 1, dataset = d)
          for (d, pts) in enumerate((ref, cur)) for p in pts]
    smld = BasicSMLD(em, IdealCamera(1:100, 1:100, 0.1), 300, 2, Dict{String,Any}())
    (_, info) = driftcorrect(smld; dataset_mode = :continuous, degree = 0)
    @test maximum(abs.(info.model.inter[2].dm .- info.model.inter[1].dm)) < 0.05
end
