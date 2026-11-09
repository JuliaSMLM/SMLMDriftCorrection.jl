# Sparse data (the 2026-11-02 review, M1): the background floor of the intra cost, the evidence rule for large inter
# moves, and the per-dataset alignment support in DriftInfo. The end-to-end fixtures are in long/sparse_registered.jl.
using Test, SMLMDriftCorrection, SMLMSim, Random, Logging

const DC = SMLMDriftCorrection

include(joinpath(@__DIR__, "points_smld.jl"))

@testset "intra cost: background floor" begin
    k = 10
    lk = log(k)
    b = -lk - DC._BG_C^2 / 2                       # one neighbor at _BG_C sigma among k
    # the term is the log of the neighbor mixture plus the background
    for lse in (-200.0, -12.5, -3.0, 0.0, 2.0)
        @test DC._neighbor_term(lse, lk) ≈ log(exp(lse) + exp(-DC._BG_C^2 / 2)) - lk
    end
    # an isolated localization (all neighbors far, lse -> -Inf) adds a constant: no pull on the fit
    @test DC._neighbor_term(-1e6, lk) ≈ b
    @test DC._neighbor_term(-1e6, lk) == DC._neighbor_term(-2e6, lk)
    # a localization with a close partner is unchanged to within exp(-_BG_C^2 / 2)
    @test abs(DC._neighbor_term(0.0, lk) - (0.0 - lk)) < 2 * exp(-DC._BG_C^2 / 2)
    # Float32 data
    @test DC._neighbor_term(-1f6, Float32(lk)) isa Float32
    @test isfinite(DC._neighbor_term(-1f6, Float32(lk)))
end

# Eight separated sites (200 nm apart), ten localizations each; the reference holds the same sites.
sites8 = [(1.0 + 0.2i, 1.5) for i in 0:7]
site_locs(sites, n; rng = MersenneTwister(1)) =
    [(p[1] + 0.003randn(rng), p[2] + 0.003randn(rng)) for p in sites for _ in 1:n]

@testset "evidence rule for large inter moves" begin
    ref = site_locs(sites8, 10)
    tree = DC._coord_tree(points_smld([ref]), 2)
    # the target holds the same sites shifted by +3 um in x; the incumbent (0) covers nothing
    tgt = points_smld([[(p[1] + 3.0, p[2]) for p in site_locs(sites8, 10; rng = MersenneTwister(2))]])
    # eight sites lead, nothing against (z = sqrt(8) > 2): the move is evidenced
    @test !DC._far_without_evidence(tgt, tree, [3.0, 0.0], [0.0, 0.0], 2; raw_cc = false)
    # a far candidate covering nothing is not evidenced, and a near one is not judged
    @test DC._far_without_evidence(tgt, tree, [5.0, 0.0], [0.0, 0.0], 2; raw_cc = false)
    @test !DC._far_without_evidence(tgt, tree, [0.15, 0.0], [0.0, 0.0], 2; raw_cc = false)
    # four sites alone are not significant (z = 2): five are needed when the incumbent covers none
    tgt4 = points_smld([[(p[1] + 3.0, p[2]) for p in site_locs(sites8[1:4], 10)]])
    tgt5 = points_smld([[(p[1] + 3.0, p[2]) for p in site_locs(sites8[1:5], 10)]])
    @test DC._far_without_evidence(tgt4, tree, [3.0, 0.0], [0.0, 0.0], 2; raw_cc = false)
    @test !DC._far_without_evidence(tgt5, tree, [3.0, 0.0], [0.0, 0.0], 2; raw_cc = false)
    # a raw CC sits up to half a bin diagonal off the truth: 40 nm off covers nothing at 30 nm but is judged at
    # 30 nm + 35 nm, the precision it was measured to
    cc = [3.0 + 0.028, 0.028]
    @test DC._far_without_evidence(tgt, tree, cc, [0.0, 0.0], 2; raw_cc = false)
    @test !DC._far_without_evidence(tgt, tree, cc, [0.0, 0.0], 2; raw_cc = true)
    # the guard: a far CC without evidence is refused, an evidenced one accepted
    inter = [0.0, 0.0]
    @test !DC._accept_if_better_overlap!(inter, [5.0, 0.0], tgt, tree, 2)
    @test inter == [0.0, 0.0]
    @test DC._accept_if_better_overlap!(inter, [3.0, 0.0], tgt, tree, 2)
    @test inter == [3.0, 0.0]
end

function four_datasets(density; seed = 1)
    Random.seed!(seed)
    (s, _) = simulate(StaticSMLMConfig(density, 0.13, 50, 4, 300, 50.0, 2, [0.0, 1.0]);
                      pattern = Nmer2D(n = 6, d = 0.2),
                      molecule = GenericFluor(; photons = 5000.0, k_on = 0.5, k_off = 50.0),
                      camera = IdealCamera(1:64, 1:64, 0.1))
    return s
end
throw3(s) = (t = deepcopy(s); for e in t.emitters; e.dataset == 3 && (e.x += 2.0; e.y -= 1.3); end; t)

@testset "dataset support: aligned vs thrown" begin
    for (density, label) in ((2.0, "dense"), (0.05, "sparse, 11-22 localizations per dataset"))
        s = four_datasets(density)
        sup = DC._dataset_support(s)
        @test length(sup) == 4
        @test [x.dataset for x in sup] == 1:4
        @test all(x -> x.supported && x.z > 2, sup)
        @test all(x -> 0 < x.radius <= DC._CC_VERIFY_RADIUS, sup)
        @test all(x -> x.colocalized > x.expected, sup)
        t = DC._dataset_support(throw3(s))
        @test !t[3].supported
        @test t[3].z < 1
        @test all(x -> x.supported, t[[1, 2, 4]])
    end
    # dense data: the radius shrinks below 30 nm, where chance coverage would saturate
    @test all(x -> x.radius < 0.025, DC._dataset_support(four_datasets(2.0)))
    # empty dataset and a single dataset
    s = four_datasets(0.5)
    e = BasicSMLD([x for x in s.emitters if x.dataset != 2], s.camera, s.n_frames, 4, Dict{String,Any}())
    sup = DC._dataset_support(e)
    @test sup[2].n_locs == 0 && !sup[2].supported && isnan(sup[2].z)
    @test all(x -> x.supported, sup[[1, 3, 4]])
    one = BasicSMLD([x for x in s.emitters if x.dataset == 1], s.camera, s.n_frames, 1, Dict{String,Any}())
    @test isempty(DC._dataset_support(one))
    # 3D
    pts = [(1.0 + 0.2i, 1.0 + 0.2j, 0.1 * ((i + j) % 3)) for i in 0:6 for j in 0:6]
    rng = MersenneTwister(5)
    jit(p) = (p[1] + 0.005randn(rng), p[2] + 0.005randn(rng), p[3] + 0.01randn(rng))
    s3 = points_smld([[jit(p) for p in pts for _ in 1:3] for _ in 1:3])
    @test all(x -> x.supported, DC._dataset_support(s3))
    s3t = points_smld([[jit(p) for p in pts for _ in 1:3], [jit(p) for p in pts for _ in 1:3],
                       [jit((p[1] + 0.1, p[2] + 0.1, p[3])) for p in pts for _ in 1:3]])
    @test !DC._dataset_support(s3t)[3].supported
end

@testset "driftcorrect names an unaligned dataset" begin
    # dataset 3 holds structure the others do not have: it cannot be aligned, and the result says so
    s = four_datasets(2.0)
    rng = MersenneTwister(9)
    em = [e for e in s.emitters if e.dataset != 3]
    for e in s.emitters
        e.dataset == 3 || continue
        f = deepcopy(e)
        f.x = 0.5 + 5.4rand(rng)
        f.y = 0.5 + 5.4rand(rng)
        push!(em, f)
    end
    smld = BasicSMLD(em, s.camera, s.n_frames, 4, Dict{String,Any}())
    local info
    @test_logs (:warn, r"dataset\(s\) \[3\] do not co-localize") match_mode = :any begin
        (_, info) = driftcorrect(smld; quality = :singlepass, degree = 0)
    end
    @test length(info.dataset_support) == 4
    @test !info.dataset_support[3].supported
    @test all(x -> x.supported, info.dataset_support[[1, 2, 4]])
    # an aligned run warns about nothing and reports every dataset supported
    (_, info2) = @test_logs min_level = Logging.Warn driftcorrect(s; quality = :fft)
    @test all(x -> x.supported, info2.dataset_support)
    # continuation recomputes it
    (_, info3) = driftcorrect(s, info2; max_iterations = 1)
    @test length(info3.dataset_support) == 4
    # a positional DriftInfo still constructs (empty support)
    @test isempty(DC.DriftInfo(info.model, 0.0, :cpu, 1, true, 0.0, Float64[], nothing, (;)).dataset_support)
end
