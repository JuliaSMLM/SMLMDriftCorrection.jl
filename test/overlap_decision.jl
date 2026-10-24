# Overlap decision scoring: McNemar leads, occupancy (monotone, order independent, compact sites, counts vs a KD-tree), and the arbiter with stub score and cost.
using Test, SMLMDriftCorrection, SMLMSim, Random

const DC = SMLMDriftCorrection
include("points_smld.jl")

# -----------------------------------------------------------------------
# Overlap decisions need a significant lead: McNemar's z on the discordant
# localizations (inlier under one shift only) must exceed _OVERLAP_Z.
# -----------------------------------------------------------------------
@testset "overlap leads (McNemar)" begin
    mk(nb, nc, nboth) = (BitVector(vcat(trues(nb), falses(nc), trues(nboth))),
                         BitVector(vcat(falses(nb), trues(nc), trues(nboth))))
    @test !DC._overlap_leads(mk(3, 0, 50)...)      # a 3-localization lead is noise
    @test DC._overlap_leads(mk(100, 10, 50)...)
    @test !DC._overlap_leads(mk(0, 0, 50)...)      # identical flags
    @test !DC._overlap_leads(mk(30, 30, 50)...)    # symmetric
    @test !DC._overlap_leads(reverse(mk(100, 10, 50))...)  # the trailing side never leads
end

# -----------------------------------------------------------------------
# Occupancy: inlier localizations kept when each is dropped if an already-kept
# inlier lies within _OCC_SEP, so a site localized many times counts once.
# -----------------------------------------------------------------------
@testset "occupancy" begin
    # 4 sites 1 um apart, each localized 10 times within 3 nm: occupancy 4, count 40
    pts = [(i + 0.001 * j, 2.0 + 0.001 * j) for i in 1:4 for j in 0:9]
    sd = points_smld([pts])
    all_in = trues(length(pts))
    @test DC._occupancy(sd, all_in) ≈ 4
    # two localizations 40 nm apart are two sites
    sd = points_smld([[(1.0, 1.0), (1.040, 1.0)]])
    @test DC._occupancy(sd, trues(2)) ≈ 2
    # only flagged localizations count
    sd = points_smld([[(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (3.001, 1.0)]])
    @test DC._occupancy(sd, BitVector([1, 0, 1, 1])) ≈ 2
    @test DC._occupancy(sd, falses(4)) ≈ 0
    # 3D uses z: same (x, y), 40 nm apart in z are two sites, 3 nm apart one
    sd = points_smld([[(1.0, 1.0, 0.0), (1.0, 1.0, 0.040)]])
    @test DC._occupancy(sd, trues(2)) ≈ 2
    sd = points_smld([[(1.0, 1.0, 0.0), (1.0, 1.0, 0.003)]])
    @test DC._occupancy(sd, trues(2)) ≈ 1
end

# -----------------------------------------------------------------------
# Arbiter: the candidate with the largest score wins (occupancy in
# _cc_primary_inter!); ties keep refined, then CC, then the incumbent.
# -----------------------------------------------------------------------
@testset "arbiter" begin
    # 10 sites of one localization each (indices 1:10), 6 more sites (11:16), and
    # one site localized 30 times (17:46)
    pts = vcat([(Float64(i), 1.0) for i in 1:16], [(20.0 + 0.001 * j, 1.0) for j in 1:30])
    sd = points_smld([pts])
    flag(idx) = (f = falses(length(pts)); f[idx] .= true; f)
    occ(f) = DC._occupancy(sd, f)
    ref, cc, inc = [0.0, 0.0], [0.020, 0.0], [0.040, 0.0]
    inR, inC, inI = flag(1:10), flag(17:46), falses(length(pts))
    # the count prefers the CC (30 > 10); occupancy prefers refined (10 > 1)
    @test DC._arbitrate(ref, inR, cc, inC, inc, inI) === cc
    @test DC._arbitrate(ref, inR, cc, inC, inc, inI; score = occ) === ref
    # a strictly larger occupancy wins, from the CC or the incumbent
    @test DC._arbitrate(ref, flag(1:5), cc, flag(1:6), inc, falses(16)) === cc
    @test DC._arbitrate(ref, flag(1:5), cc, flag(1:6), inc, flag(1:7); score = occ) === inc
    # no CC candidate
    @test DC._arbitrate(ref, flag(1:5), nothing, nothing, inc, flag(1:6); score = occ) === inc
    # ties: refined over CC and incumbent, then CC over incumbent
    @test DC._arbitrate(ref, flag(1:5), cc, flag(6:10), inc, flag(11:15); score = occ) === ref
    @test DC._arbitrate(ref, flag(1:4), cc, flag(6:10), inc, flag(11:15); score = occ) === cc
    @test DC._arbitrate(ref, flag(1:5), nothing, nothing, inc, flag(6:10); score = occ) === ref
end

# Near ties: the occupancy winner picks the basin; among candidates within
# _CC_VERIFY_RADIUS of it that it does not lead significantly at the site level, the
# lowest entropy cost is kept (stub cost by shift x); equal costs keep incumbent,
# refined, CC.
@testset "arbiter: near ties go to the lowest entropy cost" begin
    flag(idx; n = 16) = (f = falses(n); f[idx] .= true; f)
    ref, cc, inc = [0.0, 0.0], [0.015, 0.0], [0.5, 0.0]
    cost_of(d) = θ -> d[round(θ[1], digits = 3)]
    # CC leads by occupancy but lies 15 nm from refined, which costs less: refined
    @test DC._arbitrate(ref, flag(1:5), cc, flag(1:8), inc, falses(16);
                        score = count, cost = cost_of(Dict(0.0 => 1.0, 0.015 => 2.0, 0.5 => 0.0))) === ref
    # the near candidate costs more: the winner stays
    @test DC._arbitrate(ref, flag(1:5), cc, flag(1:8), inc, falses(16);
                        score = count, cost = cost_of(Dict(0.0 => 3.0, 0.015 => 2.0, 0.5 => 0.0))) === cc
    # a far candidate is never compared, however low its cost
    @test DC._arbitrate(ref, flag(1:5), cc, flag(1:4), inc, flag(1:9);
                        score = count, cost = cost_of(Dict(0.0 => 5.0, 0.015 => 5.0, 0.5 => 0.0))) === inc
    # equal costs keep the incumbent, then refined, then CC
    @test DC._arbitrate(ref, flag(1:5), cc, flag(1:8), [0.02, 0.0], flag(1:7);
                        score = count, cost = cost_of(Dict(0.0 => 1.0, 0.015 => 1.0, 0.02 => 1.0))) == [0.02, 0.0]
    @test DC._arbitrate(ref, flag(1:5), cc, flag(1:8), inc, falses(16);
                        score = count, cost = cost_of(Dict(0.0 => 1.0, 0.015 => 1.0, 0.5 => 1.0))) === ref
    # costs equal within _COST_RTOL count as equal
    @test DC._arbitrate(ref, flag(1:5), cc, flag(1:8), [0.02, 0.0], flag(1:7);
                        score = count, cost = cost_of(Dict(0.0 => 1.0, 0.015 => 1.0 - 1e-12, 0.02 => 1.0))) == [0.02, 0.0]
    # a near candidate the winner leads significantly (20 sites vs 3; 4 vs 0 is not significant) is excluded, however low its cost
    @test DC._arbitrate(ref, flag(1:3; n = 25), cc, flag(1:20; n = 25), inc, falses(25);
                        score = count, cost = cost_of(Dict(0.0 => 0.0, 0.015 => 5.0, 0.5 => 5.0))) === cc
    @test DC._arbitrate(ref, flag(1:16; n = 25), cc, flag(1:20; n = 25), inc, falses(25);
                        score = count, cost = cost_of(Dict(0.0 => 0.0, 0.015 => 5.0, 0.5 => 5.0))) === ref
    # a duplicated shift costs one evaluation
    calls = Ref(0)
    @test DC._arbitrate(ref, flag(1:5), copy(ref), flag(1:5), [0.01, 0.0], flag(1:5);
                        score = count, cost = θ -> (calls[] += 1; 0.0)) == [0.01, 0.0]
    @test calls[] == 2
    # no cost call when the winner is alone in its neighborhood
    @test DC._arbitrate(ref, flag(1:5), cc, flag(1:4), inc, flag(1:9);
                        score = count, cost = _ -> error("cost evaluated")) === inc
end
# -----------------------------------------------------------------------
# Occupancy sums inverse local density weights, so it is monotone in the flags and independent of
# the emitter order.
# -----------------------------------------------------------------------
@testset "occupancy is monotone, order independent, and counts compact sites" begin
    Random.seed!(11)
    (s_m, _) = simulate(StaticSMLMConfig(3.0, 0.13, 50, 1, 300, 50.0, 2, [0.0, 1.0]);
        pattern  = Nmer2D(n = 3, d = 0.03),
        molecule = GenericFluor(; photons = 5000.0, k_on = 0.5, k_off = 50.0),
        camera   = IdealCamera(1:64, 1:64, 0.1))
    n = length(s_m.emitters)
    w = DC._occupancy_weights(s_m)
    rng = Random.MersenneTwister(3)
    for _ in 1:50
        sub = rand(rng, n) .< 0.3
        sup = sub .| (rand(rng, n) .< 0.3)
        @test DC._occupancy(w, BitVector(sub)) <= DC._occupancy(w, BitVector(sup)) + 1e-12
    end
    for _ in 1:10
        perm = Random.randperm(rng, n)
        s_p = BasicSMLD(s_m.emitters[perm], s_m.camera, 300, 1, Dict{String,Any}())
        f = BitVector(rand(rng, n) .< 0.5)
        @test isapprox(DC._occupancy(DC._occupancy_weights(s_p), f[perm]), DC._occupancy(w, f); rtol = 1e-12)
    end
    # compact sites (k copies within a few nm, sites > 60 nm apart): occupancy = covered sites
    pts = [(1.0 + 0.1 * i + 0.002 * c, 1.0 + 0.001 * c) for i in 0:9 for c in 0:(i % 4 + 1)]
    sd = points_smld([pts])
    site_of = [i for i in 0:9 for c in 0:(i % 4 + 1)]
    for covered in (0:9, [1, 4, 7])
        f = BitVector([i in covered for i in site_of])
        @test DC._occupancy(sd, f) ≈ length(covered)
    end
end

# The grid count is exact: same n_i as a KD-tree range count.
@testset "density counts match a KD-tree range count" begin
    for nd in (2, 3)
        Random.seed!(21)
        (s_c, _) = simulate(StaticSMLMConfig(2.0, 0.13, 50, 1, 300, 50.0, nd, [0.0, 1.0]);
            pattern  = nd == 2 ? Nmer2D(n = 4, d = 0.05) : Nmer3D(n = 4, d = 0.05),
            molecule = GenericFluor(; photons = 5000.0, k_on = 0.5, k_off = 50.0),
            camera   = IdealCamera(1:64, 1:64, 0.1))
        mat = permutedims(hcat([e.x for e in s_c.emitters], [e.y for e in s_c.emitters],
                               (nd == 3 ? [[e.z for e in s_c.emitters]] : [])...))
        ref = DC.inrangecount(DC.KDTree(mat), mat, DC._OCC_SEP)
        @test DC._count_within(DC._PopKDTree(s_c), DC._OCC_SEP, 1:length(s_c.emitters)) == ref
    end
end

# Identical inlier flags tie whatever the weights, and ties go to the CC: the guard accepts
# without needing the density weights (a 3D Gaussian aggregate made them cost seconds).
@testset "overlap guard: identical flags accept the CC" begin
    mk(n, seed) = (rng = Random.MersenneTwister(seed);
        BasicSMLD([Emitter3DFit{Float64}(2 + 0.01randn(rng), 2 + 0.01randn(rng), 2 + 0.01randn(rng),
                                         1000.0, 10.0, 0.01, 0.01, 0.01, 0.0, 0.0; frame = 1, dataset = 1)
                   for _ in 1:n], IdealCamera(1:64, 1:64, 0.1), 1, 1, Dict{String,Any}()))
    tgt = mk(2_000, 9)
    tree = DC._coord_tree(mk(1024, 10), 3)
    cand = fill(0.001, 3)
    @test DC._overlap_inliers(tgt, zeros(3), tree, DC._CC_VERIFY_RADIUS, 3) ==
          DC._overlap_inliers(tgt, cand, tree, DC._CC_VERIFY_RADIUS, 3)
    inter = zeros(3)
    @test DC._accept_if_better_overlap!(inter, cand, tgt, tree, 3)
    @test inter == cand
    # differing flags still decide on occupancy: a far incumbent loses to the CC, and a far CC to the incumbent
    @test DC._accept_if_better_overlap!(fill(5.0, 3), zeros(3), tgt, tree, 3)
    inter = zeros(3)
    @test !DC._accept_if_better_overlap!(inter, fill(5.0, 3), tgt, tree, 3)
    @test inter == zeros(3)
end
