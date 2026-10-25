# Cross-basin overlap decisions scored by occupancy, :fft and the registered tiers.
using Test, SMLMDriftCorrection, SMLMSim, Random

const DC = SMLMDriftCorrection

# -----------------------------------------------------------------------
# Cross-basin decisions are scored by occupancy, not by inlier count: a
# reference with 50 matching sites plus 4 copies of 30 others displaced by
# (3, 3) um. The capped CC proposes the wrong basin, which holds fewer of the
# target's sites (30 vs 80) but, by repeated localizations, more inlier
# localizations than the true shift.
# -----------------------------------------------------------------------
xbasin_smld(seed) = begin
    Random.seed!(seed)
    cam = IdealCamera(1:64, 1:64, 0.1)
    (s, _) = simulate(StaticSMLMConfig(2.5, 0.13, 50, 2, 300, 50.0, 2, [0.0, 1.0]);
        pattern  = Nmer2D(n = 1, d = 0.0),
        molecule = GenericFluor(; photons = 5.0e4, k_on = 1.0, k_off = 50.0), camera = cam)
    sites = sort(collect(intersect((Set(e.track_id for e in s.emitters if e.dataset == d)
                                    for d in 1:2)...)))[1:80]
    in_ds(d, tid) = [e for e in s.emitters if e.dataset == d && e.track_id == tid]
    ref = [first(in_ds(1, tid)) for tid in sites[1:50]]
    for tid in sites[51:80], e in in_ds(1, tid)[1:min(4, end)]
        r = deepcopy(e); r.x += 3.0; r.y += 3.0
        push!(ref, r)
    end
    tgt = [deepcopy(first(in_ds(2, tid))) for tid in sites]
    BasicSMLD(vcat(ref, tgt), cam, 300, 2, Dict{String,Any}())
end

@testset "cross-basin overlap decision" begin
    smld = xbasin_smld(3)
    (_, info) = DC.driftcorrect(smld; quality = :fft)
    @test hypot((info.model.inter[2].dm .- info.model.inter[1].dm)...) < 0.050
    # the guard behind it: the zero incumbent must veto the (-3, -3) candidate
    sd2 = DC.filter_by_dataset(smld, 2)
    tree1 = DC._coord_tree(DC.filter_by_dataset(smld, 1), 2)
    inter = [0.0, 0.0]
    @test !DC._accept_if_better_overlap!(inter, [-3.0, -3.0], sd2, tree1, 2)
    @test inter == [0.0, 0.0]
end

# End to end through the registered tiers: the seed choice and the arbiter score by
# occupancy too (seed 2 left 4.2 um in 0.2.9; seeds 2 and 3 without occupancy in
# the arbiter).
@testset "cross-basin overlap decision, registered tiers" begin
    for seed in (2, 3), q in (:singlepass, :iterative)
        smld = xbasin_smld(seed)
        kw = q == :iterative ? (; max_iterations = 4) : (;)
        (_, info) = DC.driftcorrect(smld; quality = q, degree = 0, kw...)
        @test hypot((info.model.inter[2].dm .- info.model.inter[1].dm)...) < 0.050
    end
end
