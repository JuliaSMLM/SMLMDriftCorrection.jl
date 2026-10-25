# Stage-move (dropped CC over 5 um) warnings.
using Test, SMLMDriftCorrection, SMLMSim, Random, Logging

const DC = SMLMDriftCorrection

# -----------------------------------------------------------------------
# A CC over 5 um is never applied; when it would have overlapped clearly better
# than the kept shift (a stage that moved), driftcorrect warns once per dataset.
# -----------------------------------------------------------------------
@testset "stage-move warning" begin
    Random.seed!(7)
    (s_big, _) = simulate(StaticSMLMConfig(0.2, 0.13, 50, 3, 300, 50.0, 2, [0.0, 1.0]);
        pattern  = Nmer2D(n = 1, d = 0.0),
        molecule = GenericFluor(; photons = 2.0e4, k_on = 1.0, k_off = 50.0),
        camera   = IdealCamera(1:200, 1:200, 0.1))
    moved = deepcopy(s_big)
    for e in moved.emitters
        e.dataset == 3 && (e.x += 10.0)
    end
    @test_logs (:warn, r"dataset 3: .*likely a collection failure") match_mode = :any DC.driftcorrect(
        moved; quality = :singlepass, degree = 1)
    # ordinary small inter shift: nothing to warn about
    for e in s_big.emitters
        e.dataset == 3 && (e.x += 0.1)
    end
    @test_logs min_level = Logging.Warn DC.driftcorrect(s_big; quality = :singlepass, degree = 1)
end

@testset "stage-move warning, once per dataset (:iterative)" begin
    Random.seed!(7)
    (s_big, _) = simulate(StaticSMLMConfig(0.2, 0.13, 50, 3, 300, 50.0, 2, [0.0, 1.0]);
        pattern  = Nmer2D(n = 1, d = 0.0),
        molecule = GenericFluor(; photons = 2.0e4, k_on = 1.0, k_off = 50.0),
        camera   = IdealCamera(1:200, 1:200, 0.1))
    for e in s_big.emitters
        e.dataset == 3 && (e.x += 10.0)
    end
    (logs, res) = Test.collect_test_logs(; min_level = Logging.Warn) do
        DC.driftcorrect(s_big; quality = :iterative, degree = 1, max_iterations = 3)
    end
    moves = filter(l -> occursin("likely a collection failure", l.message), logs)
    @test length(moves) == 1
    @test occursin("dataset 3:", moves[1].message)
    # the 10 um CC was reported, not applied
    @test maximum(abs.(res[2].model.inter[3].dm .- res[2].model.inter[1].dm)) < 5.0
end

# The dropped CC was measured against an earlier consensus; the check also scores a
# fresh CC against the final one. A stale cached shift that overlaps nothing must not
# hide a displaced dataset.
@testset "stage-move warning scores a fresh CC" begin
    Random.seed!(7)
    (s_big, _) = simulate(StaticSMLMConfig(0.2, 0.13, 50, 3, 300, 50.0, 2, [0.0, 1.0]);
        pattern  = Nmer2D(n = 1, d = 0.0),
        molecule = GenericFluor(; photons = 2.0e4, k_on = 1.0, k_off = 50.0),
        camera   = IdealCamera(1:200, 1:200, 0.1))
    for e in s_big.emitters
        e.dataset == 3 && (e.x += 10.0)
    end
    model = DC.LegendrePolynomial(s_big; degree = 0)
    dropped = Vector{Union{Nothing,Vector{Float64}}}(nothing, 3)
    dropped[3] = [0.0, 7.0]    # stale: overlaps nothing at the final consensus
    warned = fill(false, 3)
    @test_logs (:warn, r"dataset 3: .*likely a collection failure") match_mode = :any DC._warn_dropped_cc!(
        model, s_big, dropped, warned)
    @test warned == [false, false, true]
end

# :fft applies the same rule: a CC over 5 um is not applied and, where it overlaps clearly
# better than the kept shift, one warning per dataset names it.
@testset "stage-move: :fft applies no shift over 5 um" begin
    @test !DC._cc_too_far([5.0, -5.0])
    @test !DC._cc_too_far([4.0, 4.0, 4.0])
    @test DC._cc_too_far([0.0, nextfloat(5.0), 0.0])
    @test DC._cc_too_far([-nextfloat(5.0), 0.0])

    Random.seed!(7)
    (s_big, _) = simulate(StaticSMLMConfig(0.2, 0.13, 50, 3, 300, 50.0, 2, [0.0, 1.0]);
        pattern  = Nmer2D(n = 1, d = 0.0),
        molecule = GenericFluor(; photons = 2.0e4, k_on = 1.0, k_off = 50.0),
        camera   = IdealCamera(1:200, 1:200, 0.1))
    moved = deepcopy(s_big)
    for e in moved.emitters
        e.dataset == 3 && (e.x += 10.0)
    end
    res = @test_logs (:warn, r"dataset 3: .*likely a collection failure") min_level = Logging.Warn DC.driftcorrect(
        moved; quality = :fft)
    @test maximum(abs.(res[2].model.inter[3].dm .- res[2].model.inter[1].dm)) <= DC._CC_MAX_SHIFT
    # ordinary small inter shift: applied, nothing to warn about
    for e in s_big.emitters
        e.dataset == 3 && (e.x += 0.1)
    end
    res = @test_logs min_level = Logging.Warn DC.driftcorrect(s_big; quality = :fft)
    @test isapprox(res[2].model.inter[3].dm[1] - res[2].model.inter[1].dm[1], 0.1; atol = 0.02)
end
