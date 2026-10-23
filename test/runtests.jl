# runtests.jl — CI / `Pkg.test()` suite.
#
# Lightweight subset that exercises the public interface, package
# compatibility, edge-case guards, and deterministic-unit behavior. Designed
# to run in ~1-2 minutes and to be insensitive to thread scheduling / RNG
# differences across Julia versions.
#
# Stochastic SMLMSim-driven integration coverage (RMSD checks, multi-method
# comparisons, multi-iteration runs, auto-ROI, the noisy
# `position_frame_correlation` smoke) lives in `test/extended_tests.jl` and is
# NOT executed by default. To run the full suite locally:
#
#     julia --project test/extended_tests.jl

using SMLMDriftCorrection
using SMLMSim
using Test
using Random
using Logging

const DC = SMLMDriftCorrection

@testset "SMLMDriftCorrection.jl (CI)" begin
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

    # -----------------------------------------------------------------------
    # findshift smoke — identity on small data
    # -----------------------------------------------------------------------
    @testset "findshift smoke" begin
        @test isapprox(DC.findshift(smld_small, smld_small; histbinsize = 0.10),
                       [0.0, 0.0])
    end

    # -----------------------------------------------------------------------
    # findshift sub-bin refinement — two SMLMSim datasets of the same sites,
    # DS2 shifted by known sub-bin amounts (a 5×5 grid of bin phases). The
    # integer CC peak is within half a bin (25 nm) per axis, so the refinement
    # must land well inside that; a 7×7 log-quadratic fit was pulled up to
    # 3 bins off by pair counts around the peak.
    # -----------------------------------------------------------------------
    @testset "findshift sub-bin accuracy" begin
        Random.seed!(3)
        (smld_pair, _) = simulate(
            StaticSMLMConfig(5.0, 0.13, 30, 2, 300, 50.0, 2, [0.0, 1.0]);
            pattern  = Nmer2D(n = 1, d = 0.0),
            molecule = GenericFluor(; photons = 5000.0, k_on = 1.0, k_off = 50.0),
            camera   = IdealCamera(1:64, 1:64, 0.1))
        ref = DC.filter_by_dataset(smld_pair, 1)
        tgt0 = DC.filter_by_dataset(smld_pair, 2)
        err = Float64[]
        for fx in 0:0.2:0.8, fy in 0:0.2:0.8
            truth = [0.3 + 0.05 * fx, 0.4 + 0.05 * fy]
            tgt = deepcopy(tgt0)
            for e in tgt.emitters
                e.x += truth[1]; e.y += truth[2]
            end
            push!(err, maximum(abs.(DC.findshift(ref, tgt; histbinsize = 0.05) .- truth)))
        end
        @test maximum(err) < 0.015
    end

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
    # SMLMSim emitters repositioned at the given (x, y[, z]) points
    sim_points(nd, pts) = begin
        Random.seed!(5)
        (s_o, _) = simulate(StaticSMLMConfig(1.0, 0.13, 50, 1, 300, 50.0, nd, [-1.0, 1.0]);
            pattern  = nd == 2 ? Nmer2D(n = 1, d = 0.0) : Nmer3D(n = 1, d = 0.0),
            molecule = GenericFluor(; photons = 5000.0, k_on = 0.5, k_off = 50.0),
            camera   = IdealCamera(1:64, 1:64, 0.1))
        em = [deepcopy(s_o.emitters[1]) for _ in pts]
        for (e, p) in zip(em, pts)
            e.x = p[1]; e.y = p[2]
            nd == 3 && (e.z = p[3])
        end
        BasicSMLD(em, s_o.camera, 300, 1, Dict{String,Any}())
    end

    @testset "occupancy" begin
        # 4 sites 1 um apart, each localized 10 times within 3 nm: occupancy 4, count 40
        pts = [(i + 0.001 * j, 2.0 + 0.001 * j) for i in 1:4 for j in 0:9]
        sd = sim_points(2, pts)
        all_in = trues(length(pts))
        @test DC._occupancy(sd, all_in) == 4
        # two localizations 40 nm apart are two sites
        sd = sim_points(2, [(1.0, 1.0), (1.040, 1.0)])
        @test DC._occupancy(sd, trues(2)) == 2
        # only flagged localizations count
        sd = sim_points(2, [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (3.001, 1.0)])
        @test DC._occupancy(sd, BitVector([1, 0, 1, 1])) == 2
        @test DC._occupancy(sd, falses(4)) == 0
        # 3D uses z: same (x, y), 40 nm apart in z are two sites, 3 nm apart one
        sd = sim_points(3, [(1.0, 1.0, 0.0), (1.0, 1.0, 0.040)])
        @test DC._occupancy(sd, trues(2)) == 2
        sd = sim_points(3, [(1.0, 1.0, 0.0), (1.0, 1.0, 0.003)])
        @test DC._occupancy(sd, trues(2)) == 1
    end

    # -----------------------------------------------------------------------
    # Arbiter: the candidate with the largest score wins (occupancy in
    # _cc_primary_inter!); ties keep refined, then CC, then the incumbent.
    # -----------------------------------------------------------------------
    @testset "arbiter" begin
        # 10 sites of one localization each (indices 1:10), 6 more sites (11:16), and
        # one site localized 30 times (17:46)
        pts = vcat([(Float64(i), 1.0) for i in 1:16], [(20.0 + 0.001 * j, 1.0) for j in 1:30])
        sd = sim_points(2, pts)
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
    # _CC_VERIFY_RADIUS of it, the lowest entropy cost is kept (stub cost by shift x).
    @testset "arbiter: near ties go to the lowest entropy cost" begin
        flag(idx) = (f = falses(16); f[idx] .= true; f)
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
        # equal costs keep the order refined, CC, incumbent
        @test DC._arbitrate(ref, flag(1:5), cc, flag(1:8), inc, falses(16);
                            score = count, cost = cost_of(Dict(0.0 => 1.0, 0.015 => 1.0, 0.5 => 1.0))) === ref
        @test DC._arbitrate(ref, flag(1:3), cc, flag(1:8), [0.02, 0.0], falses(16);
                            score = count, cost = cost_of(Dict(0.0 => 1.0, 0.015 => 1.0, 0.02 => 1.0))) === ref
        # no cost call when the winner is alone in its neighborhood
        @test DC._arbitrate(ref, flag(1:5), cc, flag(1:4), inc, flag(1:9);
                            score = count, cost = _ -> error("cost evaluated")) === inc
    end

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

    # -----------------------------------------------------------------------
    # peak_offset_3pt: a zero neighbour is zero up to FFT roundoff (~1e-16 of the peak).
    # -----------------------------------------------------------------------
    @testset "peak_offset_3pt roundoff neighbour" begin
        @test DC.peak_offset_3pt(10.0, 90.0, 3.79e-16) == DC.peak_offset_3pt(10.0, 90.0, 0.0)
        @test DC.peak_offset_3pt(3.79e-16, 90.0, 10.0) == DC.peak_offset_3pt(0.0, 90.0, 10.0)
        @test DC.peak_offset_3pt(1.0, 0.0, 1.0) == 0.0
        @test isapprox(DC.peak_offset_3pt(10.0, 90.0, 0.0), -0.1; atol = 1e-12)
    end

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

    # -----------------------------------------------------------------------
    # position_frame_correlation deterministic unit
    #
    # Build a tiny SMLD with a KNOWN frame-correlated x-offset (linear in
    # frame number) and verify the diagnostic returns a strong positive
    # correlation in x and ~0 in y. Apply the inverse known offset and
    # verify x correlation drops materially. No driftcorrect, no random
    # init, no SMLMSim involvement.
    # -----------------------------------------------------------------------
    @testset "position_frame_correlation: deterministic unit" begin
        if isdefined(SMLMDriftCorrection, :position_frame_correlation)
            # Use simulate for a deterministic seeded baseline, then ADD a
            # known linear x-drift. Frame numbers come from the simulation;
            # the drift itself is hand-built and exact.
            Random.seed!(7)
            params_pfc = StaticSMLMConfig(20.0, 0.13, 30, 1, 200, 50.0, 2, [0.0, 1.0])
            (smld_pfc, _) = simulate(params_pfc;
                pattern  = Nmer2D(n = 4, d = 0.2),
                molecule = GenericFluor(; photons = 5000.0, k_on = 0.05, k_off = 50.0),
                camera   = IdealCamera(1:32, 1:32, 0.1))

            # Hand-built linear drift in x only: c1 * P1(t) where t ∈ [-1,1].
            # Magnitude (±300 nm at endpoints) is large vs inter-emitter spacing
            # (~200 nm at density 20/μm² in this FOV), so the drift actually
            # breaks local spatial clusters — without that the diagnostic
            # correctly sees no change between drifted and (exactly) corrected.
            poly_x = DC.LegendrePoly1D(1, smld_pfc.n_frames)
            poly_x.coefficients .= [0.3]   # ±300 nm at endpoints, 0 at midpoint

            smld_drifted = deepcopy(smld_pfc)
            for e in smld_drifted.emitters
                e.x += DC.evaluate_at_frame(poly_x, e.frame)
                # y unchanged
            end

            res_d = DC.position_frame_correlation(smld_drifted; K = 20, mode = :intra)

            # Shape / range checks
            @test res_d.mode == :intra
            @test res_d.K == 20
            @test 0.0 <= res_d.summary.mean_abs_corr_x <= 1.0
            @test 0.0 <= res_d.summary.mean_abs_corr_y <= 1.0
            @test res_d.summary.mean_abs_corr_z === nothing

            # Apply EXACT inverse drift — x residual correlation should drop
            # essentially to where y already sits (no drift in y).
            smld_corrected = deepcopy(smld_drifted)
            for e in smld_corrected.emitters
                e.x -= DC.evaluate_at_frame(poly_x, e.frame)
            end
            res_c = DC.position_frame_correlation(smld_corrected; K = 20, mode = :intra)

            # The deterministic test: applying the EXACT inverse drift must
            # reduce x-corr (compared to drifted) AND not significantly raise
            # y-corr (since y was untouched). These are properties of the
            # diagnostic itself, not of any optimizer.
            @test res_c.summary.mean_abs_corr_x < res_d.summary.mean_abs_corr_x
            @test res_c.summary.mean_abs_corr_y ≈ res_d.summary.mean_abs_corr_y atol = 0.05
        end
    end

    # -----------------------------------------------------------------------
    # driftcorrect interface smoke — verify return type / DriftInfo fields.
    # Uses tiny multi-dataset data so inter alignment runs. Does NOT assert
    # noisy RMSD values that would be sensitive to thread/RNG variance.
    # -----------------------------------------------------------------------
    @testset "driftcorrect interface smoke" begin
        params_3ds = StaticSMLMConfig(5.0, 0.13, 30, 3, 200, 50.0, 2, [0.0, 1.0])
        (smld_test, _) = simulate(params_3ds;
            pattern  = Nmer2D(n = 4, d = 0.2),
            molecule = GenericFluor(; photons = 5000.0, k_on = 0.05, k_off = 50.0),
            camera   = IdealCamera(1:32, 1:32, 0.1))
        Random.seed!(123)
        model = DC.LegendrePolynomial(smld_test; degree = 2,
                                       initialize = "random", rscale = 0.1)
        model.inter[1].dm .= 0.0
        smld_drifted = DC.applydrift(smld_test, model)

        # Singlepass — default
        (smld_c, info) = DC.driftcorrect(smld_drifted)
        @test smld_c isa DC.SMLD
        @test info isa DC.DriftInfo
        @test info isa DC.AbstractSMLMInfo
        @test info.model isa DC.LegendrePolynomial
        @test info.elapsed_s > 0
        @test info.backend == :cpu
        @test info.iterations == 1                       # singlepass
        @test info.converged == true
        @test info.entropy isa Float64
        @test info.history isa Vector{Float64}

        # Config dispatch
        cfg = DC.DriftConfig(; quality = :singlepass, degree = 2)
        (smld_c2, info_c2) = DC.driftcorrect(smld_drifted, cfg)
        @test smld_c2 isa DC.SMLD
        @test info_c2 isa DC.DriftInfo
        @test info_c2.iterations == 1

        # FFT — quick path, no optimization, deterministic
        (smld_fft, info_fft) = DC.driftcorrect(smld_drifted; quality = :fft)
        @test info_fft.iterations == 0
        @test info_fft.converged == true

        # Warm-start path returns valid info
        (smld_w, info_w) = DC.driftcorrect(smld_drifted; warm_start = info.model)
        @test info_w isa DC.DriftInfo
        @test info_w.elapsed_s > 0
    end

    # -----------------------------------------------------------------------
    # Continuous mode smoke — exercises chunking + the CC-seed inter path
    # (_ccseed_inter_continuous!). Tiny data, so assert only that it runs and
    # returns valid output; full-sim drift recovery lives in extended_tests.jl.
    # -----------------------------------------------------------------------
    @testset "continuous mode smoke" begin
        Random.seed!(7)
        cmodel = DC.LegendrePolynomial(smld_small; degree = 2,
                                        initialize = "random", rscale = 0.05)
        cmodel.inter[1].dm .= 0.0
        smld_cdrift = DC.applydrift(smld_small, cmodel)

        (sc, cinfo) = DC.driftcorrect(smld_cdrift; dataset_mode = :continuous,
                                       n_chunks = 2, degree = 2)
        @test sc isa DC.SMLD
        @test cinfo isa DC.DriftInfo
        @test cinfo.model isa DC.LegendrePolynomial
        @test length(sc.emitters) == length(smld_cdrift.emitters)
        @test cinfo.elapsed_s > 0

        # _ccseed_inter_continuous! runs without error on a 2-chunk model and
        # produces one inter shift per chunk.
        ci = DC.chunk_smld(smld_small; n_chunks = 2)
        m2 = DC.LegendrePolynomial(ci.smld; degree = 2)
        @test_nowarn DC._ccseed_inter_continuous!(m2, ci.smld)
        @test length(m2.inter) == ci.smld.n_datasets
    end

    # -----------------------------------------------------------------------
    # Registered CC-primary inter and the :fft tier vs a hot spot (0.2.8
    # regression). A bright aggregate (300 fluorophores at one site) is imaged
    # only in DS1-2, so the unnormalized histogram CC of DS3-4 against DS1 is
    # captured by it: the peak lands where each dataset's densest cluster
    # overlays the hot spot, microns away. 0.2.8 accepted that peak in both
    # paths (1.3-3.4 um across seeds); the overlap guard must keep the
    # incumbent instead. All datasets are mutually aligned (truth: every inter
    # shift is 0).
    # -----------------------------------------------------------------------
    @testset "Hot spot in early datasets: CC-primary inter + :fft (registered)" begin
        Random.seed!(2026)
        cam_hs = IdealCamera(1:64, 1:64, 0.1)                  # 6.4 μm FOV
        fluor_hs = GenericFluor(; photons = 5000.0, k_on = 1.0, k_off = 50.0)
        (s_struct, _) = simulate(StaticSMLMConfig(5.0, 0.13, 50, 4, 300, 50.0, 2, [0.0, 1.0]);
            pattern = Nmer2D(n = 4, d = 0.2), molecule = fluor_hs, camera = cam_hs)
        agg = [Emitter2DFit{Float64}(3.2, 3.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
                                     σ_xy = 0.0, track_id = i) for i in 1:300]
        (s_hot, _) = simulate(StaticSMLMConfig(1.0, 0.13, 50, 2, 300, 50.0, 2, [0.0, 1.0]);
            starting_conditions = agg, molecule = fluor_hs, camera = cam_hs)
        smld_hs = BasicSMLD(vcat(s_struct.emitters, s_hot.emitters), cam_hs, 300, 4,
                            Dict{String,Any}())
        model_hs = DC.LegendrePolynomial(smld_hs; degree = 2)  # zeros: intra = 0, inter = 0
        DC._cc_primary_inter!(model_hs, smld_hs, 100; n_passes = 2, init = true)
        @test all(hypot(model_hs.inter[d].dm...) < 0.03 for d in 1:4)

        (_, info_fft_hs) = DC.driftcorrect(smld_hs; quality = :fft)
        @test all(hypot(info_fft_hs.model.inter[d].dm...) < 0.03 for d in 1:4)

        # A zero incumbent chosen as the refine seed must be refined from, not replaced
        # by findinter!'s own (hot-spot-captured) CC: DS3 displaced by a true 20 nm.
        smld_hs20 = deepcopy(smld_hs)
        for e in smld_hs20.emitters
            e.dataset == 3 && (e.x += 0.020)
        end
        model_hs20 = DC.LegendrePolynomial(smld_hs20; degree = 2)
        DC._cc_primary_inter!(model_hs20, smld_hs20, 100; n_passes = 2, init = true)
        @test isapprox(model_hs20.inter[3].dm[1], 0.020; atol = 0.006)
    end

    # -----------------------------------------------------------------------
    # Sub-bin true shift on sparse data (:fft, 2D and 3D): the 50 nm CC is
    # quantized, and the overlap guard must still accept it over a far
    # incumbent (a guard that let the incumbent borrow a neighboring position's
    # score left 57 / 69 nm here).
    # -----------------------------------------------------------------------
    @testset "Overlap guard accepts a quantized CC (sub-bin shift, sparse)" begin
        cam_q = IdealCamera(1:64, 1:64, 0.1)
        for nd in (2, 3)
            Random.seed!(1)
            (s_t, _) = simulate(StaticSMLMConfig(1.0, 0.13, 50, 1, 300, 50.0, nd, [-1.0, 1.0]);
                pattern = nd == 2 ? Nmer2D(n = 1, d = 0.0) : Nmer3D(n = 1, d = 0.0),
                molecule = GenericFluor(; photons = 5000.0, k_on = 0.5, k_off = 50.0),
                camera = cam_q)
            truth_t = fill(0.040, nd)
            e_t = deepcopy(s_t.emitters)
            for e in e_t
                e.x += truth_t[1]
                e.y += truth_t[2]
                nd == 3 && (e.z += truth_t[3])
                e.dataset = 2
            end
            smld_t = BasicSMLD(vcat(s_t.emitters, e_t), cam_q, 300, 2, Dict{String,Any}())
            (_, info_t) = DC.driftcorrect(smld_t; quality = :fft)
            @test sqrt(sum(abs2, info_t.model.inter[2].dm .- info_t.model.inter[1].dm .-
                                 truth_t)) < 0.025
        end
    end

    # -----------------------------------------------------------------------
    # Truth 35 nm from the zero incumbent (just past the 30 nm radius) and a
    # hot spot only in the reference that, in a raw-count CC, captures the
    # peak onto the target's small aggregate (0.2.8: 4.2 um). The capped CC
    # must not be captured. The :fft tier has no entropy refine, so it is held
    # to the CC's bin-level precision.
    # -----------------------------------------------------------------------
    @testset "CC-primary inter: incumbent just past the radius vs captured CC" begin
        Random.seed!(1)
        cam_r = IdealCamera(1:64, 1:64, 0.1)
        fl_r = GenericFluor(; photons = 5000.0, k_on = 0.5, k_off = 50.0)
        cfg_r = StaticSMLMConfig(8.0, 0.13, 50, 1, 300, 50.0, 2, [0.0, 1.0])
        (s_r, _) = simulate(cfg_r; pattern = Nmer2D(n = 1, d = 0.0), molecule = fl_r,
                            camera = cam_r)
        site(x, n) = [Emitter2DFit{Float64}(x, x, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
                                            σ_xy = 0.0, track_id = i) for i in 1:n]
        (small_r, _) = simulate(cfg_r; starting_conditions = site(1.2, 50), molecule = fl_r,
                                camera = cam_r)
        (hot_r, _) = simulate(cfg_r; starting_conditions = site(4.2, 300), molecule = fl_r,
                              camera = cam_r)
        structure_r = vcat(s_r.emitters, small_r.emitters)
        truth_r = [0.025, 0.025]
        target_r = deepcopy(structure_r)
        for e in target_r
            e.x += truth_r[1]
            e.y += truth_r[2]
            e.dataset = 2
        end
        smld_r = BasicSMLD(vcat(structure_r, hot_r.emitters, target_r), cam_r, 300, 2,
                           Dict{String,Any}())
        model_r = DC.LegendrePolynomial(smld_r; degree = 2)
        DC._cc_primary_inter!(model_r, smld_r, 100; n_passes = 2, init = true)
        @test isapprox(model_r.inter[2].dm .- model_r.inter[1].dm, truth_r; atol = 0.010)
        (_, info_r) = DC.driftcorrect(smld_r; quality = :fft)
        @test isapprox(info_r.model.inter[2].dm .- info_r.model.inter[1].dm, truth_r;
                       atol = 0.030)

        # The cap itself: the reference-only hot spot captures a raw-count CC
        # (microns off) but not the capped one.
        ref_r = DC.filter_by_dataset(smld_r, 1)
        tgt_r = DC.filter_by_dataset(smld_r, 2)
        cc_raw = DC.findshift(ref_r, tgt_r; histbinsize = 0.05)
        cc_cap = DC.findshift(ref_r, tgt_r; histbinsize = 0.05, maxcount = DC._CC_MAXCOUNT)
        @test sqrt(sum(abs2, cc_raw .- truth_r)) > 1.0
        @test sqrt(sum(abs2, cc_cap .- truth_r)) < 0.030
        @test_throws ArgumentError DC.findshift(ref_r, tgt_r; histbinsize = 0.05, maxcount = 0)
        @test_throws ArgumentError DC.findshift_damped(ref_r, tgt_r; histbinsize = 0.05, maxcount = -1)
    end

    # -----------------------------------------------------------------------
    # Zero-overlap tie: sparse isolated sites with near-noiseless localizations
    # and a true shift near half a 50 nm bin. The CC is correct but ~35-40 nm
    # off, so it and the far incumbent both score zero overlap within the 30 nm
    # radius; the guard must give the tie to the CC (it kept the incumbent and
    # ended 213 nm off). Built from SMLMSim with very high photon counts.
    # -----------------------------------------------------------------------
    @testset "Overlap guard: zero-overlap tie goes to the CC" begin
        rng_t = MersenneTwister(3)
        sites_t = [Tuple(0.05 .* rand(rng_t, 12:110, 3)) for _ in 1:15]
        em_t = [Emitter3DFit{Float64}(p[1], p[2], p[3] - 2.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
                                      track_id = i) for (i, p) in enumerate(sites_t)]
        Random.seed!(3)
        (s_t, _) = simulate(StaticSMLMConfig(1.0, 0.13, 50, 2, 300, 50.0, 3, [-1.0, 1.0]);
            starting_conditions = em_t,
            molecule = GenericFluor(; photons = 1.0e7, k_on = 0.4, k_off = 50.0),
            camera = IdealCamera(1:64, 1:64, 0.1))
        truth_t = fill(0.123, 3)
        for e in s_t.emitters
            if e.dataset == 2
                e.x += truth_t[1]
                e.y += truth_t[2]
                e.z += truth_t[3]
            end
        end
        (_, info_t) = DC.driftcorrect(s_t; quality = :fft)
        @test sqrt(sum(abs2, info_t.model.inter[2].dm .- info_t.model.inter[1].dm .- truth_t)) < 0.050
    end

    # -----------------------------------------------------------------------
    # 3D cross-correlation on a thin stack (z range under 1 μm): the z ROI was
    # round(min z)..round(max z) of smld1 alone, so a stack within ±0.5 μm got a
    # zero-depth histogram and findshift threw (FFTW plan error), taking the
    # :fft tier down with it.
    # -----------------------------------------------------------------------
    @testset "findshift 3D: thin z stack" begin
        Random.seed!(1)
        (s_z, _) = simulate(StaticSMLMConfig(5.0, 0.13, 50, 2, 300, 50.0, 3, [-0.1, 0.1]);
            pattern = Nmer3D(n = 6, d = 0.2),
            molecule = GenericFluor(; photons = 5000.0, k_on = 0.5, k_off = 50.0),
            camera = IdealCamera(1:64, 1:64, 0.1))
        truth_z = [0.10, -0.05, 0.05]
        for e in s_z.emitters
            if e.dataset == 2
                e.x += truth_z[1]
                e.y += truth_z[2]
                e.z += truth_z[3]
            end
        end
        @test maximum(abs(e.z) for e in s_z.emitters) < 0.5          # the thin-stack case
        sh_z = DC.findshift(DC.filter_by_dataset(s_z, 1), DC.filter_by_dataset(s_z, 2);
                            histbinsize = 0.05)
        @test sqrt(sum(abs2, sh_z .- truth_z)) < 0.020
        (_, info_z) = DC.driftcorrect(s_z; quality = :fft)
        @test sqrt(sum(abs2, info_z.model.inter[2].dm .- info_z.model.inter[1].dm .-
                             truth_z)) < 0.020
    end

    # -----------------------------------------------------------------------
    # align_smld smoke — known shift recovery on small data.
    # -----------------------------------------------------------------------
    @testset "align_smld smoke" begin
        # Use the dataset=1 subset as an "independent acquisition" baseline
        # (smld_small is single-dataset already)
        smld_base = smld_small
        true_shift = [0.3, -0.2]
        smld_shifted = deepcopy(smld_base)
        for e in smld_shifted.emitters
            e.x += true_shift[1]
            e.y += true_shift[2]
        end
        (aligned, info) = DC.align_smld([smld_base, smld_shifted]; method = :fft)
        @test info isa DC.AlignInfo
        @test length(aligned) == 2
        @test info.shifts[1] == [0.0, 0.0]
        @test info.transforms[1] isa DC.ShiftTransform
        # FFT recovery within 150 nm — generous tolerance for tiny data
        @test isapprox(info.shifts[2], true_shift; atol = 0.150)
    end

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
end
