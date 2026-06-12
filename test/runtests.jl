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
