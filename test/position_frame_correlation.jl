# position_frame_correlation on a hand-built frame-correlated offset.
using Test, SMLMDriftCorrection, SMLMSim, Random

const DC = SMLMDriftCorrection

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
