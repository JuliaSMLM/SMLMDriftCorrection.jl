# Hot spots: CC-primary inter and :fft vs an aggregate present in some datasets, and the capped CC.
using Test, SMLMDriftCorrection, SMLMSim, Random

const DC = SMLMDriftCorrection

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
