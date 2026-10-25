# findshift on small inputs: identity, sub-bin accuracy, thin z stack, peak_offset_3pt.
using Test, SMLMDriftCorrection, SMLMSim, Random

const DC = SMLMDriftCorrection

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
# peak_offset_3pt: a zero neighbour is zero up to FFT roundoff (~1e-16 of the peak).
# -----------------------------------------------------------------------
@testset "peak_offset_3pt roundoff neighbour" begin
    @test DC.peak_offset_3pt(10.0, 90.0, 3.79e-16) == DC.peak_offset_3pt(10.0, 90.0, 0.0)
    @test DC.peak_offset_3pt(3.79e-16, 90.0, 10.0) == DC.peak_offset_3pt(0.0, 90.0, 10.0)
    @test DC.peak_offset_3pt(1.0, 0.0, 1.0) == 0.0
    @test isapprox(DC.peak_offset_3pt(10.0, 90.0, 0.0), -0.1; atol = 1e-12)
end

# 3D cross-correlation on a thin stack (z range under 1 μm): the z ROI was
# round(min z)..round(max z) of smld1 alone, so a stack within ±0.5 μm got a
# zero-depth histogram and findshift threw (FFTW plan error). The :fft tier
# end to end is in test/long/sparse_cc.jl.
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
end
