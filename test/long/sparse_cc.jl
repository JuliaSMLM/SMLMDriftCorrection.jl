# Sparse and quantized CC: sub-bin shifts, zero-overlap ties, thin z stack through :fft.
using Test, SMLMDriftCorrection, SMLMSim, Random

const DC = SMLMDriftCorrection

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

# 3D thin z stack (z range under 1 μm) through the :fft tier; the findshift part is
# in test/findshift.jl.
@testset "driftcorrect :fft: thin z stack" begin
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
    (_, info_z) = DC.driftcorrect(s_z; quality = :fft)
    @test sqrt(sum(abs2, info_z.model.inter[2].dm .- info_z.model.inter[1].dm .-
                         truth_z)) < 0.020
end
