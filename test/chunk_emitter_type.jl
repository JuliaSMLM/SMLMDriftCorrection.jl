# Chunked continuous mode on emitter types outside SMLMData (review finding M2, 2026-11-02).
# GaussMLE's fit types (Emitter2DFitGaussMLE, Emitter2DFitSigma, Emitter3DFitGaussMLE, ...) are
# mutable, carry extra fields (pvalue, PSF width) and define only the default positional
# constructor. chunk_smld rebuilt every emitter through SMLMData's keyword constructor, so a
# chunked run on the fitter's own output threw a MethodError. The test types below copy that
# shape: an extra field and no keyword constructor.
using Test, SMLMDriftCorrection, SMLMSim, Random

const DC = SMLMDriftCorrection

mutable struct NoKwFit2D{T} <: DC.SMLMData.AbstractEmitter
    x::T; y::T; photons::T; bg::T
    σ_x::T; σ_y::T; σ_xy::T; σ_photons::T; σ_bg::T
    pvalue::T
    frame::Int; dataset::Int; track_id::Int; id::Int
end

mutable struct NoKwFit3D{T} <: DC.SMLMData.AbstractEmitter
    x::T; y::T; z::T; photons::T; bg::T
    σ_x::T; σ_y::T; σ_z::T; σ_photons::T; σ_bg::T
    pvalue::T
    frame::Int; dataset::Int; track_id::Int; id::Int
end

to_nokw(e::Emitter2DFit, i, n) = NoKwFit2D(e.x, e.y, e.photons, e.bg, e.σ_x, e.σ_y, e.σ_xy,
                                           e.σ_photons, e.σ_bg, i / n, e.frame, e.dataset, i, i)
to_nokw(e::Emitter3DFit, i, n) = NoKwFit3D(e.x, e.y, e.z, e.photons, e.bg, e.σ_x, e.σ_y, e.σ_z,
                                           e.σ_photons, e.σ_bg, i / n, e.frame, e.dataset, i, i)

function as_nokw(smld)
    n = length(smld.emitters)
    BasicSMLD([to_nokw(e, i, n) for (i, e) in enumerate(smld.emitters)], smld.camera,
              smld.n_frames, smld.n_datasets, copy(smld.metadata))
end

samefields(a, b; skip) = all(f -> f in skip || isequal(getfield(a, f), getfield(b, f)),
                             fieldnames(typeof(a)))

@testset "Chunked continuous on an emitter type without a keyword constructor" begin
    cam = IdealCamera(1:32, 1:32, 0.1)
    fl = GenericFluor(; photons = 5000.0, k_on = 0.5, k_off = 50.0)
    cases = (
        ("2D", StaticSMLMConfig(8.0, 0.13, 30, 1, 800, 50.0, 2, [0.0, 1.0]),
         Nmer2D(n = 4, d = 0.2), (; n_chunks = 4)),
        ("3D", StaticSMLMConfig(8.0, 0.13, 30, 1, 800, 50.0, 3, [-0.5, 0.5]),
         Nmer3D(n = 4, d = 0.2), (; chunk_frames = 200)),
    )
    @testset "$label" for (label, simcfg, pattern, chunking) in cases
        Random.seed!(7)
        (truth, _) = simulate(simcfg; pattern, molecule = fl, camera = cam)
        drift = DC.LegendrePolynomial(truth; degree = 2, initialize = "random", rscale = 0.1)
        smld_fit = DC.applydrift(truth, drift)
        smld_nokw = as_nokw(smld_fit)
        ET = eltype(smld_nokw.emitters)

        # chunk_smld relabels frame and dataset and keeps every other field and the type
        ci = DC.chunk_smld(smld_nokw; chunking...)
        @test ci.n_chunks == 4
        @test eltype(ci.smld.emitters) == ET
        @test all(samefields(a, b; skip = (:frame, :dataset))
                  for (a, b) in zip(ci.smld.emitters, smld_nokw.emitters))
        @test sort(unique(e.dataset for e in ci.smld.emitters)) == 1:4
        @test all(1 <= e.frame <= ci.last_chunk_frames for e in ci.smld.emitters)
        # the input is not modified
        @test all(samefields(a, b; skip = ()) for (a, b) in
                  zip(smld_nokw.emitters, as_nokw(smld_fit).emitters))

        # driftcorrect chunks in two places: a fresh run and a continuation from its DriftInfo
        cfg = DriftConfig(; dataset_mode = :continuous, degree = 2, chunking...)
        Random.seed!(1)
        (first_nokw, info_nokw) = driftcorrect(smld_nokw, cfg)
        Random.seed!(1)
        (first_fit, info_fit) = driftcorrect(smld_fit, cfg)
        Random.seed!(2)
        (cont_nokw, _) = driftcorrect(smld_nokw, info_nokw; max_iterations = 2)
        Random.seed!(2)
        (cont_fit, _) = driftcorrect(smld_fit, info_fit; max_iterations = 2)
        @testset "$step" for (step, out_nokw, out_fit) in (("fresh", first_nokw, first_fit),
                                                           ("continuation", cont_nokw, cont_fit))
            @test eltype(out_nokw.emitters) == ET
            # only the positions change; frame, dataset and the extra field come back as given
            @test all(samefields(a, b; skip = (:x, :y, :z))
                      for (a, b) in zip(out_nokw.emitters, smld_nokw.emitters))
            # the same localizations as SMLMData's type give the same correction
            pos(s) = label == "2D" ? [(e.x, e.y) for e in s.emitters] :
                                     [(e.x, e.y, e.z) for e in s.emitters]
            @test pos(out_nokw) == pos(out_fit)
            # and the correction removes the imposed drift (up to a global offset)
            avg(v) = sum(v) / length(v)
            function rms_vs_truth(s)
                dx = [a.x - t.x for (a, t) in zip(s.emitters, truth.emitters)]
                dy = [a.y - t.y for (a, t) in zip(s.emitters, truth.emitters)]
                sqrt(avg(abs2.(dx .- avg(dx))) + avg(abs2.(dy .- avg(dy))))
            end
            @test rms_vs_truth(out_nokw) < 0.05
            @test rms_vs_truth(out_nokw) < rms_vs_truth(smld_fit)
        end
    end
end
