# Continuous chunk seeds: a hot spot in chunk 1 must not capture the boundary CC.
using Test, SMLMDriftCorrection, SMLMSim, Random

const DC = SMLMDriftCorrection

@testset "Continuous chunk seed: hot spot in chunk 1" begin
    site(x, n) = [Emitter2DFit{Float64}(x, x, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
                                      σ_xy = 0.0, track_id = i) for i in 1:n]

    @testset "seed=$seed" for seed in (1, 2, 3)
        Random.seed!(seed)
        cam = IdealCamera(1:64, 1:64, 0.1)
        fl = GenericFluor(; photons = 5000.0, k_on = 0.5, k_off = 50.0)
        cfg = StaticSMLMConfig(8.0, 0.13, 50, 1, 300, 50.0, 2, [0.0, 1.0])
        (s, _) = simulate(cfg; pattern = Nmer2D(n = 1, d = 0.0), molecule = fl,
                          camera = cam)
        (sm, _) = simulate(cfg; starting_conditions = site(1.2, 50), molecule = fl,
                           camera = cam)
        (hs, _) = simulate(cfg; starting_conditions = site(4.2, 300), molecule = fl,
                           camera = cam)
        structure = vcat(s.emitters, sm.emitters)

        @testset "truth=$truth" for truth in ([0.25, -0.15], [0.30, 0.20])
            target = deepcopy(structure)
            for e in target
                e.x += truth[1]
                e.y += truth[2]
                e.dataset = 2
            end
            smld = BasicSMLD(vcat(structure, hs.emitters, target), cam, 300, 2,
                             Dict{String,Any}())
            chunk1 = DC.filter_by_dataset(smld, 1)
            chunk2 = DC.filter_by_dataset(smld, 2)
            cc_raw = DC.findshift(chunk1, chunk2; histbinsize = 0.02)
            cc_cap = DC.findshift(chunk1, chunk2; histbinsize = 0.02,
                                  maxcount = DC._CC_MAXCOUNT)
            @test sqrt(sum(abs2, cc_raw .- truth)) > 1.0
            @test sqrt(sum(abs2, cc_cap .- truth)) < 0.010

            @testset "$quality" for quality in (:singlepass, :iterative)
                (_, info) = driftcorrect(smld; dataset_mode = :continuous, degree = 0,
                                         quality)
                relative = info.model.inter[2].dm .- info.model.inter[1].dm
                @test sqrt(sum(abs2, relative .- truth)) < 0.010
            end
        end
    end
end
