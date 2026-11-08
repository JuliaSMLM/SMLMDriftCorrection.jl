# Independent alignment: a hot spot only in the reference must not capture the CC.
using Test, SMLMDriftCorrection, SMLMSim, Random

const DC = SMLMDriftCorrection

@testset "align_smld: hot spot in reference" begin
    site(x, n) = [Emitter2DFit{Float64}(x, x, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
                                      σ_xy = 0.0, track_id = i) for i in 1:n]

    @testset "seed=$seed" for seed in (1, 2)
        Random.seed!(seed)
        cam = IdealCamera(1:128, 1:128, 0.1)
        fl = GenericFluor(; photons = 5000.0, k_on = 0.5, k_off = 50.0)
        cfg = StaticSMLMConfig(12.0, 0.13, 50, 1, 300, 50.0, 2, [0.0, 1.0])
        (s, _) = simulate(cfg; pattern = Nmer2D(n = 1, d = 0.0), molecule = fl,
                          camera = cam)
        (sm, _) = simulate(cfg; starting_conditions = site(3.4, 100), molecule = fl,
                           camera = cam)
        (hs, _) = simulate(cfg; starting_conditions = site(6.4, 600), molecule = fl,
                           camera = cam)
        structure = vcat(s.emitters, sm.emitters)
        reference = BasicSMLD(vcat(structure, hs.emitters), cam, 300, 1,
                              Dict{String,Any}())

        @testset "truth=$truth" for truth in ([0.25, -0.15], [0.30, 0.20])
            target_emitters = deepcopy(structure)
            for e in target_emitters
                e.x += truth[1]
                e.y += truth[2]
            end
            target = BasicSMLD(target_emitters, cam, 300, 1, Dict{String,Any}())
            cc_raw = DC.findshift(reference, target; histbinsize = 0.05)
            cc_cap = DC.findshift(reference, target; histbinsize = 0.05,
                                  maxcount = DC._CC_MAXCOUNT)
            @test sqrt(sum(abs2, cc_raw .- truth)) > 1.0
            @test maximum(abs.(cc_raw)) < DC._CC_MAX_SHIFT
            @test sqrt(sum(abs2, cc_cap .- truth)) < 0.010

            @testset "method=$method" for method in (:fft, :entropy)
                (_, info) = align_smld([reference, target]; method)
                @test sqrt(sum(abs2, info.shifts[2] .- truth)) < 0.010
            end
            @testset "transform=:affine" begin
                (_, info) = align_smld([reference, target]; transform = :affine)
                @test sqrt(sum(abs2, info.shifts[2] .- truth)) < 0.010
            end

            if seed == 1 && truth == [0.25, -0.15]
                @testset "align_smld transform=:affine with Float32 localizations" begin
                    ref32, tgt32 = map((reference, target)) do smld
                        emitters = [Emitter2DFit{Float32}(
                            Float32(e.x), Float32(e.y), Float32(e.photons), Float32(e.bg),
                            Float32(e.σ_x), Float32(e.σ_y), Float32(e.σ_photons), Float32(e.σ_bg);
                            σ_xy = Float32(e.σ_xy), frame = e.frame, dataset = e.dataset,
                            track_id = e.track_id, id = e.id) for e in smld.emitters]
                        BasicSMLD(emitters, smld.camera, smld.n_frames, smld.n_datasets,
                                  copy(smld.metadata))
                    end
                    (_, info) = align_smld([ref32, tgt32]; transform = :affine)
                    @test sqrt(sum(abs2, info.shifts[2] .- truth)) < 0.010
                end
            end
        end
    end
end
