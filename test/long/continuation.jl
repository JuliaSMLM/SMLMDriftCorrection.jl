using Test, SMLMDriftCorrection, SMLMSim
using Random

const DC = SMLMDriftCorrection

Random.seed!(42)
(smld, _) = simulate(StaticSMLMConfig(5.0, 0.13, 30, 2, 200, 50.0, 2, [0.0, 1.0]);
    pattern  = Nmer2D(n = 4, d = 0.2),
    molecule = GenericFluor(; photons = 5000.0, k_on = 0.05, k_off = 50.0),
    camera   = IdealCamera(1:32, 1:32, 0.1))

@testset "continuous continuation preserves chunking" begin
    for chunking in ((n_chunks = 3,), (chunk_frames = 70,))
        @testset "$chunking" begin
            (corrected, info) = DC.driftcorrect(smld; dataset_mode = :continuous,
                degree = 1, maxn = 20, chunking...)
            @test info.dataset_mode == :continuous
            @test info.n_chunks == get(chunking, :n_chunks, 0)
            @test info.chunk_frames == get(chunking, :chunk_frames, 0)
            @test info.model.ndatasets == 3 * smld.n_datasets

            (continued, new_info) = @test_logs (:info, r"continuous-boundary max gap") match_mode = :any DC.driftcorrect(
                corrected, info; max_iterations = 1, maxn = 20, verbose = 1)
            @test new_info.dataset_mode == :continuous
            @test new_info.n_chunks == info.n_chunks
            @test new_info.chunk_frames == info.chunk_frames
            @test new_info.model.ndatasets == info.model.ndatasets
            @test new_info.model.n_frames == info.model.n_frames
            @test new_info.iterations == info.iterations + 1
            @test continued.n_datasets == smld.n_datasets
            @test continued.n_frames == smld.n_frames
            @test [e.dataset for e in continued.emitters] == [e.dataset for e in smld.emitters]
            @test [e.frame for e in continued.emitters] == [e.frame for e in smld.emitters]
            @test all(e -> isfinite(e.x) && isfinite(e.y), continued.emitters)

            @test_throws ArgumentError DC.driftcorrect(corrected, info; dataset_mode = :registered)
            @test_throws ArgumentError DC.driftcorrect(corrected, info; n_chunks = 0, chunk_frames = 0)
        end
    end
end

@testset "registered continuation" begin
    (corrected, info) = DC.driftcorrect(smld; degree = 1, maxn = 20)
    (continued, new_info) = DC.driftcorrect(corrected, info; max_iterations = 1, maxn = 20)
    @test new_info.dataset_mode == :registered
    @test new_info.n_chunks == 0
    @test new_info.chunk_frames == 0
    @test new_info.iterations == info.iterations + 1
    @test continued.n_datasets == smld.n_datasets
    @test continued.n_frames == smld.n_frames
    @test length(continued.emitters) == length(smld.emitters)
    @test all(e -> isfinite(e.x) && isfinite(e.y), continued.emitters)
    @test_throws ArgumentError DC.driftcorrect(corrected, info; dataset_mode = :unknown)
end
