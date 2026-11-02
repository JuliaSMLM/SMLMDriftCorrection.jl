# Fail one CC at a time, including damped histograms and floating-point affine tiles.
using Test, SMLMDriftCorrection, SMLMSim, Random, Logging

const DC = SMLMDriftCorrection

function with_broken_cc(body; fail_at = 0, err = MethodError(DC.FourierTools.ccorr, ()))
    cc = (calls = Threads.Atomic{Int}(0), fail_at = Threads.Atomic{Int}(fail_at),
          err = Ref{Any}(err))
    @eval function DC.FourierTools.ccorr(a::Array{<:Real,N}, b::Array{<:Real,N}, args...; kw...) where {N}
        cc = $cc
        k = Threads.atomic_add!(cc.calls, 1) + 1
        k == cc.fail_at[] && throw(cc.err[])
        # Supply default dimensions explicitly: the original two-argument wrapper
        # would dispatch back here and count the same CC twice.
        pos = isempty(args) ? (ntuple(identity, N),) : args
        types = Tuple{AbstractArray{eltype(a),N},AbstractArray{eltype(b),N},map(typeof, pos)...}
        return invoke(DC.FourierTools.ccorr, types, a, b, pos...; kw...)
    end
    method = which(DC.FourierTools.ccorr, Tuple{Matrix{Float64},Matrix{Float64}})
    kwmethod = which(Core.kwcall,
        Tuple{typeof((centered = true,)),typeof(DC.FourierTools.ccorr),Matrix{Float64},Matrix{Float64}})
    try
        return Base.invokelatest(body, cc)
    finally
        Base.delete_method(kwmethod)
        Base.delete_method(method)
    end
end

function root_cause(e)
    e isa TaskFailedException && return root_cause(e.task.exception)
    e isa CompositeException && return root_cause(first(e.exceptions))
    return e
end

function failure_root(body)
    try
        body()
        return nothing
    catch e
        return root_cause(e)
    end
end

function reset_cc!(cc, fail_at)
    cc.calls[] = 0
    cc.fail_at[] = fail_at
    Random.seed!(42)
end

function check_cc_sites(body, cc; sample = false)
    reset_cc!(cc, 0)
    body()
    n = cc.calls[]
    @test n > 0
    sites = sample ? unique([1, 2, n]) : (1:n)
    lost = Int[]
    for k in sites
        reset_cc!(cc, k)
        failure_root(body) === cc.err[] || push!(lost, k)
    end
    @test isempty(lost)
    return n
end

# Identical, separated sites make entropy refinement stationary. Positive-degree
# fits use zero warm starts and midpoint frames, with no random intra initialization.
function cc_fixture(points, shifts; camera = IdealCamera(1:12, 1:12, 0.1),
                    frames = (2,), n_frames = 3)
    nd = length(first(points))
    em = [nd == 2 ?
        Emitter2DFit{Float64}(p[1] + s[1], p[2] + s[2], 1000.0, 10.0,
            0.01, 0.01, 0.0, 0.0; σ_xy = 0.0, frame = f, dataset = d) :
        Emitter3DFit{Float64}(p[1] + s[1], p[2] + s[2], p[3] + s[3],
            1000.0, 10.0, 0.01, 0.01, 0.01, 0.0, 0.0; frame = f, dataset = d)
        for (d, s) in enumerate(shifts) for f in frames for p in points]
    return BasicSMLD(em, camera, n_frames, length(shifts), Dict{String,Any}())
end

points = [(0.15, 0.15), (0.25, 0.55), (0.40, 0.25), (0.55, 0.85),
          (0.75, 0.45), (0.85, 0.95), (0.95, 0.65), (0.35, 0.95)]
smld = cc_fixture(points, fill((0.0, 0.0), 3))
pair = cc_fixture(points, fill((0.0, 0.0), 2))
smlds = [DC.filter_by_dataset(pair, d) for d in 1:2]
points3 = [(p..., 0.15 + 0.05 * mod(i, 3)) for (i, p) in enumerate(points)]
smld3 = cc_fixture(points3, fill((0.0, 0.0, 0.0), 2))
chunked = cc_fixture(points, [(0.0, 0.0)]; frames = (2, 5), n_frames = 6)
zero_intra = DC.LegendrePolynomial(smld; degree = 1)
warm = DC.LegendrePolynomial(smld; degree = 0)
_, info = driftcorrect(smld; quality = :fft, degree = 1, maxn = 3)

# Three ordinary datasets and one 2 μm outlier force the damped third FFT pass.
outlier = cc_fixture(points, [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (2.0, 0.0)];
    camera = IdealCamera(1:32, 1:12, 0.1))
# A narrow camera field still contains the dataset moved 10 μm and keeps FFTs small.
moved = cc_fixture(points, [(0.0, 0.0), (0.0, 0.0), (10.0, 0.0)];
    camera = IdealCamera(1:112, 1:12, 0.1), frames = (1, 2))
# The grid reduces to 2 by 2 tiles, with at least 225 localizations in every tile
# even after the upper field boundary is excluded. Both affine passes must succeed.
dense = [(0.2 + 0.02 * i, 0.2 + 0.02 * j) for i in 0:31 for j in 0:31]
affine = cc_fixture(dense, fill((0.0, 0.0), 2))
affine_smlds = [DC.filter_by_dataset(affine, d) for d in 1:2]

# Install once for the sweep to avoid repeated method invalidation and compilation.
with_broken_cc() do cc
    with_logger(NullLogger()) do
        paths = [
            "registered singlepass, three datasets" => (() -> driftcorrect(smld;
                degree = 0, maxn = 3)),
            "registered singlepass, two datasets" => (() -> driftcorrect(pair;
                degree = 0, maxn = 3)),
            "registered iterative" => (() -> driftcorrect(smld; quality = :iterative,
                warm_start = zero_intra, max_iterations = 2, maxn = 3)),
            "continuous singlepass" => (() -> driftcorrect(smld; dataset_mode = :continuous,
                degree = 0, maxn = 3)),
            "continuous iterative" => (() -> driftcorrect(smld; dataset_mode = :continuous,
                quality = :iterative, warm_start = zero_intra, max_iterations = 2, maxn = 3)),
            "continuous chunk_frames" => (() -> driftcorrect(chunked; dataset_mode = :continuous,
                chunk_frames = 3, degree = 0, maxn = 3)),
            "registered warm_start" => (() -> driftcorrect(smld; warm_start = warm, maxn = 3)),
            "registered 3D singlepass" => (() -> driftcorrect(smld3; degree = 0, maxn = 3)),
            "continuation" => (() -> driftcorrect(smld, info; max_iterations = 1, maxn = 3)),
            "findinter! zero seed" => (() -> DC.findinter!(
                DC.LegendrePolynomial(smld; degree = 0), smld, 2, [1], 3)),
            "align entropy" => (() -> align_smld(smlds; method = :entropy, maxn = 3)),
            "align fft" => (() -> align_smld(smlds; method = :fft)),
        ]
        for (name, body) in paths
            @testset "$name" begin
                check_cc_sites(body, cc)
            end
        end
        @testset "FFT including damped pass 3" begin
            n = check_cc_sites(() -> driftcorrect(outlier; quality = :fft, maxn = 3), cc)
            @test n > 2 * (outlier.n_datasets - 1)
        end
        @testset "Dropped CC including fresh warning CC" begin
            n = check_cc_sites(() -> driftcorrect(moved; degree = 0, maxn = 3), cc)
            @test n > (moved.n_datasets - 1) + 2 * moved.n_datasets
        end
        @testset "Affine global and tile CCs" begin
            n = check_cc_sites(() -> align_smld(affine_smlds; transform = :affine), cc;
                sample = true)
            @test n >= 10  # two global CCs and at least four tiles per pass
        end
    end

    @testset "Each failed CC logs an error, including repeated calls under one logger" begin
        @test_logs (:error, r"cross-correlation failed") min_level = Logging.Error begin
            reset_cc!(cc, 1)
            @test failure_root(() -> driftcorrect(smld; degree = 0, maxn = 3)) === cc.err[]
        end
        @test_logs (:error, r"cross-correlation failed") (:error, r"cross-correlation failed") min_level = Logging.Error begin
            for _ in 1:2
                reset_cc!(cc, 1)
                @test failure_root(() -> driftcorrect(smld; degree = 0, maxn = 3)) === cc.err[]
            end
        end
    end

    @testset "Interrupts reach the caller without an error record" begin
        cc.err[] = InterruptException()
        @test_logs min_level = Logging.Error begin
            reset_cc!(cc, 1)
            @test failure_root(() -> DC.findinter!(
                DC.LegendrePolynomial(smld; degree = 0), smld, 2, [1], 3)) === cc.err[]
        end
        @test_logs min_level = Logging.Error begin
            reset_cc!(cc, 1)
            @test failure_root(() -> align_smld(smlds; method = :entropy, maxn = 3)) === cc.err[]
        end
    end
end

@testset "3D shared ROI with negative camera x edges" begin
    camera = IdealCamera(collect(-0.2:0.1:1.4), collect(0.0:0.1:1.2))
    truth = (0.20, -0.10, 0.05)
    negative = cc_fixture(points3, [(0.0, 0.0, 0.0), truth]; camera)
    shift = DC.findshift(DC.filter_by_dataset(negative, 1), DC.filter_by_dataset(negative, 2);
        histbinsize = 0.05)
    @test maximum(abs.(shift .- collect(truth))) <= 0.05
end

@testset "Healthy CC after injected methods are removed" begin
    @test_logs min_level = Logging.Error driftcorrect(smld; degree = 0, maxn = 3)
    @test_logs min_level = Logging.Error align_smld(smlds; method = :entropy, maxn = 3)
end
