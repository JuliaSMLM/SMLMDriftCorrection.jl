# GPU Benchmark Baseline: CPU performance measurements for SMLMDriftCorrection
#
# Measures the key computational kernels at realistic SMLM scales (100K and 1M points):
#   1. KDTree build + KNN query (k=100)
#   2. Single entropy cost function evaluation (the inner loop)
#   3. Full driftcorrect(:singlepass) call
#
# Run with: julia --project=dev -t4 dev/gpu_bench_baseline.jl

using SMLMData
using SMLMDriftCorrection
using NearestNeighbors
using BenchmarkTools
using Random
using Statistics

DC = SMLMDriftCorrection

println("=" ^ 72)
println("SMLMDriftCorrection CPU Benchmark Baseline")
println("=" ^ 72)
println("Julia threads: $(Threads.nthreads())")
println("Julia version: $(VERSION)")
println()

# --- Helper: build a synthetic SMLD2D with N emitters ---
function make_synthetic_smld(N::Int; fov=100.0, sigma=0.010, n_frames=20000, n_datasets=10)
    rng = MersenneTwister(42)
    pixel_size = 0.1  # 100 nm pixels

    # Compute camera grid to cover FOV
    n_pixels = ceil(Int, fov / pixel_size)
    camera = IdealCamera(1:n_pixels, 1:n_pixels, pixel_size)

    emitters = Vector{Emitter2DFit{Float64}}(undef, N)
    for i in 1:N
        x = rand(rng) * fov
        y = rand(rng) * fov
        photons = 1000.0 + rand(rng) * 4000.0
        bg = 10.0 + rand(rng) * 20.0
        sx = sigma * (0.8 + 0.4 * rand(rng))
        sy = sigma * (0.8 + 0.4 * rand(rng))
        s_xy = 0.0
        s_photons = sqrt(photons)
        s_bg = sqrt(bg)
        frame = rand(rng, 1:n_frames)
        dataset = rand(rng, 1:n_datasets)
        track_id = 0
        id = i
        emitters[i] = Emitter2DFit{Float64}(x, y, photons, bg, sx, sy, s_xy,
                                              s_photons, s_bg, frame, dataset, track_id, id)
    end

    smld = BasicSMLD{Float64, Emitter2DFit{Float64}}(
        emitters, camera, n_frames, n_datasets, Dict{String,Any}()
    )
    return smld
end

# ============================================================================
# Benchmark 1: KDTree build + KNN query
# ============================================================================
function bench_kdtree(N::Int, k::Int)
    rng = MersenneTwister(123)
    data = rand(rng, 2, N) .* 100.0  # 100 um FOV

    # Warmup
    tree = KDTree(data; leafsize=10)
    knn(tree, data, k + 1, true)

    println("  KDTree build (N=$N, 2D):")
    t_build = @benchmark KDTree($data; leafsize=10) samples=5 evals=1
    display(t_build)
    println()

    println("  KNN query (N=$N, k=$k):")
    tree = KDTree(data; leafsize=10)
    t_query = @benchmark knn($tree, $data, $(k + 1), true) samples=5 evals=1
    display(t_query)
    println()

    return t_build, t_query
end

# ============================================================================
# Benchmark 2: Single entropy cost function evaluation
# ============================================================================
function bench_entropy(N::Int, k::Int)
    rng = MersenneTwister(456)
    x  = rand(rng, N) .* 100.0
    y  = rand(rng, N) .* 100.0
    sx = fill(0.010, N)
    sy = fill(0.010, N)

    # Build KDTree + KNN (not timed here, separate benchmark)
    data = Matrix{Float64}(undef, 2, N)
    @inbounds for i in 1:N
        data[1, i] = x[i]
        data[2, i] = y[i]
    end
    tree = KDTree(data; leafsize=10)
    idxs, _ = knn(tree, data, k + 1, true)

    # Preallocate
    kldiv = Vector{Float64}(undef, k)

    # Warmup
    DC.entropy1_2D(idxs, x, y, sx, sy; divmethod="KL", kldiv=kldiv)

    println("  entropy1_2D (N=$N, k=$k):")
    t = @benchmark DC.entropy1_2D($idxs, $x, $y, $sx, $sy; divmethod="KL", kldiv=$kldiv) samples=5 evals=1
    display(t)
    println()

    return t
end

# ============================================================================
# Benchmark 2b: Adaptive entropy cost (NeighborState) -- the actual inner loop
# ============================================================================
function bench_entropy_adaptive(N::Int, k::Int)
    rng = MersenneTwister(789)
    x  = rand(rng, N) .* 100.0
    y  = rand(rng, N) .* 100.0
    sx = fill(0.010, N)
    sy = fill(0.010, N)

    state = DC.NeighborState(N, k, 0.1)
    DC.build_neighbors!(state, x, y)

    # Warmup
    DC.entropy1_2D_adaptive(state.neighbor_indices, state.k, x, y, sx, sy, state.kldiv)

    println("  entropy1_2D_adaptive (N=$N, k=$k):")
    t = @benchmark DC.entropy1_2D_adaptive(
        $(state.neighbor_indices), $(state.k),
        $x, $y, $sx, $sy, $(state.kldiv)
    ) samples=5 evals=1
    display(t)
    println()

    return t
end

# ============================================================================
# Benchmark 3: ub_entropy (combined KDTree + entropy -- the full cost eval)
# ============================================================================
function bench_ub_entropy(N::Int, k::Int)
    rng = MersenneTwister(101)
    x  = rand(rng, N) .* 100.0
    y  = rand(rng, N) .* 100.0
    sx = fill(0.010, N)
    sy = fill(0.010, N)

    # Warmup
    DC.ub_entropy(x, y, sx, sy; maxn=k)

    println("  ub_entropy (N=$N, k=$k) [KDTree + entropy combined]:")
    t = @benchmark DC.ub_entropy($x, $y, $sx, $sy; maxn=$k) samples=3 evals=1
    display(t)
    println()

    return t
end

# ============================================================================
# Benchmark 4: Full driftcorrect(:singlepass)
# ============================================================================
function bench_driftcorrect(smld; label="")
    println("  driftcorrect(:singlepass) $label  N=$(length(smld.emitters)), " *
            "$(smld.n_datasets) datasets, $(smld.n_frames) frames:")

    # Warmup (full run)
    DC.driftcorrect(smld; quality=:singlepass, verbose=0)

    t = @benchmark DC.driftcorrect($smld; quality=:singlepass, verbose=0) samples=3 evals=1
    display(t)
    println()

    return t
end

# ============================================================================
# Run all benchmarks
# ============================================================================
println()
println("-" ^ 72)
println("Building synthetic data...")
println("-" ^ 72)

smld_100k = make_synthetic_smld(100_000)
println("  100K SMLD: $(length(smld_100k.emitters)) emitters, " *
        "$(smld_100k.n_datasets) datasets, $(smld_100k.n_frames) frames")

smld_1m = make_synthetic_smld(1_000_000)
println("  1M SMLD:   $(length(smld_1m.emitters)) emitters, " *
        "$(smld_1m.n_datasets) datasets, $(smld_1m.n_frames) frames")

k = 100

# --- N = 100K ---
println()
println("=" ^ 72)
println("BENCHMARK SUITE: N = 100,000  (k=$k)")
println("=" ^ 72)

println("\n--- 1. KDTree build + KNN query ---")
bench_kdtree(100_000, k)

println("\n--- 2. entropy1_2D (KNN-indexed, no tree build) ---")
bench_entropy(100_000, k)

println("\n--- 2b. entropy1_2D_adaptive (matrix-indexed, no tree build) ---")
bench_entropy_adaptive(100_000, k)

println("\n--- 3. ub_entropy (combined tree+entropy) ---")
bench_ub_entropy(100_000, k)

println("\n--- 4. Full driftcorrect(:singlepass) ---")
bench_driftcorrect(smld_100k; label="[100K]")

# --- N = 1M ---
println()
println("=" ^ 72)
println("BENCHMARK SUITE: N = 1,000,000  (k=$k)")
println("=" ^ 72)

println("\n--- 1. KDTree build + KNN query ---")
bench_kdtree(1_000_000, k)

println("\n--- 2. entropy1_2D (KNN-indexed, no tree build) ---")
bench_entropy(1_000_000, k)

println("\n--- 2b. entropy1_2D_adaptive (matrix-indexed, no tree build) ---")
bench_entropy_adaptive(1_000_000, k)

println("\n--- 3. ub_entropy (combined tree+entropy) ---")
bench_ub_entropy(1_000_000, k)

println("\n--- 4. Full driftcorrect(:singlepass) ---")
bench_driftcorrect(smld_1m; label="[1M]")

println()
println("=" ^ 72)
println("Benchmark complete.")
println("=" ^ 72)
