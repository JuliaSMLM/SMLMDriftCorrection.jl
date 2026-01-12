# Bottleneck profiling for drift correction cost functions
# Run with: julia --project=dev dev/profile_bottleneck.jl
#
# Goal: Identify what fraction of time is spent in:
#   1. KDTree construction
#   2. KNN queries
#   3. Entropy divergence calculations
#   4. logsumexp operations

using SMLMDriftCorrection
using SMLMSim
using NearestNeighbors
using Printf
using Statistics

DC = SMLMDriftCorrection

println("=" ^ 70)
println("BOTTLENECK PROFILER - Drift Correction Cost Functions")
println("=" ^ 70)
println("Julia threads: $(Threads.nthreads())")
println()

# --- Generate test data at different scales ---
function generate_test_data(n_datasets, n_frames, camera_size; density=10.0)
    params = StaticSMLMParams(
        density,    # density (emitters per μm²) - high for dSTORM-like
        0.13,       # σ_psf
        50,         # minphotons
        n_datasets,
        n_frames,
        50.0,       # framerate
        2,          # ndims
        [0.0, 1.0]
    )

    smld_true, smld_model, smld_noisy = simulate(params;
        pattern=Nmer2D(n=6, d=0.2),
        molecule=GenericFluor(; photons=5000.0, k_on=0.01, k_off=10.0),  # more blinks
        camera=IdealCamera(1:camera_size, 1:camera_size, 0.1))

    return smld_noisy
end

# --- Instrumented entropy calculation ---
function profile_entropy_components(x, y, σ_x, σ_y; maxn=200, n_runs=5)
    N = length(x)
    maxn = min(maxn, N - 1)

    # Preallocate
    data = Matrix{Float64}(undef, 2, N)

    times_matrix = Float64[]
    times_kdtree = Float64[]
    times_knn = Float64[]
    times_entropy = Float64[]

    for _ in 1:n_runs
        # 1. Matrix construction
        t1 = time_ns()
        @inbounds for i in 1:N
            data[1, i] = x[i]
            data[2, i] = y[i]
        end
        t2 = time_ns()
        push!(times_matrix, (t2 - t1) / 1e6)

        # 2. KDTree construction
        t3 = time_ns()
        kdtree = KDTree(data; leafsize=10)
        t4 = time_ns()
        push!(times_kdtree, (t4 - t3) / 1e6)

        # 3. KNN query
        t5 = time_ns()
        idxs, _ = knn(kdtree, data, maxn + 1, true)
        t6 = time_ns()
        push!(times_knn, (t6 - t5) / 1e6)

        # 4. Entropy calculation (entropy1_2D equivalent)
        t7 = time_ns()
        entropy_val = profile_entropy1_2D(idxs, x, y, σ_x, σ_y, maxn)
        t8 = time_ns()
        push!(times_entropy, (t8 - t7) / 1e6)
    end

    return (
        matrix = median(times_matrix),
        kdtree = median(times_kdtree),
        knn = median(times_knn),
        entropy = median(times_entropy),
        total = median(times_matrix) + median(times_kdtree) +
                median(times_knn) + median(times_entropy)
    )
end

# Inline entropy1_2D for profiling (matches cost_entropy.jl logic)
function profile_entropy1_2D(idxs, x, y, σ_x, σ_y, maxn)
    N = length(x)
    log_maxn = log(Float64(maxn))
    kldiv = Vector{Float64}(undef, maxn)
    out = 0.0

    @inbounds for i in 1:N
        idx = idxs[i]
        xi, yi = x[i], y[i]
        sxi, syi = σ_x[i], σ_y[i]

        for j in 1:maxn
            jj = idx[j+1]
            kldiv[j] = divKL_2D_inline(xi, yi, sxi, syi, x[jj], y[jj], σ_x[jj], σ_y[jj])
        end

        out += logsumexp_inline(kldiv, maxn) - log_maxn
    end

    # entropy_HD contribution
    c = 0.5 * log(2 * π * ℯ)
    hd = 0.0
    @inbounds for i in eachindex(σ_x)
        hd += c + 0.5 * log(σ_x[i] * σ_y[i])
    end

    return hd - out / N
end

@inline function divKL_2D_inline(x1, y1, sx1, sy1, x2, y2, sx2, sy2)
    si2_x = sx1^2
    si2_y = sy1^2
    sj2_x = sx2^2
    sj2_y = sy2^2

    out = log(sj2_x / si2_x) + si2_x / sj2_x + (x1 - x2)^2 / sj2_x
    out += log(sj2_y / si2_y) + si2_y / sj2_y + (y1 - y2)^2 / sj2_y
    out -= 2.0
    out /= 2.0
    return out
end

@inline function logsumexp_inline(x, n)
    max_val = x[1]
    @inbounds for i in 2:n
        if x[i] > max_val
            max_val = x[i]
        end
    end

    sum_exp = 0.0
    @inbounds for i in 1:n
        sum_exp += exp(-x[i] - (-max_val))
    end

    return -max_val + log(sum_exp)
end

# --- Profile KDTree cost function components ---
function profile_kdtree_components(x, y; d_cutoff=0.01, k=4, n_runs=5)
    N = length(x)
    data = Matrix{Float64}(undef, 2, N)

    times_matrix = Float64[]
    times_kdtree = Float64[]
    times_knn = Float64[]
    times_cost = Float64[]

    for _ in 1:n_runs
        # 1. Matrix construction
        t1 = time_ns()
        @inbounds for i in 1:N
            data[1, i] = x[i]
            data[2, i] = y[i]
        end
        t2 = time_ns()
        push!(times_matrix, (t2 - t1) / 1e6)

        # 2. KDTree construction
        t3 = time_ns()
        kdtree = KDTree(data; leafsize=10)
        t4 = time_ns()
        push!(times_kdtree, (t4 - t3) / 1e6)

        # 3. KNN query
        t5 = time_ns()
        idxs, dists = knn(kdtree, data, k, true)
        t6 = time_ns()
        push!(times_knn, (t6 - t5) / 1e6)

        # 4. Cost calculation
        t7 = time_ns()
        cost = 0.0
        @inbounds for nn in 2:length(dists)
            for d in dists[nn]
                cost -= exp(-d / d_cutoff)
            end
        end
        t8 = time_ns()
        push!(times_cost, (t8 - t7) / 1e6)
    end

    return (
        matrix = median(times_matrix),
        kdtree = median(times_kdtree),
        knn = median(times_knn),
        cost = median(times_cost),
        total = median(times_matrix) + median(times_kdtree) +
                median(times_knn) + median(times_cost)
    )
end

# --- Run profiling at multiple scales ---
println("Generating test datasets...")
println()

# Small scale (dev testing)
smld_small = generate_test_data(5, 500, 64; density=20.0)
N_small = length(smld_small.emitters)

# Medium scale
smld_medium = generate_test_data(10, 1000, 128; density=20.0)
N_medium = length(smld_medium.emitters)

# Large scale (closer to real dSTORM - target ~20-50k per dataset)
smld_large = generate_test_data(12, 5000, 256; density=50.0)
N_large = length(smld_large.emitters)

println("Dataset sizes:")
println("  Small:  N = $N_small")
println("  Medium: N = $N_medium")
println("  Large:  N = $N_large")
println()

# Extract coordinates
function extract_coords(smld)
    x = Float64[e.x for e in smld.emitters]
    y = Float64[e.y for e in smld.emitters]
    σ_x = Float64[e.σ_x for e in smld.emitters]
    σ_y = Float64[e.σ_y for e in smld.emitters]
    return x, y, σ_x, σ_y
end

# --- Profile each scale ---
function run_profile(smld, label, maxn)
    x, y, σ_x, σ_y = extract_coords(smld)
    N = length(x)

    println("-" ^ 70)
    println("$label (N=$N, maxn=$maxn)")
    println("-" ^ 70)

    # Warmup
    profile_entropy_components(x, y, σ_x, σ_y; maxn=min(maxn, N-1), n_runs=1)
    profile_kdtree_components(x, y; n_runs=1)

    # Profile Entropy cost
    println("\n  ENTROPY COST FUNCTION:")
    ent = profile_entropy_components(x, y, σ_x, σ_y; maxn=min(maxn, N-1), n_runs=5)
    @printf("    Matrix build:    %8.2f ms  (%5.1f%%)\n", ent.matrix, 100*ent.matrix/ent.total)
    @printf("    KDTree build:    %8.2f ms  (%5.1f%%)\n", ent.kdtree, 100*ent.kdtree/ent.total)
    @printf("    KNN query:       %8.2f ms  (%5.1f%%)\n", ent.knn, 100*ent.knn/ent.total)
    @printf("    Entropy calc:    %8.2f ms  (%5.1f%%)\n", ent.entropy, 100*ent.entropy/ent.total)
    @printf("    TOTAL:           %8.2f ms\n", ent.total)

    # Profile KDTree cost
    println("\n  KDTREE COST FUNCTION:")
    kdt = profile_kdtree_components(x, y; n_runs=5)
    @printf("    Matrix build:    %8.2f ms  (%5.1f%%)\n", kdt.matrix, 100*kdt.matrix/kdt.total)
    @printf("    KDTree build:    %8.2f ms  (%5.1f%%)\n", kdt.kdtree, 100*kdt.kdtree/kdt.total)
    @printf("    KNN query (k=4): %8.2f ms  (%5.1f%%)\n", kdt.knn, 100*kdt.knn/kdt.total)
    @printf("    Cost calc:       %8.2f ms  (%5.1f%%)\n", kdt.cost, 100*kdt.cost/kdt.total)
    @printf("    TOTAL:           %8.2f ms\n", kdt.total)

    println("\n  SPEEDUP: KDTree is $(round(ent.total/kdt.total, digits=1))x faster than Entropy")

    return (entropy=ent, kdtree=kdt)
end

# Run profiles
results_small = run_profile(smld_small, "SMALL", 200)
results_medium = run_profile(smld_medium, "MEDIUM", 200)
results_large = run_profile(smld_large, "LARGE", 200)

# --- Effect of maxn ---
println("\n" * "=" ^ 70)
println("EFFECT OF maxn ON ENTROPY COST (N=$N_medium)")
println("=" ^ 70)

x_med, y_med, σ_x_med, σ_y_med = extract_coords(smld_medium)

for maxn in [50, 100, 200, 500]
    if maxn >= N_medium
        continue
    end
    profile_entropy_components(x_med, y_med, σ_x_med, σ_y_med; maxn=maxn, n_runs=1)  # warmup
    ent = profile_entropy_components(x_med, y_med, σ_x_med, σ_y_med; maxn=maxn, n_runs=5)
    @printf("  maxn=%3d: KNN=%6.1fms  Entropy=%6.1fms  Total=%6.1fms\n",
            maxn, ent.knn, ent.entropy, ent.total)
end

# --- Estimate cost per optimization iteration ---
println("\n" * "=" ^ 70)
println("OPTIMIZATION COST ESTIMATES")
println("=" ^ 70)

# Typical optimization: 10000 iterations, ~1.5 evals per iteration for Nelder-Mead
n_iters = 10000
evals_per_iter = 1.5

for (label, N, res) in [("Small", N_small, results_small),
                         ("Medium", N_medium, results_medium),
                         ("Large", N_large, results_large)]
    ent_per_eval = res.entropy.total / 1000  # convert to seconds
    kdt_per_eval = res.kdtree.total / 1000

    ent_total = ent_per_eval * n_iters * evals_per_iter
    kdt_total = kdt_per_eval * n_iters * evals_per_iter

    @printf("\n  %s (N=%d):\n", label, N)
    @printf("    Entropy: %.1f sec per dataset (%.1f min for 12 datasets)\n",
            ent_total, ent_total * 12 / 60)
    @printf("    KDTree:  %.1f sec per dataset (%.1f min for 12 datasets)\n",
            kdt_total, kdt_total * 12 / 60)
end

println("\n" * "=" ^ 70)
println("CONCLUSIONS")
println("=" ^ 70)
println("""
Key findings will be printed here based on the actual timings.
Look at the percentages above to identify the true bottleneck.

Expected bottlenecks:
  - For Entropy: KNN query OR Entropy divergence calculation
  - For KDTree: KNN query (tree build is cheap)

Optimization opportunities:
  1. Reduce maxn (affects KNN and entropy calc linearly)
  2. Spatial subsampling (affects ALL components)
  3. Cache KDTree when θ changes are small (only helps if tree build is significant)
  4. Precompute σ² values (minor gain in entropy calc)
""")
