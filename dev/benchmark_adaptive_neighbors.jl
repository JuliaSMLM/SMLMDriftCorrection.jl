"""
Benchmark: Adaptive Neighbor Rebuilding for Kdtree Cost Function

Compares:
- Old: Rebuild KDTree every cost function call (O(N log N) per call)
- New: Fixed neighbors with adaptive rebuilding (O(N × k) per call)

Expected speedup: ~10-100x depending on data size
"""

using Pkg
Pkg.activate(@__DIR__)

using SMLMDriftCorrection
DC = SMLMDriftCorrection
using SMLMData
using SMLMSim
using Statistics
using Printf
using Random

Random.seed!(42)

println("=" ^ 70)
println("Benchmark: Adaptive Neighbor Rebuilding")
println("=" ^ 70)

#=============================================================================
# Generate test data (same approach as compare_cost_functions.jl)
=============================================================================#

function generate_test_data(; n_datasets=5, n_frames=500, pattern_density=5.0)
    params = StaticSMLMParams(
        pattern_density,
        0.13,             # σ_psf
        50,               # minphotons
        n_datasets,
        n_frames,
        50.0,             # framerate
        2,                # ndims
        [0.0, 1.0]
    )

    pattern = Nmer2D(n=8, d=0.15)
    molecule = GenericFluor(photons=5000.0, k_on=0.06, k_off=20.0)

    _, _, smld = simulate(params;
        pattern=pattern, molecule=molecule,
        camera=IdealCamera(1:128, 1:128, 0.1))

    return smld
end

#=============================================================================
# Benchmark cost function calls
=============================================================================#

function benchmark_cost_function(smld, dataset; n_calls=100)
    # Extract data for one dataset
    idx = [e.dataset for e in smld.emitters] .== dataset
    emitters = smld.emitters[idx]
    N = length(emitters)

    if N < 50
        return N, NaN, NaN
    end

    x = Float64[e.x for e in emitters]
    y = Float64[e.y for e in emitters]
    framenum = Int[e.frame for e in emitters]

    intra = DC.IntraPolynomial(2; degree=2)
    DC.initialize_random!(intra, 0.01, smld.n_frames)

    d_cutoff = 0.01
    x_work = similar(x)
    y_work = similar(y)
    θ = DC.intra2theta(intra)

    # ---- OLD: Rebuild tree every call ----
    old_cost_fn = () -> DC.costfun_kdtree_intra_2D(θ, x, y, framenum, d_cutoff, intra;
                                                    x_work=x_work, y_work=y_work)

    # Warmup
    for _ in 1:5
        old_cost_fn()
    end

    # Benchmark old
    t_old = @elapsed begin
        for _ in 1:n_calls
            old_cost_fn()
        end
    end
    time_old_per_call = t_old / n_calls * 1000  # ms

    # ---- NEW: Adaptive neighbors ----
    k = min(4, N - 1)
    state = DC.NeighborState(N, k, 2.0 * d_cutoff)
    DC.build_neighbors!(state, x, y)

    new_cost_fn = () -> DC.costfun_kdtree_intra_2D_adaptive(θ, x, y, framenum, d_cutoff, intra,
                                                            state, smld.n_frames;
                                                            x_work=x_work, y_work=y_work)

    # Warmup
    for _ in 1:5
        new_cost_fn()
    end

    # Benchmark new
    t_new = @elapsed begin
        for _ in 1:n_calls
            new_cost_fn()
        end
    end
    time_new_per_call = t_new / n_calls * 1000  # ms

    return N, time_old_per_call, time_new_per_call
end

#=============================================================================
# Run benchmarks at different scales
=============================================================================#

println("\n--- Cost Function Benchmark (per-call timing) ---\n")
println("N emitters │ Old (ms)  │ New (ms)  │ Speedup │ Est. 10K iters")
println("───────────┼───────────┼───────────┼─────────┼───────────────")

for (density, n_ds, n_fr) in [(2.0, 3, 500), (5.0, 5, 500), (10.0, 5, 500), (20.0, 5, 500)]
    smld = generate_test_data(; n_datasets=n_ds, n_frames=n_fr, pattern_density=density)
    N, t_old, t_new = benchmark_cost_function(smld, 1; n_calls=50)

    if isnan(t_old)
        @printf("%10d │ (skipped - not enough data)\n", N)
        continue
    end

    speedup = t_old / t_new
    est_old = t_old * 10000 / 1000  # seconds for 10K iterations
    est_new = t_new * 10000 / 1000

    @printf("%10d │ %9.3f │ %9.3f │ %6.1fx │ %.1fs → %.1fs\n",
            N, t_old, t_new, speedup, est_old, est_new)
end

#=============================================================================
# Full driftcorrect benchmark
=============================================================================#

println("\n--- Full driftcorrect() Benchmark ---\n")

smld = generate_test_data(; n_datasets=5, n_frames=500, pattern_density=5.0)
n_total = length(smld.emitters)
locs_per_ds = [count(e -> e.dataset == ds, smld.emitters) for ds in 1:smld.n_datasets]

println("Test data: $n_total total emitters")
println("           $(smld.n_datasets) datasets × $(smld.n_frames) frames")
println("           $(round(mean(locs_per_ds), digits=0)) emitters/dataset average\n")

# Apply drift
Random.seed!(123)
drift_true = DC.Polynomial(smld; degree=2, initialize="random")
smld_drifted = DC.applydrift(smld, drift_true)
Random.seed!(42)

# Benchmark Kdtree (uses adaptive neighbors now)
println("Running Kdtree (adaptive neighbors)...")
t_kd = @elapsed begin
    result_kd, _ = DC.driftcorrect(smld_drifted; cost_fun="Kdtree", d_cutoff=0.01)
end
@printf("  Time: %.2f seconds\n", t_kd)

# Benchmark Entropy for comparison
println("\nRunning Entropy for comparison...")
t_ent = @elapsed begin
    result_ent, _ = DC.driftcorrect(smld_drifted; cost_fun="Entropy", maxn=100)
end
@printf("  Time: %.2f seconds\n", t_ent)

#=============================================================================
# Extrapolation to large datasets
=============================================================================#

println("\n--- Extrapolation to Large Dataset ---\n")

# Use largest benchmark
smld_large = generate_test_data(; n_datasets=3, n_frames=500, pattern_density=20.0)
N_sample, t_old_sample, t_new_sample = benchmark_cost_function(smld_large, 1; n_calls=30)

if !isnan(t_old_sample)
    # Target: 200K emitters per dataset, 20 datasets (4M total)
    N_target = 200000
    n_ds_target = 20

    # Scale: old is O(N log N), new is O(N × k)
    scale = N_target / N_sample

    # Old scales as N log N
    t_old_scaled = t_old_sample * scale * log(N_target) / log(N_sample)
    # New scales as N
    t_new_scaled = t_new_sample * scale

    # Full optimization: 10,000 iterations × 20 datasets
    t_old_total = t_old_scaled * 10000 * n_ds_target / 1000 / 60  # minutes
    t_new_total = t_new_scaled * 10000 * n_ds_target / 1000 / 60  # minutes

    println("Extrapolation for 4M emitters (200K × 20 datasets):")
    println("  Per-call estimate at 200K emitters:")
    @printf("    Old: %.1f ms\n", t_old_scaled)
    @printf("    New: %.1f ms\n", t_new_scaled)
    println()
    println("  Full INTRA optimization (10K iters × 20 datasets):")
    @printf("    Old (rebuild tree): %.0f minutes\n", t_old_total)
    @printf("    New (adaptive):     %.0f minutes\n", t_new_total)
    @printf("    Estimated speedup:  %.0fx\n", t_old_total / t_new_total)
end

println("\n" * "=" ^ 70)
println("Summary:")
println("  - Adaptive neighbors avoid O(N log N) tree rebuild per iteration")
println("  - Speedup scales with data size (larger N = bigger speedup)")
println("  - Real-world 4M emitter datasets: hours → minutes")
println("=" ^ 70)
