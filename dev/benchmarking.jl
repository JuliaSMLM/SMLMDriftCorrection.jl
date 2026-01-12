# Benchmarking script for SMLMDriftCorrection.jl
# Run with: julia --project=dev dev/benchmarking.jl
# Or: JULIA_NUM_THREADS=auto julia --project=dev dev/benchmarking.jl

using SMLMDriftCorrection
using SMLMSim
using BenchmarkTools
using NearestNeighbors
using Printf
using Statistics

DC = SMLMDriftCorrection

println("=" ^ 70)
println("SMLMDriftCorrection Benchmarks")
println("=" ^ 70)
println("Julia threads: $(Threads.nthreads())")
println()

# --- Data Generation ---
println("Generating test data...")

params_2d = StaticSMLMParams(
    2.0,      # density (ρ): emitters per μm²
    0.13,     # σ_psf: PSF width in μm
    50,       # minphotons
    10,       # ndatasets
    1000,     # nframes
    50.0,     # framerate
    2,        # ndims
    [0.0, 1.0]
)

smld_true, smld_model, smld_noisy = simulate(
    params_2d;
    pattern=Nmer2D(n=6, d=0.2),
    molecule=GenericFluor(; photons=5000.0, k_on=0.001, k_off=50.0),
    camera=IdealCamera(1:256, 1:256, 0.1)
)

# Create drifted data for correction benchmarks
drift_model = DC.Polynomial(smld_noisy; degree=2, initialize="random")
smld_drifted = DC.applydrift(smld_noisy, drift_model)

N = length(smld_noisy.emitters)
println("Generated $N localizations across $(smld_noisy.n_datasets) datasets")
println()

# --- Extract data for low-level benchmarks ---
x = [e.x for e in smld_noisy.emitters]
y = [e.y for e in smld_noisy.emitters]
σ_x = [e.σ_x for e in smld_noisy.emitters]
σ_y = [e.σ_y for e in smld_noisy.emitters]
data_2d = hcat(x, y)
se_2d = hcat(σ_x, σ_y)

# Dataset 1 data for intra benchmarks
dataset_mask = [e.dataset for e in smld_noisy.emitters] .== 1
x1 = x[dataset_mask]
y1 = y[dataset_mask]
σ_x1 = σ_x[dataset_mask]
σ_y1 = σ_y[dataset_mask]
framenum1 = [e.frame for e in smld_noisy.emitters[dataset_mask]]
data_1 = permutedims(hcat(x1, y1))
se_1 = permutedims(hcat(σ_x1, σ_y1))
N1 = length(x1)
println("Dataset 1: $N1 localizations")
println()

# === BENCHMARKS ===

println("-" ^ 70)
println("1. ENTROPY CALCULATION (ub_entropy)")
println("-" ^ 70)

# Matrix form: ub_entropy expects N x K (points as rows)
data_mat = data_2d      # N x 2
se_mat = se_2d          # N x 2
# For KDTree/costfun: K x N (dimensions as rows)
data_kdtree = permutedims(data_2d)  # 2 x N
se_kdtree = permutedims(se_2d)      # 2 x N

println("\nub_entropy (2D, N=$N, maxn=200):")
b1 = @benchmark DC.ub_entropy($data_mat, $se_mat; maxn=200) samples=10 evals=1
display(b1)
println()

println("\nub_entropy (2D, N=$N, maxn=50):")
b2 = @benchmark DC.ub_entropy($data_mat, $se_mat; maxn=50) samples=10 evals=1
display(b2)
println()

println("-" ^ 70)
println("2. KDTREE CONSTRUCTION + KNN QUERY")
println("-" ^ 70)

println("\nKDTree construction (N=$N):")
b3 = @benchmark KDTree($data_kdtree; leafsize=10) samples=50 evals=1
display(b3)
println()

kdtree = KDTree(data_kdtree; leafsize=10)
println("\nKNN query k=4 (N=$N):")
b4 = @benchmark knn($kdtree, $data_kdtree, 4, true) samples=50 evals=1
display(b4)
println()

println("-" ^ 70)
println("3. OPTIMIZATION (findintra!)")
println("-" ^ 70)

# Fresh model for optimization benchmark
drift_model_opt = DC.Polynomial(smld_drifted; degree=2)

println("\nfindintra! KDTree (dataset 1):")
intra_fresh = DC.IntraPolynomial(2; degree=2)
b10 = @benchmark DC.findintra!($intra_fresh, "Kdtree", $smld_drifted, 1, 0.01, 200) samples=3 evals=1
display(b10)
println()

println("\nfindintra! Entropy (dataset 1):")
intra_fresh2 = DC.IntraPolynomial(2; degree=2)
b11 = @benchmark DC.findintra!($intra_fresh2, "Entropy", $smld_drifted, 1, 0.01, 200) samples=3 evals=1
display(b11)
println()

println("-" ^ 70)
println("4. FULL DRIFT CORRECTION PIPELINE")
println("-" ^ 70)

println("\ndriftcorrect (Entropy, degree=2):")
b12 = @benchmark driftcorrect($smld_drifted; cost_fun="Entropy", degree=2) samples=3 evals=1
display(b12)
println()

println("\ndriftcorrect (Kdtree, degree=2):")
b13 = @benchmark driftcorrect($smld_drifted; cost_fun="Kdtree", degree=2) samples=3 evals=1
display(b13)
println()

println("-" ^ 70)
println("5. ALLOCATION ANALYSIS")
println("-" ^ 70)

println("\nDetailed allocation for ub_entropy:")
@time DC.ub_entropy(data_mat, se_mat; maxn=200)

println("\nDetailed allocation for driftcorrect (Entropy):")
@time driftcorrect(smld_drifted; cost_fun="Entropy", degree=2)

println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)
println()
@printf("%-40s %12s %15s\n", "Benchmark", "Time", "Memory")
println("-" ^ 70)
@printf("%-40s %12.2f ms %12.2f MiB\n", "ub_entropy (N=$N, maxn=200)",
        median(b1).time/1e6, b1.memory/1024^2)
@printf("%-40s %12.2f μs %12.2f KiB\n", "KDTree construction (N=$N)",
        median(b3).time/1e3, b3.memory/1024)
@printf("%-40s %12.2f μs %12.2f KiB\n", "KNN query k=4 (N=$N)",
        median(b4).time/1e3, b4.memory/1024)
@printf("%-40s %12.2f ms %12.2f MiB\n", "findintra! KDTree",
        median(b10).time/1e6, b10.memory/1024^2)
@printf("%-40s %12.2f ms %12.2f MiB\n", "findintra! Entropy",
        median(b11).time/1e6, b11.memory/1024^2)
@printf("%-40s %12.2f s  %12.2f GiB\n", "driftcorrect (Entropy)",
        median(b12).time/1e9, b12.memory/1024^3)
@printf("%-40s %12.2f ms %12.2f MiB\n", "driftcorrect (Kdtree)",
        median(b13).time/1e6, b13.memory/1024^2)
println()
speedup = median(b12).time / median(b13).time
@printf("Entropy vs Kdtree: Kdtree is %.1fx faster\n", speedup)
println("=" ^ 70)
