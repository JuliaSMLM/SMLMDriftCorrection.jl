#!/usr/bin/env julia
# Profile SMLMDriftCorrection to find performance hotspots
#
# Run: julia --project=dev --threads=4 dev/profile_hotspots.jl

using SMLMDriftCorrection
using SMLMSim
using Random
using Profile
using Statistics

const DC = SMLMDriftCorrection

# ============================================================================
# Generate test data at two scales
# ============================================================================

println("="^70)
println("GENERATING TEST DATA")
println("="^70)

Random.seed!(42)

# Small: 3 datasets, 1000 frames (~fast iteration)
params_small = StaticSMLMConfig(
    10.0, 0.13, 30, 3, 1000, 50.0, 2, [0.0, 1.0]
)
smld_small, _ = simulate(
    params_small;
    pattern=Nmer2D(n=6, d=0.2),
    molecule=GenericFluor(; photons=5000.0, k_on=0.02, k_off=50.0),
    camera=IdealCamera(1:64, 1:64, 0.1)
)
println("Small: $(length(smld_small.emitters)) locs, $(smld_small.n_datasets) datasets, $(smld_small.n_frames) frames")

# Medium: 5 datasets, 5000 frames (~realistic)
params_med = StaticSMLMConfig(
    10.0, 0.13, 30, 5, 5000, 50.0, 2, [0.0, 1.0]
)
smld_med, _ = simulate(
    params_med;
    pattern=Nmer2D(n=6, d=0.2),
    molecule=GenericFluor(; photons=5000.0, k_on=0.02, k_off=50.0),
    camera=IdealCamera(1:64, 1:64, 0.1)
)
println("Medium: $(length(smld_med.emitters)) locs, $(smld_med.n_datasets) datasets, $(smld_med.n_frames) frames")

# Apply known drift so correction has real work to do
drift_model_small = DC.LegendrePolynomial(smld_small; degree=2, initialize="random", rscale=0.1)
smld_small_drifted = DC.applydrift(smld_small, drift_model_small)

drift_model_med = DC.LegendrePolynomial(smld_med; degree=2, initialize="random", rscale=0.1)
smld_med_drifted = DC.applydrift(smld_med, drift_model_med)

# ============================================================================
# Warmup (compile everything)
# ============================================================================

println("\n" * "="^70)
println("WARMUP (compilation)")
println("="^70)
@time driftcorrect(smld_small_drifted; quality=:singlepass, verbose=0)
println("Warmup complete.")

# ============================================================================
# Timing breakdown by quality tier
# ============================================================================

println("\n" * "="^70)
println("TIMING: SMALL DATA ($(length(smld_small_drifted.emitters)) locs)")
println("="^70)

for q in [:fft, :singlepass, :iterative]
    maxiter = q == :iterative ? 3 : 10
    times = Float64[]
    for trial in 1:3
        t = @elapsed (_, info) = driftcorrect(smld_small_drifted; quality=q,
            max_iterations=maxiter, verbose=0)
        push!(times, t)
    end
    println("  $q: $(round(median(times), digits=3))s (median of 3)")
end

println("\n" * "="^70)
println("TIMING: MEDIUM DATA ($(length(smld_med_drifted.emitters)) locs)")
println("="^70)

for q in [:fft, :singlepass, :iterative]
    maxiter = q == :iterative ? 3 : 10
    times = Float64[]
    for trial in 1:3
        t = @elapsed (_, info) = driftcorrect(smld_med_drifted; quality=q,
            max_iterations=maxiter, verbose=0)
        push!(times, t)
    end
    println("  $q: $(round(median(times), digits=3))s (median of 3)")
end

# ============================================================================
# Component-level timing (medium data, singlepass)
# ============================================================================

println("\n" * "="^70)
println("COMPONENT BREAKDOWN (medium data, singlepass)")
println("="^70)

# Time the individual phases manually
smld_test = smld_med_drifted
model = DC.LegendrePolynomial(smld_test; degree=2)

# -- Intra-dataset --
println("\n--- Intra-dataset correction ---")
for nn in 1:smld_test.n_datasets
    DC.initialize_random!(model.intra[nn], 0.01, smld_test.n_frames)
end
t_intra = @elapsed begin
    Threads.@threads for nn = 1:smld_test.n_datasets
        DC.findintra!(model.intra[nn], smld_test, nn, 200)
    end
end
println("  Total intra ($(smld_test.n_datasets) datasets, threaded): $(round(t_intra, digits=3))s")

# Per-dataset (sequential, to see individual costs)
model2 = DC.LegendrePolynomial(smld_test; degree=2)
for nn in 1:smld_test.n_datasets
    DC.initialize_random!(model2.intra[nn], 0.01, smld_test.n_frames)
end
for nn in 1:smld_test.n_datasets
    t = @elapsed DC.findintra!(model2.intra[nn], smld_test, nn, 200)
    idx = [e.dataset == nn for e in smld_test.emitters]
    n_locs = sum(idx)
    println("    DS$nn ($n_locs locs): $(round(t, digits=3))s")
end

# -- Inter-dataset --
println("\n--- Inter-dataset correction ---")
# Use model from intra step
precomputed = DC.correctdrift(smld_test, model)

t_inter1 = @elapsed begin
    Threads.@threads for nn = 2:smld_test.n_datasets
        DC.findinter!(model, smld_test, nn, [1], 200; precomputed_corrected=precomputed)
    end
end
println("  Inter pass 1 (all vs DS1, threaded): $(round(t_inter1, digits=3))s")

t_inter2 = @elapsed begin
    for nn = 2:smld_test.n_datasets
        ref_datasets = collect(1:(nn-1))
        DC.findinter!(model, smld_test, nn, ref_datasets, 200)
    end
end
println("  Inter pass 2 (vs earlier, sequential): $(round(t_inter2, digits=3))s")

# -- Final entropy + correctdrift --
println("\n--- Final operations ---")
t_correct = @elapsed smld_corrected = DC.correctdrift(smld_test, model)
println("  correctdrift (deepcopy + apply): $(round(t_correct, digits=3))s")

t_entropy = @elapsed DC._compute_entropy(smld_corrected, 200)
println("  _compute_entropy (final): $(round(t_entropy, digits=3))s")

# ============================================================================
# Micro-benchmarks: entropy and KDTree
# ============================================================================

println("\n" * "="^70)
println("MICRO-BENCHMARKS: ENTROPY + KDTREE")
println("="^70)

# Extract vectors for direct profiling
emitters = smld_corrected.emitters
x = Float64[e.x for e in emitters]
y = Float64[e.y for e in emitters]
σ_x = Float64[e.σ_x for e in emitters]
σ_y = Float64[e.σ_y for e in emitters]
N = length(x)

println("  N = $N localizations")

# KDTree build
using NearestNeighbors
data = Matrix{Float64}(undef, 2, N)
for i in 1:N
    data[1, i] = x[i]
    data[2, i] = y[i]
end

t_tree = @elapsed kdtree = KDTree(data; leafsize=10)
println("  KDTree build: $(round(t_tree*1000, digits=2))ms")

# KNN query
for maxn in [50, 100, 200]
    t_knn = @elapsed idxs, _ = knn(kdtree, data, min(maxn+1, N), true)
    println("  KNN (maxn=$maxn): $(round(t_knn*1000, digits=2))ms")
end

# Entropy computation (without tree build)
for maxn in [50, 100, 200]
    idxs, _ = knn(kdtree, data, min(maxn+1, N), true)
    t_ent = @elapsed DC.entropy1_2D(idxs, x, y, σ_x, σ_y; divmethod="KL")
    println("  entropy1_2D (maxn=$maxn): $(round(t_ent*1000, digits=2))ms")
end

# logsumexp isolation
using StatsFuns: logsumexp
kldiv = rand(200)
t_lse = @elapsed for _ in 1:N
    logsumexp(-kldiv)
end
println("  logsumexp × $N calls: $(round(t_lse*1000, digits=2))ms")

# divKL_2D isolation
local div_sum = 0.0
t_div = @elapsed begin
    local s = 0.0
    for i in 1:N
        for j in 1:min(200, N-1)
            s += DC.divKL_2D(x[i], y[i], σ_x[i], σ_y[i],
                             x[j], y[j], σ_x[j], σ_y[j])
        end
    end
    div_sum = s
end
println("  divKL_2D (N×200 calls): $(round(t_div*1000, digits=2))ms")

# ============================================================================
# Allocation analysis
# ============================================================================

println("\n" * "="^70)
println("ALLOCATION ANALYSIS (medium data, singlepass)")
println("="^70)

# Full driftcorrect
alloc_info = @timed driftcorrect(smld_med_drifted; quality=:singlepass, verbose=0)
println("  Total: $(round(alloc_info.bytes / 1e6, digits=1)) MB allocated, $(alloc_info.gcstats.total_time / 1e9 |> x -> round(x, digits=3))s GC")

# Just intra (one dataset)
model3 = DC.LegendrePolynomial(smld_test; degree=2)
DC.initialize_random!(model3.intra[1], 0.01, smld_test.n_frames)
alloc_intra = @timed DC.findintra!(model3.intra[1], smld_test, 1, 200)
println("  findintra! (1 DS): $(round(alloc_intra.bytes / 1e6, digits=1)) MB, $(round(alloc_intra.time, digits=3))s")

# Just inter (one dataset)
precomputed2 = DC.correctdrift(smld_test, model)
alloc_inter = @timed DC.findinter!(model, smld_test, 2, [1], 200; precomputed_corrected=precomputed2)
println("  findinter! (1 DS): $(round(alloc_inter.bytes / 1e6, digits=1)) MB, $(round(alloc_inter.time, digits=3))s")

# Just correctdrift
alloc_cd = @timed DC.correctdrift(smld_test, model)
println("  correctdrift: $(round(alloc_cd.bytes / 1e6, digits=1)) MB, $(round(alloc_cd.time, digits=3))s")

# Just _compute_entropy
alloc_ent = @timed DC._compute_entropy(smld_corrected, 200)
println("  _compute_entropy: $(round(alloc_ent.bytes / 1e6, digits=1)) MB, $(round(alloc_ent.time, digits=3))s")

# ============================================================================
# Profile (flamegraph data)
# ============================================================================

println("\n" * "="^70)
println("PROFILE: singlepass on medium data")
println("="^70)

Profile.clear()
@profile driftcorrect(smld_med_drifted; quality=:singlepass, verbose=0)

# Print flat profile (top 30 lines)
println("\n--- Flat profile (top 30) ---")
Profile.print(maxdepth=12, mincount=5, noisefloor=2, sortedby=:count)

println("\n" * "="^70)
println("DONE")
println("="^70)
