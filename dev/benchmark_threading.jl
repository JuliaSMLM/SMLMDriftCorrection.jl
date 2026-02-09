# benchmark_threading.jl - Benchmark inter-dataset threading performance
#
# Usage:
#   julia --project=dev -t1 dev/benchmark_threading.jl    # baseline (single thread)
#   julia --project=dev -t4 dev/benchmark_threading.jl    # threaded
#
# Compares singlepass and iterative quality tiers for registered and continuous modes.
# No plots or renders - just timing and RMSD numbers.

using Pkg
Pkg.activate(@__DIR__)

include("DiagnosticHelpers.jl")
using .DiagnosticHelpers
using SMLMDriftCorrection
using Printf
using Statistics

const DC = SMLMDriftCorrection

function run_benchmark(; n_datasets::Int=5, n_frames::Int=2000,
                        density::Real=2.5, degree::Int=2, seed::Int=42)
    n_threads = Threads.nthreads()
    println("=" ^ 70)
    println("Threading Benchmark: $(n_threads) thread(s)")
    println("  n_datasets=$n_datasets, n_frames=$n_frames, density=$density, degree=$degree")
    println("=" ^ 70)

    # Generate test data
    println("\nGenerating test data...")
    smld_orig = generate_test_smld(;
        density=density, n_datasets=n_datasets, n_frames=n_frames, seed=seed)
    println("  $(length(smld_orig.emitters)) emitters, $(smld_orig.n_datasets) datasets")

    # Results table
    results = NamedTuple{(:scenario, :quality, :rmsd_nm, :elapsed_s), Tuple{Symbol, Symbol, Float64, Float64}}[]

    for (scenario, dataset_mode) in [(:registered, :registered), (:continuous, :continuous)]
        # Apply drift for this scenario
        drifted = apply_test_drift(smld_orig, scenario;
            degree=degree, drift_scale=0.1, inter_scale=0.2, seed=seed)
        smld_drifted = drifted.smld_drifted

        for quality in (:singlepass, :iterative)
            label = "$scenario / $quality"
            print("  $label ... ")

            elapsed = @elapsed begin
                (smld_corrected, info) = driftcorrect(smld_drifted;
                    quality=quality, degree=degree, dataset_mode=dataset_mode)
            end

            rmsd = compute_rmsd(smld_orig, smld_corrected)
            push!(results, (scenario=scenario, quality=quality, rmsd_nm=rmsd, elapsed_s=elapsed))
            @printf("RMSD=%.2f nm, elapsed=%.2f s\n", rmsd, elapsed)
        end
    end

    # Summary table
    println("\n" * "=" ^ 70)
    @printf("%-24s  %10s  %10s  %s\n", "Scenario/Quality", "RMSD (nm)", "Time (s)", "Threads")
    println("-" ^ 70)
    for r in results
        label = "$(r.scenario) / $(r.quality)"
        @printf("%-24s  %10.2f  %10.2f  %d\n", label, r.rmsd_nm, r.elapsed_s, n_threads)
    end
    println("=" ^ 70)

    return results
end

run_benchmark()
