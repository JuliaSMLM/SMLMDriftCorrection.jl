# run_diagnostics.jl - Master script for drift correction diagnostics
#
# Runs all three diagnostic scenarios:
# - single: 1 dataset, intra-drift only
# - registered: Multi-dataset, independent inter-dataset offsets
# - continuous: Multi-dataset, drift accumulates across datasets
#
# Usage:
#   cd dev
#   julia --project run_diagnostics.jl
#
# Or run specific scenarios:
#   julia --project run_diagnostics.jl single
#   julia --project run_diagnostics.jl registered continuous

using Pkg
Pkg.activate(@__DIR__)

# Load DiagnosticHelpers module once
include("DiagnosticHelpers.jl")
using .DiagnosticHelpers

# Load shared dependencies
using SMLMDriftCorrection
using SMLMRender
using CairoMakie
using DataFrames
using Printf
using Statistics

const DC = SMLMDriftCorrection

# Include scenario function definitions (they use the already-loaded DiagnosticHelpers)
include("_scenario_single.jl")
include("_scenario_registered.jl")
include("_scenario_continuous.jl")

"""
    run_all_diagnostics(; scenarios=[:single, :registered, :continuous], kwargs...)

Run all specified diagnostic scenarios.

# Keyword Arguments
- `scenarios`: Vector of scenario symbols to run (default: all three)
- All other kwargs are passed to individual scenario functions
"""
function run_all_diagnostics(;
        scenarios::Vector{Symbol} = [:single, :registered, :continuous],
        verbose::Bool = true,
        kwargs...)

    println("=" ^ 70)
    println("SMLM DRIFT CORRECTION DIAGNOSTICS")
    println("=" ^ 70)
    println("Scenarios to run: $(join(string.(scenarios), ", "))")
    println()

    results = Dict{Symbol, Any}()
    timings = Dict{Symbol, Float64}()

    for scenario in scenarios
        t_start = time()

        if scenario == :single
            results[:single] = run_single_diagnostics(; verbose=verbose, kwargs...)

        elseif scenario == :registered
            results[:registered] = run_registered_diagnostics(; verbose=verbose, kwargs...)

        elseif scenario == :continuous
            results[:continuous] = run_continuous_diagnostics(; verbose=verbose, kwargs...)

        else
            @warn "Unknown scenario: $scenario (skipping)"
            continue
        end

        timings[scenario] = time() - t_start
        println()
    end

    # Print summary
    println()
    println("=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println()

    println("Scenario         RMSD (nm)     Entropy Δ (%)    Time (s)")
    println("-" ^ 60)

    for scenario in scenarios
        if haskey(results, scenario)
            stats = results[scenario].stats
            rmsd = stats["rmsd_nm"]
            ent_pct = stats["entropy_reduction_pct"]
            t = timings[scenario]
            @printf("%-16s %8.2f      %8.1f         %6.1f\n",
                    string(scenario), rmsd, ent_pct, t)
        end
    end

    println()
    println("Output directories:")
    for scenario in scenarios
        if haskey(results, scenario)
            dir = joinpath(@__DIR__, "output", string(scenario))
            println("  $scenario: $dir")
        end
    end

    println()
    println("=" ^ 70)
    println("ALL DIAGNOSTICS COMPLETE")
    println("=" ^ 70)

    return results
end

# Parse command line arguments
function main()
    if isempty(ARGS)
        # Run all scenarios
        run_all_diagnostics()
    else
        # Run specified scenarios
        scenarios = Symbol.(ARGS)
        valid = [:single, :registered, :continuous]
        for s in scenarios
            if s ∉ valid
                println("Invalid scenario: $s")
                println("Valid options: $(join(string.(valid), ", "))")
                return
            end
        end
        run_all_diagnostics(; scenarios=scenarios)
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
