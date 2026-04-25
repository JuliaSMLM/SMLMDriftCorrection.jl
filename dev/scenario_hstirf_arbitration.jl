# scenario_hstirf_arbitration.jl — pick a continuous-mode default for hs-tirf
# by running adaptive-singlepass and iterative+merged-cloud on the SAME input,
# then comparing pair-distance reconstruction quality + split-half consistency.
#
# This is the empirical criterion @codex and I converged on after Keith pointed
# out that ref-pipeline RMSD isn't truth. @bagol confirmed run_bagol can be the
# heavyweight gold-standard arbiter if needed, but recommended scaffolding the
# lightweight version first; this is that. If results here are ambiguous between
# methods, escalate to BaGoL.
#
# Note: requires both quality=:adaptive and quality=:iterative to be available on
# the current branch. On codex/intra-merged-cloud only :iterative is present
# (singlepass acts as quality=:singlepass / not :adaptive). Use the combined
# branch (codex/adaptive-legendre + merged-cloud, tip 9ffbcc8) for the full
# comparison. Falls back to comparing :singlepass vs :iterative on this branch.

using Pkg
Pkg.activate(@__DIR__)

include("pairdist_diagnostic.jl")
include("scenario_hstirf.jl")

using SMLMDriftCorrection
using JLD2
using Printf

const DC = SMLMDriftCorrection

# Quality tiers we'd like to compare. If :adaptive isn't available on the
# current branch, run_hstirf_diagnostics will error or be filtered out below.
const QUALITY_TIERS = [:adaptive, :iterative]

"""
    run_hstirf_arbitration(; quality_tiers=[:adaptive, :iterative], save_outputs=false)

Run drift correction with each requested quality tier on the hs-tirf Gattaquant
dataset, compute the lightweight pair-distance + split-half diagnostic on each
corrected SMLD, and print a side-by-side report.
"""
function run_hstirf_arbitration(;
        quality_tiers = QUALITY_TIERS,
        save_outputs::Bool = false,
        verbose::Int = 1)

    println("=" ^ 72)
    println("HS-TIRF DRIFT CORRECTION ARBITRATION")
    println("=" ^ 72)

    results = Dict{Symbol, Any}()
    diags   = Dict{Symbol, NamedTuple}()

    for q in quality_tiers
        println("\n[run] quality=:$q  …")
        local r
        try
            r = run_hstirf_diagnostics(;
                quality      = q,
                verbose      = verbose,
                save_outputs = save_outputs,
                clean_output = false)
        catch e
            @warn "Skipping :$q — driftcorrect threw" exception=(e, catch_backtrace())
            continue
        end
        @printf("  elapsed %.1fs   iter %d   converged=%s\n",
                r.elapsed, r.info.iterations, r.info.converged)
        results[q] = r

        d = pairdist_diagnostic(r.smld_corrected;
                k               = 20,
                max_dist_um     = 0.05,
                tail_cutoffs_um = (0.025, 0.030, 0.035, 0.040),
                split_half      = true)
        diags[q] = d
    end

    println("\n" * "=" ^ 72)
    println("PAIR-DISTANCE DIAGNOSTIC (lightweight v1)")
    println("=" ^ 72)
    for q in quality_tiers
        haskey(diags, q) || continue
        println()
        print_pairdist_report("quality=:$q", diags[q])
    end

    if length(diags) >= 2
        ks = collect(keys(diags))
        # Side-by-side compact summary (smaller = better for all numeric columns)
        println("\n" * "=" ^ 72)
        println("SUMMARY (smaller = better for all columns)")
        println("=" ^ 72)
        @printf("  %-14s  %7s  %7s  %7s  %7s  %7s  %7s  %9s  %9s\n",
                "tier", "med_nm", "IQR_nm", "std_nm", "p90_nm", "p99_nm",
                "tail30+", "Δmed_nm", "ΔIQR_nm")
        for q in ks
            d = diags[q]
            sm = haskey(d, :split_median_diff_nm) ? d.split_median_diff_nm : NaN
            si = haskey(d, :split_iqr_diff_nm)    ? d.split_iqr_diff_nm    : NaN
            @printf("  %-14s  %7.2f  %7.2f  %7.2f  %7.2f  %7.2f  %7.4f  %9.3f  %9.3f\n",
                    string(q), d.median_nm, d.iqr_nm, d.std_nm,
                    d.p90_nm, d.p99_nm, d.tail_30_nm, sm, si)
        end
    end

    return (results = results, diags = diags)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_hstirf_arbitration()
end
