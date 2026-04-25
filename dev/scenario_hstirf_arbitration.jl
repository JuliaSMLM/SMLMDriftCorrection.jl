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
        println("SUMMARY (smaller = better for all columns) — k=20, max=50nm")
        println("=" ^ 72)
        @printf("  %-14s  %7s  %7s  %7s  %7s  %7s  %7s  %9s  %9s  %8s\n",
                "tier", "med_nm", "IQR_nm", "std_nm", "p90_nm", "p99_nm",
                "tail30+", "Δmed_nm", "ΔIQR_nm", "elapsed_s")
        for q in ks
            d = diags[q]
            r = results[q]
            sm = haskey(d, :split_median_diff_nm) ? d.split_median_diff_nm : NaN
            si = haskey(d, :split_iqr_diff_nm)    ? d.split_iqr_diff_nm    : NaN
            @printf("  %-14s  %7.2f  %7.2f  %7.2f  %7.2f  %7.2f  %7.4f  %9.3f  %9.3f  %8.1f\n",
                    string(q), d.median_nm, d.iqr_nm, d.std_nm,
                    d.p90_nm, d.p99_nm, d.tail_30_nm, sm, si, r.elapsed)
        end
    end

    return (results = results, diags = diags)
end

"""
    run_hstirf_arbitration_v3(; quality_tiers, n_boot=500, sensitivity=true, ...)

v3 arbitration: same input, multiple quality tiers, with paired bootstrap CIs
on Δstats between each pair of methods AND a (k, max_dist) sensitivity grid.
Also surfaces runtime + residual_correlation per tier.

Decision rule built into the output: a tier is "clearly better" than another
only if its Δs (vs. the other) have CIs not crossing zero on a majority of
shape stats.
"""
function run_hstirf_arbitration_v3(;
        quality_tiers = QUALITY_TIERS,
        n_boot::Int = 500,
        sensitivity::Bool = true,
        sens_grid::Vector{Tuple{Int, Float64}} = [
            (10, 0.04), (10, 0.05),
            (20, 0.04), (20, 0.05),
            (40, 0.04), (40, 0.05),
        ],
        save_outputs::Bool = false,
        verbose::Int = 1)

    println("=" ^ 72)
    println("HS-TIRF DRIFT CORRECTION ARBITRATION (v3: paired bootstrap + sensitivity)")
    println("=" ^ 72)

    # 1. Run drift correction once per tier.
    results = Dict{Symbol, Any}()
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
        @printf("  elapsed %.1fs   iter %d   converged=%s   intra|corr|=(%.4f, %.4f)\n",
                r.elapsed, r.info.iterations, r.info.converged,
                r.info.residual_correlation.intra_summary.mean_abs_corr_x,
                r.info.residual_correlation.intra_summary.mean_abs_corr_y)
        results[q] = r
    end

    if length(results) < 2
        println("\nNeed ≥2 quality tiers; got $(length(results)). Aborting v3.")
        return (results = results,)
    end

    tiers = sort(collect(keys(results)))

    # 2. Sensitivity grid: per (k, max_dist) cell, compute per-emitter k-NN
    #    distance lists for each tier, then paired bootstrap pairwise.
    grid_cells = sensitivity ? sens_grid : [(20, 0.05)]
    cell_results = Dict{Tuple{Int, Float64}, Any}()  # (k, max_dist) → (per_emit, ci_pairs, point_diags)

    for (k, mx) in grid_cells
        println("\n--- sensitivity cell: k=$k, max_dist=$(round(Int, mx*1000))nm ---")
        per_emit = Dict{Symbol, Vector{Vector{Float64}}}()
        point_diags = Dict{Symbol, NamedTuple}()
        for q in tiers
            t = @elapsed begin
                pe = nn_distances_per_emitter(results[q].smld_corrected;
                        k = k, max_dist_um = mx)
                per_emit[q] = pe
                # also a point diagnostic from the full lists
                flat = Float64[]
                for v in pe; append!(flat, v); end
                point_diags[q] = shape_stats(flat; max_um = mx)
            end
            d = point_diags[q]
            @printf("  %-12s med=%6.2f iqr=%6.2f p90=%6.2f tail30+=%.4f  (build %.2fs)\n",
                    string(q), d.median_nm, d.iqr_nm, d.p90_nm, d.tail_30_nm, t)
        end

        # Paired bootstrap: every unordered pair (a, b)
        ci_pairs = Dict{Tuple{Symbol, Symbol}, NamedTuple}()
        for i in eachindex(tiers), j in eachindex(tiers)
            i < j || continue
            a, b = tiers[i], tiers[j]
            t = @elapsed ci = paired_bootstrap_compare(per_emit[a], per_emit[b];
                    n_boot = n_boot, max_um = mx)
            ci_pairs[(a, b)] = ci
            @printf("    paired Δ (%s − %s, B=%d): boot %.1fs\n", string(b), string(a), n_boot, t)
            print_paired_bootstrap_report(string(a), string(b), ci)
        end
        cell_results[(k, mx)] = (per_emit = per_emit,
                                 ci_pairs = ci_pairs,
                                 point_diags = point_diags)
    end

    # 3. Decision summary, structured per @codex into four distinct sections so
    #    a small NN-shape delta doesn't get overconfidently promoted to a default.
    #
    # Caveat (label explicitly): paired per-emitter bootstrap is first-order —
    # kNN rows share neighbours so CIs may be slightly under-spread on highly
    # clustered data. If a recommendation hinges on borderline CIs, escalate to
    # frame-block or cluster-block bootstrap (or BaGoL evidence) before changing
    # any public default.
    println("\n" * "=" ^ 72)
    println("DECISION SUMMARY")
    println("=" ^ 72)
    println("(per-emitter bootstrap is first-order; treat borderline CIs with caution)")

    # No-go threshold defaults — configurable.
    boundary_max_nm   = 100.0     # any boundary gap above this disqualifies the tier
    residual_corr_max = 0.05      # mean abs residual corr above this disqualifies
    decision_majority_frac = 0.6  # fraction of (cell × stat) wins needed to call it

    pairs_to_compare = [(tiers[i], tiers[j]) for i in eachindex(tiers) for j in eachindex(tiers) if i < j]

    for (a, b) in pairs_to_compare
        wins_a = 0; wins_b = 0; ties = 0; total = 0
        for ((k, mx), cr) in cell_results
            ci = cr.ci_pairs[(a, b)]
            for stat in (:delta_median, :delta_iqr, :delta_p90, :delta_p99, :delta_tail30)
                q = getfield(ci, stat)
                total += 1
                if q.lo > 0
                    wins_a += 1   # Δ > 0 means b > a, so a is tighter
                elseif q.hi < 0
                    wins_b += 1
                else
                    ties += 1
                end
            end
        end

        println("\n[$(a) vs $(b)]")

        # Section 1: shape metric (point estimates from k=20, mx=0.05 cell)
        ref_cell = (20, 0.05) in keys(cell_results) ? (20, 0.05) : first(keys(cell_results))
        pa = cell_results[ref_cell].point_diags[a]
        pb = cell_results[ref_cell].point_diags[b]
        shape_winner = nothing
        let scores = (
                Δmed = pb.median_nm - pa.median_nm,
                ΔIQR = pb.iqr_nm    - pa.iqr_nm,
                Δp90 = pb.p90_nm    - pa.p90_nm,
                Δp99 = pb.p99_nm    - pa.p99_nm,
                Δtail30 = pb.tail_30_nm - pa.tail_30_nm,
            )
            n_a_better = count(>(0), values(scores))
            n_b_better = count(<(0), values(scores))
            shape_winner = n_a_better > n_b_better ? string(a) :
                           n_b_better > n_a_better ? string(b) : "tie"
            @printf("  shape  : point Δ favours %-10s  (a-better=%d, b-better=%d on 5 stats at %s)\n",
                    shape_winner, n_a_better, n_b_better, "k=$(ref_cell[1]),max=$(round(Int, ref_cell[2]*1000))nm")
        end

        # Section 2: uncertainty (CI direction across cells)
        n_cells = length(cell_results)
        @printf("  CI     : %s wins %d  |  %s wins %d  |  within-noise %d  (of %d cell×stat = %d cells × 5 stats)\n",
                string(a), wins_a, string(b), wins_b, ties, total, n_cells)
        ci_winner = (wins_a / total >= decision_majority_frac) ? string(a) :
                    (wins_b / total >= decision_majority_frac) ? string(b) : "ambiguous"

        # Section 3: sanity (max boundary gap + residual correlation)
        ra = results[a]; rb = results[b]
        intra_a = max(ra.info.residual_correlation.intra_summary.mean_abs_corr_x,
                      ra.info.residual_correlation.intra_summary.mean_abs_corr_y)
        intra_b = max(rb.info.residual_correlation.intra_summary.mean_abs_corr_x,
                      rb.info.residual_correlation.intra_summary.mean_abs_corr_y)
        max_inter_a = maximum([1000*sqrt(sum(ra.info.model.inter[n].dm.^2)) for n in 1:ra.smld_corrected.n_datasets])
        max_inter_b = maximum([1000*sqrt(sum(rb.info.model.inter[n].dm.^2)) for n in 1:rb.smld_corrected.n_datasets])
        a_dq = (intra_a > residual_corr_max) || (max_inter_a > 1e3)   # 1 μm inter = catastrophic
        b_dq = (intra_b > residual_corr_max) || (max_inter_b > 1e3)
        @printf("  sanity : %s intra|corr|=%.3f  inter_max=%.0f nm   %s\n",
                string(a), intra_a, max_inter_a, a_dq ? "[DISQUALIFIED]" : "ok")
        @printf("           %s intra|corr|=%.3f  inter_max=%.0f nm   %s\n",
                string(b), intra_b, max_inter_b, b_dq ? "[DISQUALIFIED]" : "ok")

        # Section 4: recommendation
        rec = if a_dq && !b_dq
            "→ default $(b)  (a disqualified by sanity)"
        elseif b_dq && !a_dq
            "→ default $(a)  (b disqualified by sanity)"
        elseif a_dq && b_dq
            "→ NEITHER  (both disqualified by sanity)"
        elseif ci_winner == "ambiguous"
            "→ NO RECOMMENDATION  (CIs ambiguous at $(round(decision_majority_frac*100;digits=0))% threshold; both pass sanity)"
        else
            "→ default $(ci_winner)  (CI winner $(ci_winner), passes sanity)"
        end
        println("  rec    : $rec")
    end

    return (results = results, cell_results = cell_results)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_hstirf_arbitration()
end
