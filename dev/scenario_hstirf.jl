# scenario_hstirf.jl — Real-data stress test using the HS-TIRF Gattaquant nanoruler dataset
#
# Input: 03_frameconnect/smld_combined.jld2 from the project-hs-tirf srdata pipeline
# (red 642 nm DNA-PAINT on a Gattaquant 20 nm nanoruler, MIC-format H5 split into
# 18 blocks × 1000 frames). `dataset_mode=:continuous` — one long acquisition split
# into files, which exercises the endpoint-chaining warmstart that `:registered`
# mode does not.
#
# Reference run: iterative / degree=3 / shift_scale=1.0 / :continuous, 69 s,
# 3 iterations, converged. Max intra ≈ 52 nm, max inter ≈ 53 nm.
#
# Outputs → dev/output/hstirf/: same suite as scenario_hexabody (config.toml,
# info.toml, stats.md, drift_trajectory.png, shift_histogram.png, and renders).
#
# Usage:
#   julia --project=. dev/scenario_hstirf.jl
#   julia> include("dev/scenario_hstirf.jl"); run_hstirf_diagnostics()

using Pkg
Pkg.activate(@__DIR__)

# Reuse the helpers defined in scenario_hexabody.jl (plotting, toml/md writing,
# render suite). They take paths + info + config so they're scenario-agnostic.
include("scenario_hexabody.jl")

const HSTIRF_BASE = "/home/kalidke/julia_shared_dev/projects/project-hs-tirf/data/srdata/results_continuous"
const HSTIRF_INPUT = joinpath(HSTIRF_BASE, "03_frameconnect/smld_combined.jld2")
const HSTIRF_REFERENCE = joinpath(HSTIRF_BASE, "04_driftcorrect/smld_corrected.jld2")
const HSTIRF_SCENARIO = :hstirf

"""
    run_hstirf_diagnostics(; kwargs...)

Run drift correction on the HS-TIRF Gattaquant dataset (continuous mode) and
write the full output suite to `dev/output/hstirf/`.

# Keyword Arguments
- `input`, `reference`: override paths (defaults to project-hs-tirf pipeline)
- `quality=:iterative`, `degree=3`, `dataset_mode=:continuous`
- `maxn=100`, `max_iterations=10`, `convergence_tol=0.001`
- `shift_scale=1.0` (reference config — :continuous doesn't need :registered's tighter 0.05)
- `verbose=1`, `compare_reference=true`, `save_outputs=true`, `clean_output=true`

Returns a NamedTuple with `smld_corrected`, `info`, `elapsed`, `rmsd_vs_reference`, `output_dir`.
"""
function run_hstirf_diagnostics(;
        input::AbstractString = HSTIRF_INPUT,
        reference::AbstractString = HSTIRF_REFERENCE,
        quality::Symbol = :iterative,
        degree::Int = 3,
        dataset_mode::Symbol = :continuous,
        maxn::Int = 100,
        max_iterations::Int = 10,
        convergence_tol::Float64 = 0.001,
        shift_scale::Float64 = 1.0,
        auto_roi::Bool = false,
        verbose::Int = 1,
        compare_reference::Bool = true,
        save_outputs::Bool = true,
        clean_output::Bool = true)

    output_dir = joinpath(@__DIR__, "output", string(HSTIRF_SCENARIO))

    println("=" ^ 72)
    println("HS-TIRF GATTAQUANT 20 nm NANORULER DRIFT CORRECTION STRESS TEST")
    println("=" ^ 72)

    # Wipe stale outputs at the start (not the end) so a failed run can't leave
    # a mix of current + stale files behind.
    if save_outputs && clean_output
        if isdir(output_dir)
            n_removed = 0
            for f in readdir(output_dir)
                rm(joinpath(output_dir, f); force=true)
                n_removed += 1
            end
            println("\n[0/5] Cleaned $(n_removed) stale files from $(output_dir)")
        end
    end

    println("\n[1/5] Loading pre-driftcorrect SMLD")
    @printf("  path: %s\n", input)
    isfile(input) || error("hstirf SMLD not found at $input")
    t_load = @elapsed smld = JLD2.load(input, "smld")
    @printf("  loaded in %.2f s\n", t_load)
    summarize_smld(smld; label="input")

    println("\n[2/5] Running driftcorrect(quality=$(quality), degree=$(degree), mode=$(dataset_mode))")
    config = DC.DriftConfig(;
        quality = quality,
        degree = degree,
        dataset_mode = dataset_mode,
        maxn = maxn,
        max_iterations = max_iterations,
        convergence_tol = convergence_tol,
        shift_scale = shift_scale,
        auto_roi = auto_roi,
        verbose = verbose,
    )
    t_correct = @elapsed (smld_corrected, info) = DC.driftcorrect(smld, config)
    @printf("  elapsed: %.2f s   iterations: %d   converged: %s\n",
        t_correct, info.iterations, info.converged)
    @printf("  final entropy: %.1f\n", info.entropy)
    summarize_smld(smld_corrected; label="corrected")

    println("\n[3/5] Inter-dataset shifts (|shift| nm)")
    for ds in 1:smld_corrected.n_datasets
        mag = 1000 * sqrt(sum(info.model.inter[ds].dm .^ 2))
        @printf("  DS%02d: %5.1f nm\n", ds, mag)
    end

    println("\n  Residual correlation (lower = better, 0.01 ≈ reference):")
    rc = info.residual_correlation
    @printf("    intra mean |corr_x| = %.4f   |corr_y| = %.4f\n",
        rc.intra_summary.mean_abs_corr_x, rc.intra_summary.mean_abs_corr_y)
    @printf("    inter       corr_x  = %+.4f   corr_y  = %+.4f\n",
        rc.inter.corr_x, rc.inter.corr_y)

    rmsd_vs_ref = NaN
    if compare_reference && isfile(reference)
        println("\n[4/5] Comparing against reference post-correct SMLD")
        smld_ref = JLD2.load(reference, "smld")
        summarize_smld(smld_ref; label="reference")
        if length(smld_ref.emitters) == length(smld_corrected.emitters)
            n = length(smld_corrected.emitters)
            Δ2 = 0.0
            @inbounds for i in 1:n
                dx = smld_corrected.emitters[i].x - smld_ref.emitters[i].x
                dy = smld_corrected.emitters[i].y - smld_ref.emitters[i].y
                Δ2 += dx * dx + dy * dy
            end
            rmsd_vs_ref = sqrt(Δ2 / n)
            @printf("  RMSD vs. reference: %.3f nm  (N = %d)\n", 1000 * rmsd_vs_ref, n)
        else
            @printf("  Skipping RMSD: emitter counts differ (%d vs %d)\n",
                length(smld_corrected.emitters), length(smld_ref.emitters))
        end
    else
        println("\n[4/5] Reference not found or comparison disabled — skipping")
    end

    if save_outputs
        println("\n[5/5] Writing output suite → $(output_dir)")
        # clean=false — already cleaned at step [0/5].
        dir = DiagnosticHelpers.ensure_output_dir(HSTIRF_SCENARIO; clean=false)

        write_config_toml(dir, config)
        write_info_toml(dir, info, t_correct, input, reference, rmsd_vs_ref)
        write_stats_md(dir, info, t_correct, rmsd_vs_ref, config, "hstirf-gattaquant")

        fig_drift = plot_recovered_drift(info.model)
        save(joinpath(dir, "drift_trajectory.png"), fig_drift)
        println("  Saved: drift_trajectory.png")

        fig_hist = plot_inter_shift_histogram(info.model)
        save(joinpath(dir, "shift_histogram.png"), fig_hist)
        println("  Saved: shift_histogram.png")

        save_hexabody_renders(dir, smld, smld_corrected)
    else
        println("\n[5/5] save_outputs=false — skipping output suite")
    end

    println("\n" * "=" ^ 72)
    return (
        smld_corrected = smld_corrected,
        info = info,
        elapsed = t_correct,
        rmsd_vs_reference = rmsd_vs_ref,
        output_dir = output_dir,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_hstirf_diagnostics()
end
