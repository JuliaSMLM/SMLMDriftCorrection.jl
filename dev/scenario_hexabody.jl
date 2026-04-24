# scenario_hexabody.jl - Real-data stress test using the RGY hexabody SMLD
#
# Loads the pre-driftcorrect SMLD from the paper-genmab-hexabody pipeline
# (03_frameconnect output, Cell_01_bagol/RGY condition) and runs the current
# SMLMDriftCorrection against it. Compares against the reference post-correct
# SMLD (04_driftcorrect/smld_corrected.jld2) the canonical pipeline produced.
#
# Dataset characteristics (from 03_frameconnect/info.toml, 04_driftcorrect/stats.md):
#   20 datasets × 5000 frames, ~147k FrameConnect tracks, σ_median ≈ 5 nm
#   Max intra drift ≈ 109 nm, max inter shift ≈ 104 nm
#   Reference run: iterative/degree=3/shift_scale=0.05, 410 s, 5 iterations
#
# Outputs (written to dev/output/hexabody/):
#   config.toml              — DriftConfig we ran
#   info.toml                — run metadata (elapsed, iterations, converged, entropy)
#   stats.md                 — per-dataset shifts + residual correlations
#   drift_trajectory.png     — recovered intra-drift per dataset
#   shift_histogram.png      — inter-shift magnitudes per dataset
#   render_drifted_*.png     — histogram 10x / circle 50x / gaussian 20x (input)
#   render_corrected_*.png   — same renders on the corrected SMLD
#
# Usage:
#   julia --project=. dev/scenario_hexabody.jl
#   julia> include("dev/scenario_hexabody.jl"); run_hexabody_diagnostics(; quality=:singlepass)

using Pkg
Pkg.activate(@__DIR__)

include("DiagnosticHelpers.jl")
using .DiagnosticHelpers

using SMLMDriftCorrection
using SMLMData
using SMLMRender
using JLD2
using TOML
using CairoMakie
using Printf
using Statistics

const DC = SMLMDriftCorrection

# Default condition tree (paper-genmab-hexabody, HeLa SaturatingIgG10min+C1q, RGY).
# Overridable via kwargs so the script stays portable if the tree moves.
const HEXABODY_CONDITION = "/home/kalidke/julia_shared_dev/papers/paper-genmab-hexabody/data/results/juliasmlm/2024-05-26_HeLa_SaturatingIgG10min+C1q/HeLa_IgG1-2F8-RGY-AF647_5ugml_10min+C1q"
const HEXABODY_DEFAULT_CELL = "Cell_01_bagol"

hexabody_input(cell, condition=HEXABODY_CONDITION) =
    joinpath(condition, cell, "03_frameconnect/smld_combined.jld2")
hexabody_reference(cell, condition=HEXABODY_CONDITION) =
    joinpath(condition, cell, "04_driftcorrect/smld_corrected.jld2")

# Back-compat constants — Cell_01 RGY.
const HEXABODY_INPUT = hexabody_input(HEXABODY_DEFAULT_CELL)
const HEXABODY_REFERENCE = hexabody_reference(HEXABODY_DEFAULT_CELL)

# Output directory: dev/output/hexabody_<cell-slug>/
scenario_symbol(cell::AbstractString) =
    Symbol("hexabody_" * replace(lowercase(cell), r"_bagol$" => "", "_" => ""))

"""
    load_hexabody(; input=HEXABODY_INPUT)

Load the pre-driftcorrect RGY hexabody SMLD. JLD2 key is `"smld"`.
"""
function load_hexabody(; input::AbstractString = HEXABODY_INPUT)
    isfile(input) || error("hexabody SMLD not found at $input")
    return JLD2.load(input, "smld")
end

"""
    summarize_smld(smld; label="smld")

Print a one-line summary of an SMLD: datasets, frames, locs, median σ, bbox.
"""
function summarize_smld(smld; label::AbstractString = "smld")
    N = length(smld.emitters)
    σs = Float64[e.σ_x for e in smld.emitters]
    append!(σs, Float64[e.σ_y for e in smld.emitters])
    xs = Float64[e.x for e in smld.emitters]
    ys = Float64[e.y for e in smld.emitters]
    @printf("  %-24s %7d locs | %2d datasets | %5d frames | σ_med = %5.1f nm | bbox [%.1f, %.1f] × [%.1f, %.1f] μm\n",
        label, N, smld.n_datasets, smld.n_frames, 1000 * median(σs),
        minimum(xs), maximum(xs), minimum(ys), maximum(ys))
end

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

"""
    plot_recovered_drift(model) -> Figure

Match the `04_driftcorrect/drift_trajectory.png` layout genmab's pipeline
produces for multi-dataset cases:
- Top row: X drift and Y drift vs global frame, per-dataset colored segments
  (inter-shift + intra polynomial, boundary gaps preserved)
- Bottom row: XY trajectory in nm, per-dataset colored, green start-marker and
  red end-marker.
No ground truth is available for real data, so this is only the model output.
"""
function plot_recovered_drift(model)
    traj = DC.drift_trajectory(model)
    n_ds = model.ndatasets
    colors = resample_cmap(:tab20, n_ds)

    fig = Figure(size = (1400, 900))
    ax_x = Axis(fig[1, 1], xlabel = "Global frame", ylabel = "X drift (nm)",
                title = "Recovered X drift (inter + intra)")
    ax_y = Axis(fig[1, 2], xlabel = "Global frame", ylabel = "Y drift (nm)",
                title = "Recovered Y drift (inter + intra)")
    ax_xy = Axis(fig[2, 1:2], xlabel = "X drift (nm)", ylabel = "Y drift (nm)",
                 title = "XY drift trajectory",
                 aspect = DataAspect())

    for ds in 1:n_ds
        mask = traj.dataset .== ds
        c = colors[ds]
        lines!(ax_x,  traj.frames[mask], 1000 .* traj.x[mask]; color = c, linewidth = 1.4)
        lines!(ax_y,  traj.frames[mask], 1000 .* traj.y[mask]; color = c, linewidth = 1.4)
        lines!(ax_xy, 1000 .* traj.x[mask], 1000 .* traj.y[mask]; color = c, linewidth = 1.2)
    end

    # Global start / end markers on the XY panel (first frame of DS1, last frame of DSN)
    first_mask = traj.dataset .== 1
    last_mask  = traj.dataset .== n_ds
    if any(first_mask)
        i0 = findfirst(first_mask)
        scatter!(ax_xy, [1000 * traj.x[i0]], [1000 * traj.y[i0]];
                 color = :green, markersize = 14, strokecolor = :black,
                 strokewidth = 1, label = "start")
    end
    if any(last_mask)
        iN = findlast(last_mask)
        scatter!(ax_xy, [1000 * traj.x[iN]], [1000 * traj.y[iN]];
                 color = :red, markersize = 14, strokecolor = :black,
                 strokewidth = 1, label = "end")
    end
    axislegend(ax_xy; position = :rt)

    return fig
end

"""
    plot_inter_shift_histogram(model) -> Figure

Bar chart of per-dataset inter-shift magnitudes in nm.
"""
function plot_inter_shift_histogram(model)
    n = model.ndatasets
    mags = [1000 * sqrt(sum(model.inter[ds].dm .^ 2)) for ds in 1:n]
    shifts_x = [1000 * model.inter[ds].dm[1] for ds in 1:n]
    shifts_y = [1000 * model.inter[ds].dm[2] for ds in 1:n]

    fig = Figure(size = (1200, 500))
    ax1 = Axis(fig[1, 1], xlabel = "Dataset", ylabel = "|shift| (nm)",
               title = "Inter-dataset shift magnitude", xticks = 1:n)
    barplot!(ax1, 1:n, mags; color = :steelblue)

    ax2 = Axis(fig[1, 2], xlabel = "Dataset", ylabel = "shift (nm)",
               title = "Inter-dataset shifts (x, y)", xticks = 1:n)
    barplot!(ax2, (1:n) .- 0.2, shifts_x; width = 0.4, color = :steelblue, label = "Δx")
    barplot!(ax2, (1:n) .+ 0.2, shifts_y; width = 0.4, color = :orange,    label = "Δy")
    axislegend(ax2; position = :lt)

    return fig
end

# ---------------------------------------------------------------------------
# Render suite — matches genmab's 05_render config (GaussianRender zoom=20
# inferno n_sigmas=3 use_localization_precision=true integral clip 0.99 with
# a 3 μm scalebar) and additionally writes a histogram render.
# ---------------------------------------------------------------------------

function save_hexabody_renders(dir::AbstractString, smld_drifted, smld_corrected)
    # Gaussian — canonical paper-genmab-hexabody 05_render config:
    #   GaussianRender, zoom=20, inferno, n_sigmas=3, use_localization_precision,
    #   integral, clip 0.99, 3 μm scalebar.
    gconf = (strategy = GaussianRender(n_sigmas = 3.0,
                                       use_localization_precision = true,
                                       normalization = :integral),
             zoom = 20, colormap = :inferno, clip_percentile = 0.99,
             scalebar = true, scalebar_length = 3.0,
             scalebar_position = :br, scalebar_color = :white)
    for (label, smld) in (("drifted", smld_drifted), ("corrected", smld_corrected))
        result = render(smld; gconf...)
        fname = "render_$(label)_gaussian_20x.png"
        save_image(joinpath(dir, fname), result[1])
        println("  Saved: $fname")
    end

    # Histogram — canonical STEP 06 "temporal smear" config per genmab
    # (SMLMAnalysis src/config.jl:192): HistogramRender, zoom=10, turbo,
    # color_by=:absolute_frame, clip_percentile=nothing (critical — the default
    # 0.99 crushes the frame-axis colour range), scalebar=true.
    # Drifted shows rainbow streaks where the same emitter blinks across
    # different frames at different pixel positions; corrected collapses to
    # a near-constant color per structure.
    hconf = (strategy = HistogramRender(), zoom = 10,
             colormap = :turbo, color_by = :absolute_frame,
             clip_percentile = nothing, scalebar = true,
             scalebar_length = 3.0, scalebar_position = :br,
             scalebar_color = :white)
    for (label, smld) in (("drifted", smld_drifted), ("corrected", smld_corrected))
        result = render(smld; hconf...)
        fname = "render_$(label)_histogram_10x.png"
        save_image(joinpath(dir, fname), result[1])
        println("  Saved: $fname")
    end

    # Circles — paired temporal view at finer zoom. Useful for inspecting
    # whether pairs of blinks from the same underlying molecule have been
    # collapsed. Matches SMLMAnalysis src/config.jl:194.
    cconf = (strategy = CircleRender(), zoom = 50,
             color_by = :absolute_frame, colormap = :turbo,
             clip_percentile = nothing, scalebar = true,
             scalebar_length = 3.0, scalebar_position = :br,
             scalebar_color = :white)
    for (label, smld) in (("drifted", smld_drifted), ("corrected", smld_corrected))
        result = render(smld; cconf...)
        fname = "render_$(label)_circles_50x.png"
        save_image(joinpath(dir, fname), result[1])
        println("  Saved: $fname")
    end
end

# ---------------------------------------------------------------------------
# TOML / markdown writers
# ---------------------------------------------------------------------------

function write_config_toml(dir::AbstractString, config::DC.DriftConfig)
    d = Dict{String, Any}()
    d["type"] = "DriftConfig"
    d["quality"] = string(config.quality)
    d["degree"] = config.degree
    d["dataset_mode"] = string(config.dataset_mode)
    d["chunk_frames"] = config.chunk_frames
    d["n_chunks"] = config.n_chunks
    d["maxn"] = config.maxn
    d["max_iterations"] = config.max_iterations
    d["convergence_tol"] = config.convergence_tol
    d["verbose"] = config.verbose
    d["auto_roi"] = config.auto_roi
    d["σ_loc"] = config.σ_loc
    d["σ_target"] = config.σ_target
    d["roi_safety_factor"] = config.roi_safety_factor
    d["shift_scale"] = config.shift_scale

    path = joinpath(dir, "config.toml")
    open(path, "w") do io
        println(io, "# DriftConfig")
        TOML.print(io, d; sorted = true)
    end
    println("  Saved: config.toml")
end

function write_info_toml(dir::AbstractString, info::DC.DriftInfo, elapsed::Real,
                          input::AbstractString, reference::AbstractString,
                          rmsd_vs_ref::Real)
    inter_mags_nm = [1000 * sqrt(sum(info.model.inter[ds].dm .^ 2))
                     for ds in 1:info.model.ndatasets]
    rc = info.residual_correlation

    d = Dict{String, Any}()
    d["type"] = "DriftInfo"
    d["backend"] = string(info.backend)
    d["iterations"] = info.iterations
    d["converged"] = info.converged
    d["entropy_final"] = info.entropy
    d["elapsed_s"] = elapsed
    d["elapsed_s_reported_by_info"] = info.elapsed_s
    d["n_datasets"] = info.model.ndatasets
    d["n_frames"] = info.model.n_frames
    d["inter_shift_magnitude_nm"] = inter_mags_nm
    d["inter_shift_max_nm"] = maximum(inter_mags_nm)
    d["rmsd_vs_reference_nm"] = 1000 * rmsd_vs_ref
    d["input_smld"] = input
    d["reference_smld"] = reference
    d["residual_correlation"] = Dict(
        "intra_mean_abs_corr_x" => rc.intra_summary.mean_abs_corr_x,
        "intra_mean_abs_corr_y" => rc.intra_summary.mean_abs_corr_y,
        "inter_corr_x" => rc.inter.corr_x,
        "inter_corr_y" => rc.inter.corr_y,
    )
    d["entropy_history"] = info.history

    path = joinpath(dir, "info.toml")
    open(path, "w") do io
        println(io, "# SMLMDriftCorrection run info — real-data hexabody stress test")
        TOML.print(io, d; sorted = true)
    end
    println("  Saved: info.toml")
end

function write_stats_md(dir::AbstractString, info::DC.DriftInfo, elapsed::Real,
                         rmsd_vs_ref::Real, config::DC.DriftConfig,
                         cell::AbstractString)
    rc = info.residual_correlation
    n_ds = info.model.ndatasets
    inter_mags_nm = [1000 * sqrt(sum(info.model.inter[ds].dm .^ 2)) for ds in 1:n_ds]

    path = joinpath(dir, "stats.md")
    open(path, "w") do io
        println(io, "# Drift Correction Stats: RGY Hexabody $(cell) (HeLa)")
        println(io)
        println(io, "## Summary")
        println(io, "- **Mode**: $(config.dataset_mode)")
        println(io, "- **Quality**: $(config.quality)")
        println(io, "- **Degree**: $(config.degree)")
        println(io, "- **Converged**: $(info.converged)")
        println(io, "- **Iterations**: $(info.iterations)")
        println(io, "- **Datasets**: $n_ds")
        println(io, "- **Frames per dataset**: $(info.model.n_frames)")
        @printf(io, "- **Elapsed**: %.2f s\n", elapsed)
        @printf(io, "- **Final entropy**: %.1f\n", info.entropy)
        @printf(io, "- **Max inter-dataset shift**: %.1f nm\n", maximum(inter_mags_nm))
        if isfinite(rmsd_vs_ref)
            @printf(io, "- **RMSD vs. reference pipeline**: %.3f nm\n", 1000 * rmsd_vs_ref)
        end

        println(io)
        println(io, "## Inter-Dataset Shifts")
        println(io, "| Dataset | Δx (nm) | Δy (nm) | |shift| (nm) |")
        println(io, "|---------|---------|---------|--------------|")
        for ds in 1:n_ds
            dx = 1000 * info.model.inter[ds].dm[1]
            dy = 1000 * info.model.inter[ds].dm[2]
            mag = sqrt(dx^2 + dy^2)
            @printf(io, "| %d | %+.2f | %+.2f | %.2f |\n", ds, dx, dy, mag)
        end

        println(io)
        println(io, "## Residual Correlation (lower = better)")
        println(io, "### Intra-dataset (mean |correlation|)")
        @printf(io, "- x: %.5f\n", rc.intra_summary.mean_abs_corr_x)
        @printf(io, "- y: %.5f\n", rc.intra_summary.mean_abs_corr_y)
        println(io)
        println(io, "### Per-dataset residual correlation")
        println(io, "| Dataset | n_locs | corr_x | corr_y |")
        println(io, "|---------|--------|--------|--------|")
        for entry in rc.intra_per_dataset
            @printf(io, "| %d | %d | %+.5f | %+.5f |\n",
                    entry.dataset, entry.n_locs, entry.corr_x, entry.corr_y)
        end
        println(io)
        println(io, "### Inter-dataset")
        @printf(io, "- corr_x: %+.5f\n", rc.inter.corr_x)
        @printf(io, "- corr_y: %+.5f\n", rc.inter.corr_y)

        println(io)
        println(io, "## Entropy history")
        for (i, ent) in enumerate(info.history)
            @printf(io, "- iter %d: %.3f\n", i, ent)
        end
    end
    println("  Saved: stats.md")
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

"""
    run_hexabody_diagnostics(; kwargs...)

Run drift correction on the hexabody dataset, report metrics, and write a full
output suite (tomls, plots, renders) into `dev/output/hexabody/`.

# Keyword Arguments
- `input::AbstractString=HEXABODY_INPUT`
- `reference::AbstractString=HEXABODY_REFERENCE` — post-correct SMLD to compare against
- `quality::Symbol=:iterative`
- `degree::Int=3`
- `dataset_mode::Symbol=:registered`
- `maxn::Int=100`
- `max_iterations::Int=10`
- `convergence_tol::Float64=0.001`
- `shift_scale::Float64=0.05`
- `auto_roi::Bool=false`
- `verbose::Int=1`
- `compare_reference::Bool=true`
- `save_outputs::Bool=true` — toggle the full render/toml/plot suite
- `clean_output::Bool=true` — wipe dev/output/hexabody/ before writing

# Returns
NamedTuple with `smld_corrected`, `info`, `elapsed`, `rmsd_vs_reference`, `output_dir`.
"""
function run_hexabody_diagnostics(;
        cell::AbstractString = HEXABODY_DEFAULT_CELL,
        condition::AbstractString = HEXABODY_CONDITION,
        input::AbstractString = hexabody_input(cell, condition),
        reference::AbstractString = hexabody_reference(cell, condition),
        quality::Symbol = :iterative,
        degree::Int = 3,
        dataset_mode::Symbol = :registered,
        maxn::Int = 100,
        max_iterations::Int = 10,
        convergence_tol::Float64 = 0.001,
        shift_scale::Float64 = 0.05,
        auto_roi::Bool = false,
        verbose::Int = 1,
        compare_reference::Bool = true,
        save_outputs::Bool = true,
        clean_output::Bool = true)

    scenario = scenario_symbol(cell)
    output_dir = joinpath(@__DIR__, "output", string(scenario))

    println("=" ^ 72)
    println("RGY HEXABODY DRIFT CORRECTION STRESS TEST — $(cell)")
    println("=" ^ 72)

    # Wipe stale outputs at the start (not the end) so a failed run can't leave
    # a mix of current + stale files behind. `clean_output=false` keeps them.
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
    t_load = @elapsed smld = load_hexabody(; input = input)
    @printf("  loaded in %.2f s\n", t_load)
    summarize_smld(smld; label = "input")

    println("\n[2/5] Running driftcorrect(quality=$(quality), degree=$(degree))")
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
    summarize_smld(smld_corrected; label = "corrected")

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
        summarize_smld(smld_ref; label = "reference")
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

    # -----------------------------------------------------------------------
    # [5/5] Full output suite
    # -----------------------------------------------------------------------
    if save_outputs
        println("\n[5/5] Writing output suite → $(output_dir)")
        # clean=false here — we already cleaned at step [0/5]; just make sure
        # the directory exists.
        dir = DiagnosticHelpers.ensure_output_dir(scenario; clean=false)

        write_config_toml(dir, config)
        write_info_toml(dir, info, t_correct, input, reference, rmsd_vs_ref)
        write_stats_md(dir, info, t_correct, rmsd_vs_ref, config, cell)

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

# When run as a script, execute the default iterative case.
if abspath(PROGRAM_FILE) == @__FILE__
    run_hexabody_diagnostics()
end
