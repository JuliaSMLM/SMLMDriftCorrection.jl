# scenario_continuous.jl - Continuous mode drift correction diagnostics
#
# Tests multi-dataset correction where drift accumulates across datasets.
# This simulates one long acquisition split into multiple files.

using Pkg
Pkg.activate(@__DIR__)

include("DiagnosticHelpers.jl")
using .DiagnosticHelpers
using SMLMDriftCorrection
using DataFrames
using Printf
using Statistics
using CairoMakie

const DC = SMLMDriftCorrection
const SCENARIO = :continuous

"""
    run_continuous_diagnostics(; kwargs...)

Run full diagnostic suite for continuous mode scenario.

# Keyword Arguments
- `density`: emitters/μm² (default: 5.0)
- `n_datasets`: number of datasets (default: 5)
- `n_frames`: frames per dataset (default: 1000)
- `degree`: polynomial degree (default: 2)
- `drift_scale`: drift amplitude in μm (default: 0.1)
- `seed`: random seed (default: 42)
- `verbose`: print progress (default: true)
"""
function run_continuous_diagnostics(;
        density::Real = 2.5,        # emitters/μm² (sparser)
        n_datasets::Int = 5,
        n_frames::Int = 2000,       # 2000 frames per dataset
        degree::Int = 2,
        drift_scale::Real = 0.2,    # ~200nm drift per dataset
        seed::Int = 42,
        verbose::Bool = true)

    verbose && println("=" ^ 60)
    verbose && println("CONTINUOUS MODE DIAGNOSTICS")
    verbose && println("=" ^ 60)

    # =========================================================================
    # 1. Generate test data
    # =========================================================================
    verbose && println("\n[1/6] Generating test data...")

    smld_orig = generate_test_smld(;
        density = density,
        n_datasets = n_datasets,
        n_frames = n_frames,
        seed = seed
    )

    verbose && println("  Emitters: $(length(smld_orig.emitters))")
    verbose && println("  Datasets: $(smld_orig.n_datasets)")
    verbose && println("  Frames per dataset: $(smld_orig.n_frames)")
    verbose && println("  Total frames: $(n_datasets * n_frames)")

    # =========================================================================
    # 2. Apply drift (continuous - accumulates)
    # =========================================================================
    verbose && println("\n[2/6] Applying drift (continuous mode)...")

    result_drift = apply_test_drift(smld_orig, SCENARIO;
        degree = degree,
        drift_scale = drift_scale,
        seed = seed
    )
    smld_drifted = result_drift.smld_drifted
    model_true = result_drift.model_true

    verbose && println("  Drift scale: $(drift_scale) μm")
    verbose && println("  Degree: $degree")

    # Show cumulative drift at dataset boundaries
    verbose && println("  Cumulative drift at dataset boundaries:")
    for ds = 1:n_datasets
        drift_end = DC.drift_at_frame(model_true, ds, n_frames)
        verbose && @printf("    DS %d end: (%.4f, %.4f) μm\n", ds, drift_end[1], drift_end[2])
    end

    # Total drift (end of last dataset minus start of first)
    drift_start = DC.drift_at_frame(model_true, 1, 1)
    drift_end = DC.drift_at_frame(model_true, n_datasets, n_frames)
    total_drift = sqrt((drift_end[1] - drift_start[1])^2 + (drift_end[2] - drift_start[2])^2)
    verbose && @printf("  Total drift magnitude: %.4f μm\n", total_drift)

    # =========================================================================
    # 3. Run drift correction (all quality tiers)
    # =========================================================================
    verbose && println("\n[3/6] Running drift correction (all quality tiers)...")

    # Run all three quality tiers
    tier_results = Dict{Symbol, NamedTuple}()

    for quality in [:fft, :singlepass, :iterative]
        verbose && println("\n  Running :$quality...")
        (smld_result, info) = DC.driftcorrect(smld_drifted;
            degree = degree,
            dataset_mode = :continuous,
            quality = quality,
            max_iterations = 5
        )

        # Compute RMSD (relative, removing global offset) for this tier
        x_orig = [e.x for e in smld_orig.emitters]
        y_orig = [e.y for e in smld_orig.emitters]
        x_corr = [e.x for e in smld_result.emitters]
        y_corr = [e.y for e in smld_result.emitters]
        offset_x = mean(x_corr) - mean(x_orig)
        offset_y = mean(y_corr) - mean(y_orig)
        dx = (x_corr .- offset_x) .- x_orig
        dy = (y_corr .- offset_y) .- y_orig
        rmsd_tier = sqrt(mean(dx.^2 .+ dy.^2)) * 1000

        verbose && @printf("    RMSD: %.2f nm (iterations=%d, converged=%s)\n",
                          rmsd_tier, info.iterations, info.converged)

        tier_results[quality] = (
            smld = smld_result,
            info = info,
            rmsd_nm = rmsd_tier
        )
    end

    # Use singlepass as the "main" result for detailed analysis
    smld_corrected = tier_results[:singlepass].smld
    model_recovered = tier_results[:singlepass].info.model
    info = tier_results[:singlepass].info

    verbose && println("\n  === Quality Tier Summary ===")
    verbose && @printf("    :fft        RMSD: %.2f nm\n", tier_results[:fft].rmsd_nm)
    verbose && @printf("    :singlepass RMSD: %.2f nm\n", tier_results[:singlepass].rmsd_nm)
    verbose && @printf("    :iterative  RMSD: %.2f nm\n", tier_results[:iterative].rmsd_nm)

    # Show recovered cumulative drift
    verbose && println("  Recovered cumulative drift at dataset boundaries:")
    for ds = 1:n_datasets
        drift_end = DC.drift_at_frame(model_recovered, ds, n_frames)
        verbose && @printf("    DS %d end: (%.4f, %.4f) μm\n", ds, drift_end[1], drift_end[2])
    end

    # =========================================================================
    # 4. Compute metrics
    # =========================================================================
    verbose && println("\n[4/6] Computing metrics...")

    # Overall RMSD
    rmsd_nm = compute_rmsd(smld_orig, smld_corrected)
    verbose && @printf("  Overall RMSD (includes global offset): %.2f nm\n", rmsd_nm)

    # RMSD after removing global offset (shows true alignment precision)
    x_orig = [e.x for e in smld_orig.emitters]
    y_orig = [e.y for e in smld_orig.emitters]
    x_corr = [e.x for e in smld_corrected.emitters]
    y_corr = [e.y for e in smld_corrected.emitters]
    offset_x = mean(x_corr) - mean(x_orig)
    offset_y = mean(y_corr) - mean(y_orig)
    dx = (x_corr .- offset_x) .- x_orig
    dy = (y_corr .- offset_y) .- y_orig
    rmsd_relative_nm = sqrt(mean(dx.^2 .+ dy.^2)) * 1000
    verbose && @printf("  RMSD (relative alignment): %.2f nm\n", rmsd_relative_nm)

    # Per-frame RMSD
    per_frame = compute_per_frame_rmsd(smld_orig, smld_corrected)

    # Residuals
    residuals = compute_position_residuals(smld_orig, smld_corrected)
    mean_error = mean(residuals.total_nm)
    max_error = maximum(residuals.total_nm)
    verbose && @printf("  Mean error: %.2f nm\n", mean_error)
    verbose && @printf("  Max error: %.2f nm\n", max_error)

    # Entropy
    entropy_before = compute_entropy_metrics(smld_drifted)
    entropy_after = compute_entropy_metrics(smld_corrected)
    entropy_reduction = (entropy_before.ub_entropy - entropy_after.ub_entropy) / entropy_before.ub_entropy * 100

    verbose && @printf("  Entropy before: %.4f\n", entropy_before.ub_entropy)
    verbose && @printf("  Entropy after: %.4f\n", entropy_after.ub_entropy)
    verbose && @printf("  Entropy reduction: %.1f%%\n", entropy_reduction)

    # Trajectory comparison
    traj = compare_trajectories(model_true, model_recovered)

    # Boundary continuity analysis
    boundary_errors = analyze_boundary_continuity(model_true, model_recovered)
    verbose && println("\n  Boundary continuity errors (nm):")
    for (i, err) in enumerate(boundary_errors)
        verbose && @printf("    DS %d → %d: %.1f nm\n", i, i+1, err)
    end

    # Intra-drift recovery per dataset (isolated from inter-shift)
    df_intra = analyze_intra_drift_per_dataset(model_true, model_recovered)
    verbose && println("\n  Intra-drift errors (isolated from inter-shift):")
    for row in eachrow(df_intra)
        verbose && @printf("    DS %d: max %.1f nm, mean %.1f nm\n",
                          row.dataset, row.max_intra_error_nm, row.mean_intra_error_nm)
    end

    # Per-chunk RMSD
    verbose && println("\n  Per-dataset RMSD:")
    per_ds_rmsd = Float64[]
    for ds = 1:n_datasets
        mask_orig = [e.dataset == ds for e in smld_orig.emitters]
        mask_corr = [e.dataset == ds for e in smld_corrected.emitters]

        smld_orig_ds = DC.filter_emitters(smld_orig, mask_orig)
        smld_corr_ds = DC.filter_emitters(smld_corrected, mask_corr)

        rmsd_ds = compute_rmsd(smld_orig_ds, smld_corr_ds)
        push!(per_ds_rmsd, rmsd_ds)
        verbose && @printf("    DS %d: %.2f nm\n", ds, rmsd_ds)
    end

    # =========================================================================
    # 5. Generate plots (singlepass only for continuous mode)
    # =========================================================================
    verbose && println("\n[5/6] Generating plots...")

    # Generate trajectory and render outputs for each quality tier
    for quality in [:fft, :singlepass, :iterative]
        tier_smld = tier_results[quality].smld
        tier_model = tier_results[quality].info.model
        tier_traj = compare_trajectories(model_true, tier_model)

        # Trajectory comparison with quality in filename
        fig_traj = plot_trajectory_comparison(tier_traj;
            title_suffix=" (Continuous Mode, :$quality)")
        save_figure(fig_traj, SCENARIO, "$(quality)_trajectory_comparison.png")

        # Cumulative trajectory (continuous-specific)
        fig_traj_cumul = plot_trajectory_cumulative(tier_model;
            title="Cumulative Drift Trajectory (Recovered, :$quality)")
        save_figure(fig_traj_cumul, SCENARIO, "$(quality)_trajectory_cumulative.png")

        # Compare true vs recovered cumulative
        fig_cumul_compare = plot_cumulative_comparison(model_true, tier_model)
        save_figure(fig_cumul_compare, SCENARIO, "$(quality)_trajectory_cumulative_comparison.png")

        # Render suite with quality prefix
        save_render_suite(smld_drifted, tier_smld, SCENARIO; prefix="$(quality)_")

        verbose && println("  Saved plots for :$quality")
    end

    # Detailed metrics plots (using singlepass as main result)
    # Residual histogram
    fig_hist = plot_residuals(residuals)
    save_figure(fig_hist, SCENARIO, "singlepass_residual_histogram.png")

    # Residual scatter
    fig_scatter = plot_residual_scatter(residuals)
    save_figure(fig_scatter, SCENARIO, "singlepass_residual_scatter.png")

    # RMSD vs frame
    fig_rmsd = plot_rmsd_vs_frame(per_frame)
    save_figure(fig_rmsd, SCENARIO, "singlepass_rmsd_vs_frame.png")

    # Boundary analysis (continuous-specific, using singlepass)
    fig_boundary = plot_boundary_analysis(model_true, model_recovered)
    save_figure(fig_boundary, SCENARIO, "singlepass_boundary_analysis.png")

    # Per-chunk table
    df_per_chunk = DataFrame(
        dataset = 1:n_datasets,
        rmsd_nm = per_ds_rmsd,
        boundary_error_nm = vcat([0.0], boundary_errors)  # No boundary before DS 1
    )

    # =========================================================================
    # 6. Save statistics
    # =========================================================================
    verbose && println("\n[6/6] Saving statistics...")

    stats = Dict(
        "rmsd_nm" => rmsd_nm,
        "rmsd_relative_nm" => rmsd_relative_nm,
        "mean_error_nm" => mean_error,
        "max_error_nm" => max_error,
        "entropy_before" => entropy_before.ub_entropy,
        "entropy_after" => entropy_after.ub_entropy,
        "entropy_reduction_pct" => entropy_reduction,
        "n_emitters" => length(smld_orig.emitters),
        "n_datasets" => n_datasets,
        "n_frames" => n_frames,
        "total_frames" => n_datasets * n_frames,
        "degree" => degree,
        "drift_scale_um" => drift_scale,
        "total_drift_um" => total_drift,
        "mean_boundary_error_nm" => mean(boundary_errors),
        "max_boundary_error_nm" => maximum(boundary_errors),
        "mean_intra_error_nm" => mean(df_intra.mean_intra_error_nm),
        "max_intra_error_nm" => maximum(df_intra.max_intra_error_nm),
        "global_offset_x_um" => offset_x,
        "global_offset_y_um" => offset_y,
        "seed" => seed,
    )

    save_stats_md(stats, SCENARIO; filename="stats_singlepass.md")

    # Save stats_all.md (all quality tiers)
    dir = ensure_output_dir(SCENARIO; clean=false)
    open(joinpath(dir, "stats_all.md"), "w") do io
        println(io, "# Quality Tier Comparison: CONTINUOUS")
        println(io)
        println(io, "## RMSD Comparison")
        println(io)
        println(io, "| Quality Tier | RMSD (nm) | Iterations | Converged |")
        println(io, "|--------------|-----------|------------|-----------|")
        @printf(io, "| **fft** | %.2f | %d | %s |\n",
                tier_results[:fft].rmsd_nm, tier_results[:fft].info.iterations, tier_results[:fft].info.converged)
        @printf(io, "| **singlepass** | %.2f | %d | %s |\n",
                tier_results[:singlepass].rmsd_nm, tier_results[:singlepass].info.iterations, tier_results[:singlepass].info.converged)
        @printf(io, "| **iterative** | %.2f | %d | %s |\n",
                tier_results[:iterative].rmsd_nm, tier_results[:iterative].info.iterations, tier_results[:iterative].info.converged)
        println(io)
        println(io, "## Parameters")
        println(io)
        println(io, "- Emitters: $(length(smld_orig.emitters))")
        println(io, "- Datasets: $n_datasets")
        println(io, "- Frames/dataset: $n_frames")
        println(io, "- Total frames: $(n_datasets * n_frames)")
        println(io, "- Degree: $degree")
        println(io, "- Drift scale: $drift_scale μm")
        println(io, "- Total drift: $(round(total_drift, digits=4)) μm")
        println(io, "- Seed: $seed")
        println(io)
        println(io, "## Notes")
        println(io)
        println(io, "For continuous mode, FFT only aligns inter-dataset shifts without intra correction,")
        println(io, "and doesn't account for cumulative drift across boundaries.")
    end
    verbose && println("  Saved: stats_all.md")

    verbose && println("\n" * "=" ^ 60)
    verbose && println("CONTINUOUS MODE DIAGNOSTICS COMPLETE")
    verbose && println("Quality Tier Results:")
    verbose && @printf("  :fft        RMSD: %.2f nm\n", tier_results[:fft].rmsd_nm)
    verbose && @printf("  :singlepass RMSD: %.2f nm\n", tier_results[:singlepass].rmsd_nm)
    verbose && @printf("  :iterative  RMSD: %.2f nm\n", tier_results[:iterative].rmsd_nm)
    verbose && println("=" ^ 60)

    return (
        smld_orig = smld_orig,
        smld_drifted = smld_drifted,
        smld_corrected = smld_corrected,
        model_true = model_true,
        model_recovered = model_recovered,
        boundary_errors = boundary_errors,
        df_per_chunk = df_per_chunk,
        stats = stats,
        tier_results = tier_results
    )
end

"""
    analyze_boundary_continuity(model_true, model_recovered)

Compute continuity errors at dataset boundaries.
Returns vector of errors (nm) for each boundary.
"""
function analyze_boundary_continuity(model_true, model_recovered)
    n_datasets = model_true.ndatasets
    n_frames = model_true.n_frames
    errors = Float64[]

    for ds = 1:(n_datasets - 1)
        # Get drift at boundary for both models
        # True model
        end_true = DC.drift_at_frame(model_true, ds, n_frames)
        start_true = DC.drift_at_frame(model_true, ds+1, 1)
        jump_true = sqrt((start_true[1] - end_true[1])^2 + (start_true[2] - end_true[2])^2)

        # Recovered model
        end_rec = DC.drift_at_frame(model_recovered, ds, n_frames)
        start_rec = DC.drift_at_frame(model_recovered, ds+1, 1)
        jump_rec = sqrt((start_rec[1] - end_rec[1])^2 + (start_rec[2] - end_rec[2])^2)

        # Error is the difference in jump magnitudes (in nm)
        push!(errors, abs(jump_rec - jump_true) * 1000.0)
    end

    return errors
end

"""
    plot_cumulative_comparison(model_true, model_recovered)

Compare true vs recovered cumulative drift trajectories.
"""
function plot_cumulative_comparison(model_true, model_recovered)
    traj_true = DC.drift_trajectory(model_true; cumulative=true)
    traj_rec = DC.drift_trajectory(model_recovered; cumulative=true)

    fig = Figure(size=(1200, 500))

    ax1 = Axis(fig[1, 1],
        xlabel = "Frame",
        ylabel = "Cumulative X drift (μm)",
        title = "X Drift (Cumulative)")

    ax2 = Axis(fig[1, 2],
        xlabel = "Frame",
        ylabel = "Cumulative Y drift (μm)",
        title = "Y Drift (Cumulative)")

    # Color by dataset
    datasets = unique(traj_true.dataset)
    colors = Makie.wong_colors()

    # Plot true first (thin), then recovered second (thick) so recovered visible on top
    for (i, ds) in enumerate(datasets)
        mask = traj_true.dataset .== ds
        c = colors[mod1(i, length(colors))]

        lines!(ax1, traj_true.frames[mask], traj_true.x[mask],
               color=c, linestyle=:solid, linewidth=1.5, label = i==1 ? "True" : "")
        lines!(ax2, traj_true.frames[mask], traj_true.y[mask],
               color=c, linestyle=:solid, linewidth=1.5)
    end
    for (i, ds) in enumerate(datasets)
        mask = traj_true.dataset .== ds
        c = colors[mod1(i, length(colors))]

        lines!(ax1, traj_rec.frames[mask], traj_rec.x[mask],
               color=c, linestyle=:dash, linewidth=3, label = i==1 ? "Recovered" : "")
        lines!(ax2, traj_rec.frames[mask], traj_rec.y[mask],
               color=c, linestyle=:dash, linewidth=3)
    end

    axislegend(ax1, position=:lt)

    Label(fig[0, :], "Cumulative Trajectory: True vs Recovered", fontsize=20)

    return fig
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_continuous_diagnostics()
end
