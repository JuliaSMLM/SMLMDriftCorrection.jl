# run_all_scenarios.jl - Unified test runner for all 3 modes × quality tiers
#
# Outputs per mode:
#   - Drifted renders: gaussian, time histogram, time circles
#   - Corrected renders: same for each applicable tier
#   - Trajectory comparison
#   - stats.md with detailed per-tier and per-dataset metrics
#
# Single mode: only singlepass (FFT/iterative don't help - no inter-dataset)
# Continuous/Registered modes: all 3 tiers

using Pkg
Pkg.activate(@__DIR__)

include("DiagnosticHelpers.jl")
using .DiagnosticHelpers
using SMLMDriftCorrection
using SMLMRender
using CairoMakie
using Statistics
using Printf

const DC = SMLMDriftCorrection

# =============================================================================
# RMSD with offset alignment
# =============================================================================

"""
Compute RMSD after removing global offset (drift correction recovers shape, not absolute position).
"""
function compute_aligned_rmsd(smld_orig, smld_corrected)
    N = length(smld_orig.emitters)
    @assert N == length(smld_corrected.emitters) "Emitter count mismatch"

    # Compute mean offset
    mean_dx = 0.0
    mean_dy = 0.0
    for i in 1:N
        mean_dx += smld_corrected.emitters[i].x - smld_orig.emitters[i].x
        mean_dy += smld_corrected.emitters[i].y - smld_orig.emitters[i].y
    end
    mean_dx /= N
    mean_dy /= N

    # Compute RMSD after offset removal
    sum_sq = 0.0
    for i in 1:N
        dx = smld_corrected.emitters[i].x - smld_orig.emitters[i].x - mean_dx
        dy = smld_corrected.emitters[i].y - smld_orig.emitters[i].y - mean_dy
        sum_sq += dx^2 + dy^2
    end

    rmsd_um = sqrt(sum_sq / N)
    return rmsd_um * 1000.0  # Convert to nm
end

"""
Compute position residuals after removing global offset.
"""
function compute_aligned_residuals(smld_orig, smld_corrected)
    N = length(smld_orig.emitters)

    # Compute mean offset
    mean_dx = 0.0
    mean_dy = 0.0
    for i in 1:N
        mean_dx += smld_corrected.emitters[i].x - smld_orig.emitters[i].x
        mean_dy += smld_corrected.emitters[i].y - smld_orig.emitters[i].y
    end
    mean_dx /= N
    mean_dy /= N

    x_nm = Vector{Float64}(undef, N)
    y_nm = Vector{Float64}(undef, N)
    total_nm = Vector{Float64}(undef, N)

    for i in 1:N
        dx = (smld_corrected.emitters[i].x - smld_orig.emitters[i].x - mean_dx) * 1000.0
        dy = (smld_corrected.emitters[i].y - smld_orig.emitters[i].y - mean_dy) * 1000.0
        x_nm[i] = dx
        y_nm[i] = dy
        total_nm[i] = sqrt(dx^2 + dy^2)
    end

    return (x_nm=x_nm, y_nm=y_nm, total_nm=total_nm)
end

# =============================================================================
# Render helpers
# =============================================================================

"""
Save renders: gaussian, time histogram, time circles
"""
function save_renders(smld, prefix::String, dir::String; roi=nothing)
    roi_kwargs = roi !== nothing ? (roi=roi,) : ()

    # Gaussian at 20x
    result = render(smld; strategy=GaussianRender(), zoom=20,
                    colormap=:inferno, roi_kwargs...)
    save_image(joinpath(dir, "$(prefix)_gaussian.png"), result.image)

    # Histogram at 10x, color by time
    result = render(smld; strategy=HistogramRender(), zoom=10,
                    color_by=:absolute_frame, colormap=:turbo, roi_kwargs...)
    save_image(joinpath(dir, "$(prefix)_histogram.png"), result.image)

    # Circles at 50x, color by time
    result = render(smld; strategy=CircleRender(), zoom=50,
                    color_by=:absolute_frame, colormap=:turbo, roi_kwargs...)
    save_image(joinpath(dir, "$(prefix)_circles.png"), result.image)
end

"""
Plot single tier trajectory comparison against ground truth.
Recovered trajectory is offset-aligned to ground truth at frame 1 for visualization.
"""
function plot_single_trajectory(traj_true, model_recovered; tier::Symbol=:singlepass, title::String="")
    fig = Figure(size=(1400, 500))

    ax1 = Axis(fig[1, 1],
        xlabel = "Frame",
        ylabel = "X drift (μm)",
        title = "X Drift - $tier")

    ax2 = Axis(fig[1, 2],
        xlabel = "Frame",
        ylabel = "Y drift (μm)",
        title = "Y Drift - $tier")

    ax3 = Axis(fig[1, 3],
        xlabel = "X drift (μm)",
        ylabel = "Y drift (μm)",
        title = "Drift Trajectory (X vs Y)",
        aspect = DataAspect())

    # Ground truth - solid blue
    lines!(ax1, traj_true.frames, traj_true.true_x, color=:blue, linewidth=2, label="Ground Truth")
    lines!(ax2, traj_true.frames, traj_true.true_y, color=:blue, linewidth=2)
    lines!(ax3, traj_true.true_x, traj_true.true_y, color=:blue, linewidth=2, label="Ground Truth")

    # Recovered trajectory (offset-aligned)
    traj_rec = DC.drift_trajectory(model_recovered)
    offset_x = traj_true.true_x[1] - traj_rec.x[1]
    offset_y = traj_true.true_y[1] - traj_rec.y[1]
    rec_x = traj_rec.x .+ offset_x
    rec_y = traj_rec.y .+ offset_y

    lines!(ax1, traj_rec.frames, rec_x, color=:red, linewidth=2, linestyle=:dash, label="Recovered")
    lines!(ax2, traj_rec.frames, rec_y, color=:red, linewidth=2, linestyle=:dash)
    lines!(ax3, rec_x, rec_y, color=:red, linewidth=2, linestyle=:dash, label="Recovered")

    axislegend(ax1, position=:lt)
    axislegend(ax3, position=:lt)

    if !isempty(title)
        Label(fig[0, :], title, fontsize=18)
    end

    return fig
end

"""
Plot trajectory comparison for all tiers.
FFT shows flat line (no intra-drift model).
Recovered trajectories are offset-aligned to ground truth at frame 1 for visualization
(drift correction recovers shape, not absolute offset).
"""
function plot_tier_trajectories(traj_true, results::Dict;
                                  title_suffix::String="", show_fft::Bool=true)
    fig = Figure(size=(1400, 500))

    ax1 = Axis(fig[1, 1],
        xlabel = "Frame",
        ylabel = "X drift (μm)",
        title = "X Drift" * title_suffix)

    ax2 = Axis(fig[1, 2],
        xlabel = "Frame",
        ylabel = "Y drift (μm)",
        title = "Y Drift" * title_suffix)

    ax3 = Axis(fig[1, 3],
        xlabel = "X drift (μm)",
        ylabel = "Y drift (μm)",
        title = "Drift Trajectory (X vs Y)",
        aspect = DataAspect())

    # Ground truth - solid blue
    lines!(ax1, traj_true.frames, traj_true.true_x, color=:blue, linewidth=2, label="Ground Truth")
    lines!(ax2, traj_true.frames, traj_true.true_y, color=:blue, linewidth=2)
    lines!(ax3, traj_true.true_x, traj_true.true_y, color=:blue, linewidth=2, label="Ground Truth")

    # FFT - flat line (no intra correction) - only for multi-dataset
    if show_fft && haskey(results, :fft)
        fft_x = zeros(length(traj_true.frames))
        fft_y = zeros(length(traj_true.frames))
        lines!(ax1, traj_true.frames, fft_x, color=:gray, linewidth=1.5, linestyle=:dot, label="FFT (flat)")
        lines!(ax2, traj_true.frames, fft_y, color=:gray, linewidth=1.5, linestyle=:dot)
        scatter!(ax3, [0.0], [0.0], color=:gray, markersize=10, marker=:circle, label="FFT")
    end

    # Singlepass - orange dashed (offset-aligned)
    if haskey(results, :singlepass)
        traj_sp = DC.drift_trajectory(results[:singlepass].model)
        # Align to ground truth at frame 1 (drift correction only recovers shape, not absolute offset)
        offset_x = traj_true.true_x[1] - traj_sp.x[1]
        offset_y = traj_true.true_y[1] - traj_sp.y[1]
        sp_x = traj_sp.x .+ offset_x
        sp_y = traj_sp.y .+ offset_y
        lines!(ax1, traj_sp.frames, sp_x, color=:orange, linewidth=2, linestyle=:dash, label="Singlepass")
        lines!(ax2, traj_sp.frames, sp_y, color=:orange, linewidth=2, linestyle=:dash)
        lines!(ax3, sp_x, sp_y, color=:orange, linewidth=2, linestyle=:dash, label="Singlepass")
    end

    # Iterative - red dotdash - only for multi-dataset (offset-aligned)
    if haskey(results, :iterative)
        traj_it = DC.drift_trajectory(results[:iterative].model)
        offset_x = traj_true.true_x[1] - traj_it.x[1]
        offset_y = traj_true.true_y[1] - traj_it.y[1]
        it_x = traj_it.x .+ offset_x
        it_y = traj_it.y .+ offset_y
        lines!(ax1, traj_it.frames, it_x, color=:red, linewidth=2, linestyle=:dashdot, label="Iterative")
        lines!(ax2, traj_it.frames, it_y, color=:red, linewidth=2, linestyle=:dashdot)
        lines!(ax3, it_x, it_y, color=:red, linewidth=2, linestyle=:dashdot, label="Iterative")
    end

    axislegend(ax1, position=:lt)
    axislegend(ax3, position=:lt)

    return fig
end

"""
Compute per-dataset RMSD and max error.
"""
function compute_per_dataset_metrics(smld_orig, smld_corrected)
    n_datasets = smld_orig.n_datasets
    per_ds_rmsd = Float64[]
    per_ds_max = Float64[]

    for ds = 1:n_datasets
        mask_orig = [e.dataset == ds for e in smld_orig.emitters]
        mask_corr = [e.dataset == ds for e in smld_corrected.emitters]

        smld_orig_ds = DC.filter_emitters(smld_orig, mask_orig)
        smld_corr_ds = DC.filter_emitters(smld_corrected, mask_corr)

        rmsd_ds = compute_rmsd(smld_orig_ds, smld_corr_ds)
        push!(per_ds_rmsd, rmsd_ds)

        # Max error for this dataset
        max_err = 0.0
        for i in eachindex(smld_orig_ds.emitters)
            dx = (smld_corr_ds.emitters[i].x - smld_orig_ds.emitters[i].x) * 1000.0
            dy = (smld_corr_ds.emitters[i].y - smld_orig_ds.emitters[i].y) * 1000.0
            err = sqrt(dx^2 + dy^2)
            max_err = max(max_err, err)
        end
        push!(per_ds_max, max_err)
    end

    return (rmsd=per_ds_rmsd, max_error=per_ds_max)
end

"""
Save detailed stats.md with all tier results and per-dataset metrics.
"""
function save_detailed_stats(stats::Dict, scenario::Symbol, results::Dict,
                              smld_orig, per_dataset_metrics::Dict)
    dir = ensure_output_dir(scenario; clean=false)
    path = joinpath(dir, "stats.md")

    open(path, "w") do io
        println(io, "# Drift Correction: $(uppercase(string(scenario)))")
        println(io)

        # Quality tier summary
        println(io, "## Quality Tier Results")
        println(io)
        println(io, "| Tier | RMSD (nm) | Mean Error (nm) | Max Error (nm) | Iterations |")
        println(io, "|------|-----------|-----------------|----------------|------------|")

        for tier in [:fft, :singlepass, :iterative]
            if haskey(results, tier)
                rmsd = stats["rmsd_$(tier)_nm"]
                mean_err = stats["mean_error_$(tier)_nm"]
                max_err = stats["max_error_$(tier)_nm"]
                iters = results[tier].iterations
                @printf(io, "| %s | %.2f | %.2f | %.2f | %d |\n", tier, rmsd, mean_err, max_err, iters)
            end
        end
        println(io)

        println(io, "> **Expected Performance**: RMSD < 1 nm for well-constrained simulated data.")
        println(io)

        # Per-dataset table (for all entropy-based tiers)
        if smld_orig.n_datasets > 1
            for tier in [:singlepass, :iterative]
                if haskey(per_dataset_metrics, tier)
                    println(io, "## Per-Dataset Metrics ($(tier))")
                    println(io)
                    println(io, "| Dataset | RMSD (nm) | Max Error (nm) |")
                    println(io, "|---------|-----------|----------------|")

                    metrics = per_dataset_metrics[tier]
                    for ds in 1:smld_orig.n_datasets
                        @printf(io, "| %d | %.2f | %.2f |\n", ds, metrics.rmsd[ds], metrics.max_error[ds])
                    end
                    println(io)
                end
            end
        end

        # Entropy metrics
        if haskey(stats, "entropy_original")
            println(io, "## Entropy Metrics")
            println(io)
            println(io, "| State | Entropy |")
            println(io, "|-------|---------|")
            @printf(io, "| Original | %.4f |\n", stats["entropy_original"])
            @printf(io, "| Drifted | %.4f |\n", stats["entropy_drifted"])
            @printf(io, "| Corrected | %.4f |\n", stats["entropy_corrected"])
            println(io)
        end

        # Simulation parameters
        println(io, "## Simulation Parameters")
        println(io)
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        @printf(io, "| Emitters | %d |\n", stats["n_emitters"])
        @printf(io, "| Datasets | %d |\n", stats["n_datasets"])
        @printf(io, "| Frames/dataset | %d |\n", stats["n_frames"])
        @printf(io, "| Polynomial degree | %d |\n", stats["degree"])
        @printf(io, "| Drift scale | %.3f μm |\n", stats["drift_scale_um"])
        if haskey(stats, "inter_scale_um")
            @printf(io, "| Inter-shift scale | %.3f μm |\n", stats["inter_scale_um"])
        end
        @printf(io, "| Seed | %d |\n", stats["seed"])
        println(io)
    end

    println("  Saved: stats.md")
    return path
end

# =============================================================================
# Main scenario runner
# =============================================================================

"""
    run_scenario(scenario::Symbol; kwargs...)

Run applicable quality tiers for a scenario and save outputs.

Single mode: only singlepass (FFT/iterative don't help with single dataset)
Continuous/Registered: all three tiers
"""
function run_scenario(scenario::Symbol;
        density::Real = 2.5,
        n_datasets::Int = scenario == :single ? 1 : 5,
        n_frames::Int = 2000,
        degree::Int = 2,
        drift_scale::Real = 0.2,
        inter_scale::Real = 0.3,
        seed::Int = 42,
        verbose::Bool = true)

    verbose && println("=" ^ 60)
    verbose && println("SCENARIO: $(uppercase(string(scenario)))")
    verbose && println("=" ^ 60)

    # Determine dataset_mode and applicable tiers
    if scenario == :single
        dataset_mode = :continuous  # doesn't matter for single
        tiers = [:singlepass]  # FFT/iterative don't help with 1 dataset
    else
        dataset_mode = scenario == :registered ? :registered : :continuous
        tiers = [:fft, :singlepass, :iterative]
    end

    # Output directory
    dir = ensure_output_dir(scenario)

    # =========================================================================
    # 1. Generate test data
    # =========================================================================
    verbose && println("\n[1/4] Generating test data...")

    smld_orig = generate_test_smld(;
        density = density,
        n_datasets = n_datasets,
        n_frames = n_frames,
        seed = seed
    )

    verbose && println("  Emitters: $(length(smld_orig.emitters))")
    verbose && println("  Datasets: $(smld_orig.n_datasets)")
    verbose && println("  Frames per dataset: $(smld_orig.n_frames)")

    # =========================================================================
    # 2. Apply drift
    # =========================================================================
    verbose && println("\n[2/4] Applying drift...")

    result_drift = apply_test_drift(smld_orig, scenario;
        degree = degree,
        drift_scale = drift_scale,
        inter_scale = inter_scale,
        seed = seed
    )
    smld_drifted = result_drift.smld_drifted
    model_true = result_drift.model_true

    verbose && println("  Drift scale: $(drift_scale) μm")
    if scenario == :registered
        verbose && println("  Inter-shift scale: $(inter_scale) μm")
    end

    # Save drifted renders
    verbose && println("  Saving drifted renders...")
    save_renders(smld_drifted, "drifted", dir)

    # =========================================================================
    # 3. Run applicable quality tiers
    # =========================================================================
    verbose && println("\n[3/4] Running drift correction...")

    results = Dict{Symbol, DC.DriftResult}()
    rmsd_results = Dict{Symbol, Float64}()
    mean_error_results = Dict{Symbol, Float64}()
    max_error_results = Dict{Symbol, Float64}()
    per_dataset_metrics = Dict{Symbol, NamedTuple}()

    for tier in tiers
        verbose && print("  :$tier...")
        t = @elapsed begin
            result = DC.driftcorrect(smld_drifted;
                degree = degree,
                dataset_mode = dataset_mode,
                quality = tier,
                max_iterations = 5
            )
            results[tier] = result

            # Use aligned metrics (drift correction recovers shape, not absolute offset)
            rmsd_results[tier] = compute_aligned_rmsd(smld_orig, result.smld)

            # Mean and max error (aligned)
            residuals = compute_aligned_residuals(smld_orig, result.smld)
            mean_error_results[tier] = mean(residuals.total_nm)
            max_error_results[tier] = maximum(residuals.total_nm)

            # Per-dataset metrics
            if n_datasets > 1
                per_dataset_metrics[tier] = compute_per_dataset_metrics(smld_orig, result.smld)
            end
        end
        verbose && @printf(" RMSD: %.2f nm, Mean: %.2f nm, Max: %.2f nm (%.1fs)\n",
                          rmsd_results[tier], mean_error_results[tier], max_error_results[tier], t)

        # Save tier renders
        save_renders(result.smld, string(tier), dir)
    end

    if length(tiers) > 1
        verbose && println("\n  Quality Tier Summary:")
        for tier in tiers
            verbose && @printf("    :%s  RMSD: %.2f nm, Mean: %.2f nm, Max: %.2f nm\n",
                              tier, rmsd_results[tier], mean_error_results[tier], max_error_results[tier])
        end
    end

    # =========================================================================
    # 4. Generate plots and stats
    # =========================================================================
    verbose && println("\n[4/4] Generating plots and stats...")

    # Compare trajectories to ground truth
    traj_true = compare_trajectories(model_true, results[:singlepass].model)

    # Per-tier trajectory comparison plots
    for tier in tiers
        fig = plot_single_trajectory(traj_true, results[tier].model;
                                      tier=tier,
                                      title="$(uppercase(string(scenario))) - $tier")
        save(joinpath(dir, "trajectory_$(tier).png"), fig)
        verbose && println("  Saved: trajectory_$(tier).png")
    end

    # Combined trajectory comparison plot (all tiers)
    show_fft = scenario != :single
    fig = plot_tier_trajectories(traj_true, results;
                                   title_suffix=" ($(uppercase(string(scenario))))",
                                   show_fft=show_fft)
    save(joinpath(dir, "trajectory_comparison.png"), fig)
    verbose && println("  Saved: trajectory_comparison.png")

    # Stats
    stats = Dict{String, Any}(
        "scenario" => string(scenario),
        "n_emitters" => length(smld_orig.emitters),
        "n_datasets" => n_datasets,
        "n_frames" => n_frames,
        "degree" => degree,
        "drift_scale_um" => drift_scale,
        "seed" => seed,
    )

    # Add per-tier results
    for tier in tiers
        stats["rmsd_$(tier)_nm"] = rmsd_results[tier]
        stats["mean_error_$(tier)_nm"] = mean_error_results[tier]
        stats["max_error_$(tier)_nm"] = max_error_results[tier]
    end

    if scenario == :registered
        stats["inter_scale_um"] = inter_scale
    end

    # Add entropy metrics
    entropy_orig = compute_entropy_metrics(smld_orig)
    entropy_drifted = compute_entropy_metrics(smld_drifted)
    entropy_corrected = compute_entropy_metrics(results[:singlepass].smld)

    stats["entropy_original"] = entropy_orig.ub_entropy
    stats["entropy_drifted"] = entropy_drifted.ub_entropy
    stats["entropy_corrected"] = entropy_corrected.ub_entropy

    # Save detailed stats
    save_detailed_stats(stats, scenario, results, smld_orig, per_dataset_metrics)

    verbose && println("\n" * "=" ^ 60)
    verbose && println("$(uppercase(string(scenario))) COMPLETE")
    verbose && println("=" ^ 60)

    return (
        smld_orig = smld_orig,
        smld_drifted = smld_drifted,
        model_true = model_true,
        results = results,
        rmsd = rmsd_results,
        mean_error = mean_error_results,
        max_error = max_error_results,
        stats = stats
    )
end

# =============================================================================
# Run all scenarios
# =============================================================================

function run_all(; verbose::Bool = true)
    all_results = Dict{Symbol, Any}()

    for scenario in [:single, :continuous, :registered]
        all_results[scenario] = run_scenario(scenario; verbose=verbose)
        println()
    end

    # Summary
    println("\n" * "=" ^ 60)
    println("ALL SCENARIOS COMPLETE")
    println("=" ^ 60)
    println("\nRMSD Summary (nm):")
    println("-" ^ 60)
    @printf("%-12s %10s %12s %10s\n", "Scenario", "FFT", "Singlepass", "Iterative")
    println("-" ^ 60)

    for scenario in [:single, :continuous, :registered]
        r = all_results[scenario].rmsd
        if scenario == :single
            @printf("%-12s %10s %12.2f %10s\n", scenario, "n/a", r[:singlepass], "n/a")
        else
            @printf("%-12s %10.2f %12.2f %10.2f\n", scenario, r[:fft], r[:singlepass], r[:iterative])
        end
    end
    println("-" ^ 60)

    return all_results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
