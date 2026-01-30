# _scenario_single.jl - Single dataset drift correction diagnostics (function definition only)
#
# This file is included by run_diagnostics.jl. For standalone use, see scenario_single.jl.

const SCENARIO_SINGLE = :single

"""
    run_single_diagnostics(; kwargs...)

Run full diagnostic suite for single-dataset scenario.

# Keyword Arguments
- `density`: emitters/μm² (default: 5.0)
- `n_frames`: frames per dataset (default: 1000)
- `degree`: polynomial degree (default: 2)
- `drift_scale`: drift amplitude in μm (default: 0.1)
- `seed`: random seed (default: 42)
- `verbose`: print progress (default: true)
"""
function run_single_diagnostics(;
        density::Real = 5.0,
        n_frames::Int = 1000,
        degree::Int = 2,
        drift_scale::Real = 0.1,
        seed::Int = 42,
        verbose::Bool = true)

    verbose && println("=" ^ 60)
    verbose && println("SINGLE DATASET DIAGNOSTICS")
    verbose && println("=" ^ 60)

    # Clean output directory at start
    ensure_output_dir(SCENARIO_SINGLE; clean=true)

    # =========================================================================
    # 1. Generate test data
    # =========================================================================
    verbose && println("\n[1/6] Generating test data...")

    smld_orig = generate_test_smld(;
        density = density,
        n_datasets = 1,
        n_frames = n_frames,
        seed = seed
    )

    verbose && println("  Emitters: $(length(smld_orig.emitters))")
    verbose && println("  Frames: $(smld_orig.n_frames)")

    # =========================================================================
    # 2. Apply drift
    # =========================================================================
    verbose && println("\n[2/6] Applying drift...")

    result_drift = apply_test_drift(smld_orig, SCENARIO_SINGLE;
        degree = degree,
        drift_scale = drift_scale,
        seed = seed
    )
    smld_drifted = result_drift.smld_drifted
    model_true = result_drift.model_true

    verbose && println("  Drift scale: $(drift_scale) μm")
    verbose && println("  Degree: $degree")

    # =========================================================================
    # 3. Run drift correction
    # =========================================================================
    verbose && println("\n[3/6] Running drift correction...")

    (; smld, model) = DC.driftcorrect(smld_drifted; degree=degree)
    smld_corrected = smld
    model_recovered = model

    verbose && println("  Correction complete")

    # =========================================================================
    # 4. Compute metrics
    # =========================================================================
    verbose && println("\n[4/6] Computing metrics...")

    # RMSD
    rmsd_nm = compute_rmsd(smld_orig, smld_corrected)
    verbose && @printf("  RMSD: %.2f nm\n", rmsd_nm)

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

    # =========================================================================
    # 5. Generate plots
    # =========================================================================
    verbose && println("\n[5/6] Generating plots...")

    # Trajectory comparison
    fig_traj = plot_trajectory_comparison(traj; title_suffix=" (Single Dataset)")
    save_figure(fig_traj, SCENARIO_SINGLE, "trajectory_comparison.png")

    # Render comparison
    fig_render = plot_render_comparison(smld_orig, smld_drifted, smld_corrected)
    save_figure(fig_render, SCENARIO_SINGLE, "render_comparison.png")

    # Overlay comparison
    fig_overlay = plot_overlay_comparison(smld_drifted, smld_corrected)
    save_figure(fig_overlay, SCENARIO_SINGLE, "overlay_comparison.png")

    # Residual histogram
    fig_hist = plot_residuals(residuals)
    save_figure(fig_hist, SCENARIO_SINGLE, "residual_histogram.png")

    # Residual scatter
    fig_scatter = plot_residual_scatter(residuals)
    save_figure(fig_scatter, SCENARIO_SINGLE, "residual_scatter.png")

    # RMSD vs frame
    fig_rmsd = plot_rmsd_vs_frame(per_frame)
    save_figure(fig_rmsd, SCENARIO_SINGLE, "rmsd_vs_frame.png")

    # Coefficient comparison (single-dataset specific)
    fig_coeff = plot_coefficient_comparison(model_true, model_recovered)
    save_figure(fig_coeff, SCENARIO_SINGLE, "coefficient_comparison.png")

    # =========================================================================
    # 6. Save statistics
    # =========================================================================
    verbose && println("\n[6/6] Saving statistics...")

    stats = Dict(
        "rmsd_nm" => rmsd_nm,
        "mean_error_nm" => mean_error,
        "max_error_nm" => max_error,
        "entropy_before" => entropy_before.ub_entropy,
        "entropy_after" => entropy_after.ub_entropy,
        "entropy_reduction_pct" => entropy_reduction,
        "n_emitters" => length(smld_orig.emitters),
        "n_frames" => n_frames,
        "degree" => degree,
        "drift_scale_um" => drift_scale,
        "seed" => seed,
    )

    save_stats(stats, SCENARIO_SINGLE)

    verbose && println("\n" * "=" ^ 60)
    verbose && println("SINGLE DATASET DIAGNOSTICS COMPLETE")
    verbose && @printf("Final RMSD: %.2f nm\n", rmsd_nm)
    verbose && println("=" ^ 60)

    return (
        smld_orig = smld_orig,
        smld_drifted = smld_drifted,
        smld_corrected = smld_corrected,
        model_true = model_true,
        model_recovered = model_recovered,
        stats = stats
    )
end
