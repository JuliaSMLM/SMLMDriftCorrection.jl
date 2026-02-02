# scenario_registered.jl - Registered mode drift correction diagnostics
#
# Tests multi-dataset correction with independent inter-dataset offsets.
# Stage is registered between acquisitions, so each dataset has an independent
# offset plus intra-dataset drift.

using Pkg
Pkg.activate(@__DIR__)

include("DiagnosticHelpers.jl")
using .DiagnosticHelpers
using SMLMDriftCorrection
using DataFrames
using Printf
using Statistics
using SMLMRender
using CairoMakie

const DC = SMLMDriftCorrection
const SCENARIO = :registered

"""
    run_registered_diagnostics(; kwargs...)

Run full diagnostic suite for registered mode scenario.

# Keyword Arguments
- `density`: emitters/μm² (default: 5.0)
- `n_datasets`: number of datasets (default: 5)
- `n_frames`: frames per dataset (default: 1000)
- `degree`: polynomial degree (default: 2)
- `drift_scale`: intra-drift amplitude in μm (default: 0.1)
- `inter_scale`: inter-dataset offset scale in μm (default: 0.2)
- `seed`: random seed (default: 42)
- `verbose`: print progress (default: true)
"""
function run_registered_diagnostics(;
        density::Real = 2.5,        # emitters/μm² (sparser for clearer visualization)
        n_datasets::Int = 5,
        n_frames::Int = 2000,       # 2000 frames per dataset
        degree::Int = 2,
        drift_scale::Real = 0.2,    # ~200nm intra-drift
        inter_scale::Real = 0.3,    # ~300nm inter-dataset offsets
        seed::Int = 42,
        verbose::Bool = true)

    verbose && println("=" ^ 60)
    verbose && println("REGISTERED MODE DIAGNOSTICS")
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

    # =========================================================================
    # 2. Apply drift
    # =========================================================================
    verbose && println("\n[2/6] Applying drift (registered mode)...")

    result_drift = apply_test_drift(smld_orig, SCENARIO;
        degree = degree,
        drift_scale = drift_scale,
        inter_scale = inter_scale,
        seed = seed
    )
    smld_drifted = result_drift.smld_drifted
    model_true = result_drift.model_true

    verbose && println("  Intra-drift scale: $(drift_scale) μm")
    verbose && println("  Inter-shift scale: $(inter_scale) μm")
    verbose && println("  Degree: $degree")

    # Show true inter-shifts
    verbose && println("  True inter-shifts:")
    for ds = 1:n_datasets
        x, y = model_true.inter[ds].dm[1], model_true.inter[ds].dm[2]
        verbose && @printf("    DS %d: (%.4f, %.4f) μm\n", ds, x, y)
    end

    # =========================================================================
    # 3. Run drift correction (all quality tiers)
    # =========================================================================
    verbose && println("\n[3/6] Running drift correction (all quality tiers)...")

    # Run all three quality tiers
    tier_results = Dict{Symbol, NamedTuple}()

    for quality in [:fft, :singlepass, :iterative]
        verbose && println("\n  Running :$quality...")
        (smld_corr, info) = DC.driftcorrect(smld_drifted;
            degree = degree,
            dataset_mode = :registered,
            quality = quality,
            max_iterations = 5
        )

        # Compute RMSD for this tier
        rmsd_tier = compute_rmsd(smld_orig, smld_corr)
        verbose && @printf("    RMSD: %.2f nm (iterations=%d, converged=%s)\n",
                          rmsd_tier, info.iterations, info.converged)

        tier_results[quality] = (
            smld = smld_corr,
            info = info,
            rmsd_nm = rmsd_tier
        )
    end

    # Use singlepass as the "main" result for detailed analysis
    smld_corrected = tier_results[:singlepass].smld
    model_recovered = tier_results[:singlepass].info.model

    verbose && println("\n  === Quality Tier Summary ===")
    verbose && @printf("    :fft        RMSD: %.2f nm\n", tier_results[:fft].rmsd_nm)
    verbose && @printf("    :singlepass RMSD: %.2f nm\n", tier_results[:singlepass].rmsd_nm)
    verbose && @printf("    :iterative  RMSD: %.2f nm\n", tier_results[:iterative].rmsd_nm)

    # Show recovered inter-shifts
    verbose && println("  Recovered inter-shifts:")
    for ds = 1:n_datasets
        x, y = model_recovered.inter[ds].dm[1], model_recovered.inter[ds].dm[2]
        verbose && @printf("    DS %d: (%.4f, %.4f) μm\n", ds, x, y)
    end

    # =========================================================================
    # 4. Compute metrics
    # =========================================================================
    verbose && println("\n[4/6] Computing metrics...")

    # Overall RMSD
    rmsd_nm = compute_rmsd(smld_orig, smld_corrected)
    verbose && @printf("  Overall RMSD: %.2f nm\n", rmsd_nm)

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

    # Inter-shift comparison
    df_inter = inter_shift_comparison(model_true, model_recovered)
    verbose && println("\n  Inter-shift errors (nm):")
    for row in eachrow(df_inter)
        verbose && @printf("    DS %d: %.1f nm\n", row.dataset, row.error_total_nm)
    end

    # Intra-drift recovery per dataset (isolated from inter-shift)
    df_intra = analyze_intra_drift_per_dataset(model_true, model_recovered)
    verbose && println("\n  Intra-drift errors (isolated from inter-shift):")
    for row in eachrow(df_intra)
        verbose && @printf("    DS %d: max %.1f nm, mean %.1f nm\n",
                          row.dataset, row.max_intra_error_nm, row.mean_intra_error_nm)
    end

    # Per-dataset RMSD
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
    # 5. Generate plots
    # =========================================================================
    verbose && println("\n[5/6] Generating plots...")

    # Trajectory comparison
    fig_traj = plot_trajectory_comparison(traj; title_suffix=" (Registered Mode)")
    save_figure(fig_traj, SCENARIO, "trajectory_comparison.png")

    # Render suite with dataset colors (histogram, circles, gaussian + dataset overlays)
    save_render_suite(smld_drifted, smld_corrected, SCENARIO; include_dataset_colors=true)

    # Residual histogram
    fig_hist = plot_residuals(residuals)
    save_figure(fig_hist, SCENARIO, "residual_histogram.png")

    # Residual scatter
    fig_scatter = plot_residual_scatter(residuals)
    save_figure(fig_scatter, SCENARIO, "residual_scatter.png")

    # RMSD vs frame
    fig_rmsd = plot_rmsd_vs_frame(per_frame)
    save_figure(fig_rmsd, SCENARIO, "rmsd_vs_frame.png")

    # Inter-shift comparison (registered-specific)
    fig_inter = plot_inter_shift_comparison(df_inter)
    save_figure(fig_inter, SCENARIO, "inter_shift_comparison.png")

    # Per-dataset table (combined intra and inter errors)
    df_per_ds = DataFrame(
        dataset = 1:n_datasets,
        rmsd_nm = per_ds_rmsd,
        inter_error_nm = df_inter.error_total_nm,
        intra_max_error_nm = df_intra.max_intra_error_nm,
        intra_mean_error_nm = df_intra.mean_intra_error_nm
    )

    # =========================================================================
    # 6. Save statistics
    # =========================================================================
    verbose && println("\n[6/6] Saving statistics...")

    stats = Dict(
        "rmsd_nm" => rmsd_nm,
        "rmsd_fft_nm" => tier_results[:fft].rmsd_nm,
        "rmsd_singlepass_nm" => tier_results[:singlepass].rmsd_nm,
        "rmsd_iterative_nm" => tier_results[:iterative].rmsd_nm,
        "mean_error_nm" => mean_error,
        "max_error_nm" => max_error,
        "entropy_before" => entropy_before.ub_entropy,
        "entropy_after" => entropy_after.ub_entropy,
        "entropy_reduction_pct" => entropy_reduction,
        "n_emitters" => length(smld_orig.emitters),
        "n_datasets" => n_datasets,
        "n_frames" => n_frames,
        "degree" => degree,
        "drift_scale_um" => drift_scale,
        "inter_scale_um" => inter_scale,
        "mean_inter_error_nm" => mean(df_inter.error_total_nm),
        "max_inter_error_nm" => maximum(df_inter.error_total_nm),
        "mean_intra_error_nm" => mean(df_intra.mean_intra_error_nm),
        "max_intra_error_nm" => maximum(df_intra.max_intra_error_nm),
        "seed" => seed,
    )

    save_stats_md(stats, SCENARIO)

    verbose && println("\n" * "=" ^ 60)
    verbose && println("REGISTERED MODE DIAGNOSTICS COMPLETE")
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
        df_inter = df_inter,
        df_per_ds = df_per_ds,
        tier_results = tier_results,
        stats = stats
    )
end

"""
    create_dataset_overlay(smld, n_datasets; zoom=20)

Create overlay with each dataset in a different color.
"""
function create_dataset_overlay(smld, n_datasets::Int; zoom::Int=20)
    # Split by dataset
    smld_list = []
    for ds = 1:n_datasets
        mask = [e.dataset == ds for e in smld.emitters]
        push!(smld_list, DC.filter_emitters(smld, mask))
    end

    # Color palette
    colors = [:red, :green, :blue, :orange, :purple, :cyan, :magenta, :yellow]

    # Render overlay
    img = render(smld_list,
                 colors = colors[1:min(n_datasets, length(colors))],
                 zoom = zoom)

    fig = Figure(size=(800, 800))
    ax = Axis(fig[1, 1], aspect=DataAspect(),
              title="Dataset Overlay (color by dataset)")
    image!(ax, rotr90(img))
    hidedecorations!(ax)

    return fig
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_registered_diagnostics()
end
