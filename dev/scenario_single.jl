# scenario_single.jl - Single dataset drift correction diagnostics
#
# Tests intra-dataset drift correction only (no inter-dataset alignment).
# Validates basic Legendre polynomial fitting and entropy optimization.

using Pkg
Pkg.activate(@__DIR__)

include("DiagnosticHelpers.jl")
using .DiagnosticHelpers
using SMLMDriftCorrection
using Printf
using Statistics

const DC = SMLMDriftCorrection
const SCENARIO = :single

"""
    run_single_diagnostics(; kwargs...)

Run diagnostic suite for single-dataset intra-drift correction.

# How Entropy-Based Drift Correction Works

The algorithm minimizes the entropy (spread) of the point cloud. For this to
recover the TRUE drift, we need REPEATED observations of the same underlying
structure across different frames.

With SMLMSim's blinking model (k_on=0.001, k_off=50):
- Molecules turn ON once every ~1000 seconds
- Stay ON for ~20ms (1-2 frames at 50fps)
- Most molecules blink ONCE then never again within 1000 frames

So entropy minimization works by aligning the Nmer PATTERNS (multiple copies
of hexamers at different positions), not by tracking individual molecules.

# Keyword Arguments
- `density`: emitters/μm² (default: 10.0 - higher for more constraint)
- `n_frames`: frames per dataset (default: 5000 - longer acquisition)
- `degree`: polynomial degree (default: 2)
- `drift_scale`: drift amplitude in μm (default: 0.05 - modest drift)
- `seed`: random seed (default: 42)
- `verbose`: print progress (default: true)

# Expected Performance
With sufficient localizations and structured data:
- RMSD < 50 nm: Good drift recovery
- RMSD > 200 nm: Algorithm may have found local minimum

Note: Single-dataset intra-drift is inherently harder than inter-dataset
alignment because there's less redundant information.
"""
function run_single_diagnostics(;
        density::Real = 2.5,       # emitters/μm² (sparser Nmer patterns)
        n_frames::Int = 2000,      # 2000 frames (40 sec at 50fps)
        degree::Int = 2,           # Lower degree is easier to recover
        drift_scale::Real = 0.2,   # 200nm scale drift
        seed::Int = 42,
        verbose::Bool = true)

    verbose && println("=" ^ 60)
    verbose && println("SINGLE DATASET DIAGNOSTICS")
    verbose && println("=" ^ 60)

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

    n_emitters = length(smld_orig.emitters)
    locs_per_frame = n_emitters / n_frames
    verbose && println("  Total localizations: $n_emitters")
    verbose && println("  Frames: $(smld_orig.n_frames)")
    verbose && @printf("  Localizations/frame: %.1f\n", locs_per_frame)
    verbose && println("  FOV: 25.6 × 25.6 μm² (256 × 256 pixels @ 100nm)")

    if locs_per_frame < 1.0
        verbose && println("  WARNING: Very sparse data (<1 loc/frame) - drift recovery may be unreliable")
    end

    # =========================================================================
    # 1b. Sanity check: apply + correct with SAME model should give exact recovery
    # =========================================================================
    verbose && println("\n[1b] Sanity check: apply + correct with same model...")

    # Create a test model
    test_model = DC.LegendrePolynomial(smld_orig; degree=degree, initialize="random", rscale=drift_scale)
    test_model.inter[1].dm .= 0.0  # zero inter for single dataset
    smld_test_drifted = DC.applydrift(smld_orig, test_model)
    smld_test_recovered = DC.correctdrift(smld_test_drifted, test_model)

    # This should be essentially zero (numerical precision only)
    sanity_rmsd = compute_rmsd(smld_orig, smld_test_recovered)
    verbose && @printf("  Sanity check RMSD (should be ~0): %.6f nm\n", sanity_rmsd)
    if sanity_rmsd > 0.001
        verbose && println("  WARNING: Framework sanity check failed!")
    else
        verbose && println("  Framework sanity check PASSED")
    end

    # =========================================================================
    # 2. Apply drift
    # =========================================================================
    verbose && println("\n[2/6] Applying drift...")

    result_drift = apply_test_drift(smld_orig, SCENARIO;
        degree = degree,
        drift_scale = drift_scale,
        seed = seed
    )
    smld_drifted = result_drift.smld_drifted
    model_true = result_drift.model_true

    verbose && println("  Drift scale: $(drift_scale) μm")
    verbose && println("  Degree: $degree")

    # =========================================================================
    # 3. Run drift correction (singlepass only)
    # =========================================================================
    # Note: For single-dataset correction, only :singlepass is applicable.
    # - :fft requires multiple datasets for inter-dataset alignment
    # - :iterative is for inter↔intra convergence across datasets
    # Single-dataset only has intra-drift, so singlepass entropy optimization
    # is the appropriate method.
    verbose && println("\n[3/6] Running drift correction (singlepass only)...")

    (smld_corrected, info) = DC.driftcorrect(smld_drifted; degree=degree, quality=:singlepass)
    model_recovered = info.model

    verbose && @printf("  RMSD: %.2f nm (iterations=%d, converged=%s)\n",
                      compute_rmsd(smld_orig, smld_corrected), info.iterations, info.converged)

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

    verbose && @printf("  Entropy before (drifted): %.4f\n", entropy_before.ub_entropy)
    verbose && @printf("  Entropy after (corrected): %.4f\n", entropy_after.ub_entropy)
    verbose && @printf("  Entropy reduction: %.1f%%\n", entropy_reduction)

    # KEY DIAGNOSTIC: Compare entropy of original (ground truth) vs corrected
    entropy_original = compute_entropy_metrics(smld_orig)
    verbose && @printf("\n  ** ENTROPY DIAGNOSTIC **\n")
    verbose && @printf("  Entropy of ORIGINAL (undrifted): %.4f\n", entropy_original.ub_entropy)
    verbose && @printf("  Entropy of DRIFTED: %.4f\n", entropy_before.ub_entropy)
    verbose && @printf("  Entropy of CORRECTED (found): %.4f\n", entropy_after.ub_entropy)
    verbose && @printf("  Delta (corrected - original): %.4f\n", entropy_after.ub_entropy - entropy_original.ub_entropy)

    # If corrected entropy is HIGHER than original, algorithm is failing
    if entropy_after.ub_entropy > entropy_original.ub_entropy
        verbose && println("  WARNING: Corrected entropy > original entropy!")
        verbose && println("           Algorithm found wrong solution.")
    elseif entropy_after.ub_entropy < entropy_original.ub_entropy - 1.0
        verbose && println("  WARNING: Corrected entropy << original entropy!")
        verbose && println("           Algorithm may have over-compressed the point cloud.")
    else
        verbose && println("  OK: Corrected entropy ≈ original entropy")
    end

    # Trajectory comparison
    traj = compare_trajectories(model_true, model_recovered)

    # =========================================================================
    # 5. Generate plots (singlepass only for single-dataset)
    # =========================================================================
    verbose && println("\n[5/6] Generating plots...")

    # KEY DIAGNOSTIC: Drift comparison (GT vs Recovered in X, Y, and X-Y plane)
    fig_drift = plot_drift_comparison(traj)
    save_figure(fig_drift, SCENARIO, "singlepass_drift_comparison.png")

    # Trajectory comparison (alternative view)
    fig_traj = plot_trajectory_comparison(traj; title_suffix=" (Single Dataset, :singlepass)")
    save_figure(fig_traj, SCENARIO, "singlepass_trajectory_comparison.png")

    # Render suite: 6 images (histogram, circles, gaussian for drifted and corrected)
    save_render_suite(smld_drifted, smld_corrected, SCENARIO; prefix="singlepass_")

    # Residual histogram
    fig_hist = plot_residuals(residuals)
    save_figure(fig_hist, SCENARIO, "singlepass_residual_histogram.png")

    # Residual scatter
    fig_scatter = plot_residual_scatter(residuals)
    save_figure(fig_scatter, SCENARIO, "singlepass_residual_scatter.png")

    # RMSD vs frame
    fig_rmsd = plot_rmsd_vs_frame(per_frame)
    save_figure(fig_rmsd, SCENARIO, "singlepass_rmsd_vs_frame.png")

    # Coefficient comparison (single-dataset specific)
    fig_coeff = plot_coefficient_comparison(model_true, model_recovered)
    save_figure(fig_coeff, SCENARIO, "singlepass_coefficient_comparison.png")

    # =========================================================================
    # 6. Save statistics
    # =========================================================================
    verbose && println("\n[6/6] Saving statistics...")

    stats = Dict(
        "rmsd_nm" => rmsd_nm,
        "mean_error_nm" => mean_error,
        "max_error_nm" => max_error,
        "entropy_original" => entropy_original.ub_entropy,
        "entropy_drifted" => entropy_before.ub_entropy,
        "entropy_corrected" => entropy_after.ub_entropy,
        "entropy_reduction_pct" => entropy_reduction,
        "entropy_delta_vs_original" => entropy_after.ub_entropy - entropy_original.ub_entropy,
        "sanity_check_rmsd_nm" => sanity_rmsd,
        "n_emitters" => n_emitters,
        "n_frames" => n_frames,
        "locs_per_frame" => locs_per_frame,
        "degree" => degree,
        "drift_scale_um" => drift_scale,
        "seed" => seed,
    )

    # Generate notes for stats file
    notes = """
**Interpretation (for sparse single-dataset data):**
- RMSD < 20 nm: Excellent drift recovery
- RMSD < 100 nm: Good - acceptable for sparse data
- RMSD < 500 nm: Marginal - likely local minimum
- RMSD > 500 nm: Poor - wrong solution

**Why Single-Dataset is Hard:**
The SMLMSim blinking model (k_on=0.001, k_off=50) means molecules blink ~once
then never again within typical acquisitions. With <1 localization/frame,
there's insufficient constraint to uniquely determine the drift polynomial.

**Entropy Diagnostic:**
- `entropy_delta_vs_original`: Should be ~0 if drift correctly recovered
- If negative and large: algorithm over-compressed (found different minimum)
- If positive: algorithm made things worse

**Sanity Check:**
- `sanity_check_rmsd_nm`: apply + correct with SAME model (should be ~0 nm)
- Verifies the framework math is correct

**For Better Results:**
- Use multi-dataset alignment (inter-dataset correction)
- Increase density or acquisition time
- Use lower polynomial degree
"""

    save_stats_md(stats, SCENARIO; notes=notes, filename="stats_singlepass.md")

    # Save stats_all.md (only singlepass for single-dataset)
    dir = ensure_output_dir(SCENARIO; clean=false)
    open(joinpath(dir, "stats_all.md"), "w") do io
        println(io, "# Quality Tier Comparison: SINGLE")
        println(io)
        println(io, "## Applicable Methods")
        println(io)
        println(io, "For single-dataset correction, only **singlepass** is applicable:")
        println(io, "- **FFT**: Requires multiple datasets for inter-dataset alignment")
        println(io, "- **iterative**: Requires multiple datasets for inter↔intra convergence")
        println(io)
        println(io, "## Results")
        println(io)
        println(io, "| Quality Tier | RMSD (nm) | Iterations | Converged |")
        println(io, "|--------------|-----------|------------|-----------|")
        @printf(io, "| **singlepass** | %.2f | %d | %s |\n",
                rmsd_nm, info.iterations, info.converged)
        println(io)
        println(io, "## Parameters")
        println(io)
        println(io, "- Emitters: $n_emitters")
        println(io, "- Frames: $n_frames")
        println(io, "- Locs/frame: $(round(locs_per_frame, digits=1))")
        println(io, "- Degree: $degree")
        println(io, "- Drift scale: $drift_scale μm")
        println(io, "- Seed: $seed")
    end
    verbose && println("  Saved: stats_all.md")

    verbose && println("\n" * "=" ^ 60)
    verbose && println("SINGLE DATASET DIAGNOSTICS COMPLETE")
    verbose && @printf("Final RMSD: %.2f nm\n", rmsd_nm)
    verbose && @printf("Localizations/frame: %.1f\n", locs_per_frame)

    # Realistic thresholds for sparse single-dataset data
    if rmsd_nm < 20.0
        verbose && println("EXCELLENT: Drift recovered very well")
    elseif rmsd_nm < 100.0
        verbose && println("GOOD: Drift recovery acceptable for sparse data")
    elseif rmsd_nm < 500.0
        verbose && println("MARGINAL: Drift partially recovered - may be local minimum")
    else
        verbose && println("POOR: Algorithm likely found wrong solution")
    end

    verbose && println("\nNote: Single-dataset intra-drift correction is fundamentally")
    verbose && println("under-constrained with sparse blinking data. Multi-dataset")
    verbose && println("inter-alignment typically performs better.")
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

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_single_diagnostics()
end
