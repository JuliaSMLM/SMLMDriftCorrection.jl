# Demo: position_frame_correlation diagnostic
#
# Compares the position-frame correlation diagnostic across three SMLDs:
#   1. Perfect (no drift)         - expect near-zero |corr|
#   2. Drifted (uncorrected)      - expect high |corr|
#   3. Corrected (post drift fix) - expect low |corr|
#
# Note: this script assumes `position_frame_correlation` is exported (or
# accessible as `DC.position_frame_correlation`) from SMLMDriftCorrection.
# It may fail at the diagnostic call until that function is merged.

using Pkg
Pkg.activate(temp=true)
Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.add(["SMLMSim", "SMLMData", "Statistics", "Random", "Printf"])

using SMLMDriftCorrection
using SMLMSim
using SMLMData
using Statistics
using Random
using Printf

const DC = SMLMDriftCorrection

function main()
    println("="^60)
    println("POSITION-FRAME CORRELATION DIAGNOSTIC")
    println("="^60)

    # --- Generate synthetic data (matches test_warmstart_minimal.jl) ---
    println("Generating synthetic data...")
    params_2d = StaticSMLMConfig(
        10.0,     # density: emitters per μm²
        0.13,     # σ_psf
        30,       # minphotons
        5,        # ndatasets
        5000,     # nframes per dataset
        50.0,     # framerate
        2,        # ndims
        [0.0, 1.0]
    )
    (smld_orig, _) = simulate(
        params_2d;
        pattern=Nmer2D(n=6, d=0.2),
        molecule=GenericFluor(; photons=5000.0, k_on=0.02, k_off=50.0),
        camera=IdealCamera(1:64, 1:64, 0.1)
    )
    println("  $(length(smld_orig.emitters)) emitters, $(smld_orig.n_datasets) datasets")

    # --- Apply known drift ---
    Random.seed!(42)
    drift_model = DC.LegendrePolynomial(smld_orig;
        degree=3, initialize="random", rscale=0.1)
    smld_drifted = DC.applydrift(smld_orig, drift_model)

    # --- Run drift correction ---
    println("Running drift correction (iterative)...")
    (smld_corrected, info) = driftcorrect(smld_drifted;
        degree=3, quality=:iterative, dataset_mode=:registered,
        shift_scale=0.05, max_iterations=10, verbose=0)
    @printf "  converged=%s  iter=%d  elapsed=%.1fs\n" info.converged info.iterations info.elapsed_s

    # --- Run diagnostic on each SMLD (intra mode) ---
    println("\nRunning position_frame_correlation (intra mode)...")
    diag_orig      = position_frame_correlation(smld_orig;      K=20, mode=:intra)
    diag_drifted   = position_frame_correlation(smld_drifted;   K=20, mode=:intra)
    diag_corrected = position_frame_correlation(smld_corrected; K=20, mode=:intra)

    # --- Comparison table ---
    println()
    println("="^60)
    println("POSITION-FRAME CORRELATION DIAGNOSTIC")
    println("="^60)
    @printf "%-23s%-17s%-17s\n" "" "mean|corr_x|" "mean|corr_y|"
    @printf "%-23s%-17.3f%-17.3f\n" "Perfect (no drift):" diag_orig.summary.mean_abs_corr_x      diag_orig.summary.mean_abs_corr_y
    @printf "%-23s%-17.3f%-17.3f\n" "Drifted:"            diag_drifted.summary.mean_abs_corr_x   diag_drifted.summary.mean_abs_corr_y
    @printf "%-23s%-17.3f%-17.3f\n" "Corrected:"          diag_corrected.summary.mean_abs_corr_x diag_corrected.summary.mean_abs_corr_y
    println("="^60)

    # --- Inter mode on corrected SMLD ---
    println("\nRunning position_frame_correlation (inter mode) on corrected SMLD...")
    diag_inter = position_frame_correlation(smld_corrected; K=20, mode=:inter)
    @printf "  inter corr_x = %.3f\n" diag_inter.corr_x
    @printf "  inter corr_y = %.3f\n" diag_inter.corr_y

    # --- Per-dataset breakdown for corrected case ---
    println("\nPer-dataset breakdown (corrected, intra mode):")
    @printf "  %-9s%-9s%-13s%-13s\n" "dataset" "n_locs" "corr_x" "corr_y"
    for entry in diag_corrected.per_dataset
        @printf "  %-9d%-9d%-13.4f%-13.4f\n" entry.dataset entry.n_locs entry.corr_x entry.corr_y
    end

    println("\nDone.")
end

main()
