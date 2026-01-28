## Demonstrate applying and correcting drift with inter-dataset offsets

using SMLMDriftCorrection
DC = SMLMDriftCorrection
using SMLMData
using SMLMSim
using CairoMakie
using Statistics
using Random

Random.seed!(42)  # For reproducibility

# Simulation parameters use physical units
# smld structures are in units of microns and frames
println("Simulating data...")
params_2d = StaticSMLMParams(
    10.0,     # density (ρ): emitters per μm² (higher for more localizations)
    0.13,     # σ_psf: PSF width in μm (130nm)
    50,       # minphotons: minimum photons for detection
    5,        # ndatasets: number of independent datasets
    10000,    # nframes: frames per dataset (more frames = more localizations)
    50.0,     # framerate: frames per second
    2,        # ndims: 2D
    [0.0, 1.0]  # zrange: z-range (not used for 2D)
)
smld_true, smld_model, smld_noisy = simulate(
    params_2d;
    pattern=Nmer2D(n=6, d=0.2),  # hexamer with 200nm diameter
    molecule=GenericFluor(; photons=5000.0, k_on=0.001, k_off=50.0),
    camera=IdealCamera(1:128, 1:128, 0.1)  # pixelsize in μm
)

# Report localizations per dataset
println("\nLocalizations per dataset:")
for ii in 1:smld_noisy.n_datasets
    n = count(e -> e.dataset == ii, smld_noisy.emitters)
    println("  Dataset $ii: $n localizations")
end

## Setup drift model with 200nm scale inter-dataset offsets
# Using LegendrePolynomial with degree=0 (no intra-dataset drift) to isolate inter-dataset alignment
drift_true = DC.LegendrePolynomial(smld_noisy; degree=0, initialize="zeros")
for ii in 2:smld_noisy.n_datasets
    drift_true.inter[ii].dm .= 0.2 * randn(2)  # 200nm scale offsets
end

println("\nTrue inter-dataset offsets (nm):")
for ii in 1:drift_true.ndatasets
    println("  Dataset $ii: $(round.(drift_true.inter[ii].dm * 1000, digits=1))")
end

println("\nApplying drift...")
smld_drift = DC.applydrift(smld_noisy, drift_true)

## Correct Drift using entropy minimization
println("\nRunning drift correction...")
result = driftcorrect(smld_drift; degree=0, maxn=100, verbose=1)
smld_corrected = result.smld

# Helper function to calculate RMSD
function calc_rmsd(smld_orig, smld_corr)
    x_orig = [e.x for e in smld_orig.emitters]
    y_orig = [e.y for e in smld_orig.emitters]
    x_corr = [e.x for e in smld_corr.emitters]
    y_corr = [e.y for e in smld_corr.emitters]
    return sqrt(mean((x_orig .- x_corr).^2 .+ (y_orig .- y_corr).^2)) * 1000
end

rmsd = calc_rmsd(smld_noisy, smld_corrected)

println("\n" * "="^60)
println("RESULTS")
println("="^60)
println("\nOverall RMSD: $(round(rmsd, digits=1)) nm")

println("\nInter-dataset offset recovery:")
println("  Dataset   True (nm)              Recovered (nm)         Error (nm)")
println("  " * "-"^70)
for ii in 1:result.model.ndatasets
    true_offset = drift_true.inter[ii].dm * 1000
    recov_offset = result.model.inter[ii].dm * 1000
    error = recov_offset .- true_offset
    error_mag = sqrt(sum(error.^2))
    println("  $ii        $(lpad.(round.(true_offset, digits=1), 7))    $(lpad.(round.(recov_offset, digits=1), 7))    $(round(error_mag, digits=1))")
end

# Extract coordinates for plotting
smld_noisy_x = [e.x for e in smld_noisy.emitters]
smld_noisy_y = [e.y for e in smld_noisy.emitters]
smld_drift_x = [e.x for e in smld_drift.emitters]
smld_drift_y = [e.y for e in smld_drift.emitters]
smld_corrected_x = [e.x for e in smld_corrected.emitters]
smld_corrected_y = [e.y for e in smld_corrected.emitters]
dataset_ids = [e.dataset for e in smld_noisy.emitters]

f = Figure(size=(1200, 400))
ax1 = Axis(f[1, 1], aspect=DataAspect(), title="Original (no drift)")
scatter!(ax1, smld_noisy_x, smld_noisy_y; markersize=2, alpha=0.3, color=dataset_ids, colormap=:tab10)
ax2 = Axis(f[1, 2], aspect=DataAspect(), title="Drifted (200nm inter-dataset offsets)")
scatter!(ax2, smld_drift_x, smld_drift_y; markersize=2, alpha=0.3, color=dataset_ids, colormap=:tab10)
ax3 = Axis(f[1, 3], aspect=DataAspect(), title="Corrected (RMSD=$(round(rmsd, digits=0))nm)")
scatter!(ax3, smld_corrected_x, smld_corrected_y; markersize=2, alpha=0.3, color=dataset_ids, colormap=:tab10)
linkaxes!(ax1, ax2, ax3)

# Save figure
save("examples/drift_inter_dataset_test.png", f)
println("\nFigure saved to examples/drift_inter_dataset_test.png")
display(f)
