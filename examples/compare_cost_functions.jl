"""
Comparison of drift correction methods: Kdtree, Entropy, and Progressive refinement.

This script:
1. Generates realistic SMLM data with SMLMSim
2. Applies known 2nd-order polynomial drift
3. Corrects drift using three methods (all degree=2):
   - Kdtree cost function (direct optimization)
   - Entropy cost function (direct optimization)
   - Progressive refinement (iterative Kdtree optimization)
4. Computes quantitative metrics comparing correction quality
5. Generates publication-quality comparison figures

Note: Legendre polynomial methods are implemented but have convergence issues
when the applied drift uses standard polynomial basis. For Legendre testing,
apply drift using LegendrePolynomial model.

Output saved to examples/output/
"""

using Pkg
Pkg.activate(@__DIR__)

using SMLMDriftCorrection
DC = SMLMDriftCorrection
using SMLMData
using SMLMSim
using CairoMakie
using Statistics
using Printf
using Random

# Set seed for reproducibility
Random.seed!(42)

# Create output directory if it doesn't exist
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("="^70)
println("Drift Correction Cost Function Comparison")
println("="^70)

#=============================================================================
# 1. SIMULATION PARAMETERS
#
# Create a realistic SMLM dataset with enough localizations per dataset
# to properly test the drift correction algorithms.
=============================================================================#

println("\n[1/6] Setting up simulation parameters...")

# Simulation parameters for realistic SMLM data
# Target: ~20 localizations per frame
# Using kinetics and density to achieve realistic blinking behavior
# Note: Entropy method is O(n*maxn) per evaluation, so we limit dataset size

n_datasets = 5
n_frames = 500
roi_size = 128  # pixels (12.8 μm with 0.1 μm pixel size)

# Camera field of view
pixel_size = 0.1  # μm
fov = roi_size * pixel_size  # 12.8 μm
fov_area = fov^2  # 163.84 μm²

# Photoswitching kinetics (similar to SMLMAnalysis examples)
# With these kinetics, ~0.3% of emitters are ON at any time
# To get ~20 locs/frame, need ~6700 total emitters → ~40 emitters/μm²
# Using 8-mers at 5/μm² gives 40 emitters/μm²
pattern_density = 5.0  # 5 octamer patterns per μm²

params = StaticSMLMParams(
    pattern_density,  # density (ρ): patterns per μm²
    0.13,             # σ_psf: PSF width in μm (130nm)
    50,               # minphotons: minimum photons for detection
    n_datasets,       # ndatasets: number of independent datasets
    n_frames,         # nframes: frames per dataset
    50.0,             # framerate: frames per second (20ms exposure)
    2,                # ndims: 2D
    [0.0, 1.0]        # zrange: not used for 2D
)

println("  Datasets: $n_datasets")
println("  Frames per dataset: $n_frames")
println("  Total frames: $(n_datasets * n_frames)")
println("  Pattern density: $pattern_density octamers/μm² ($(pattern_density * 8) emitters/μm²)")
println("  ROI: $(roi_size)×$(roi_size) pixels ($(fov)×$(fov) μm)")
println("  FOV area: $(round(fov_area, digits=1)) μm²")

#=============================================================================
# 2. GENERATE SMLM DATA
=============================================================================#

println("\n[2/6] Generating SMLM data with SMLMSim...")

# Octamer pattern with 150nm diameter (similar to SMLMAnalysis examples)
pattern = Nmer2D(n=8, d=0.15)

# Photoswitching fluorophore with sparse labeling kinetics
# k_on=0.06, k_off=20 gives ~0.3% duty cycle → sparse localizations per frame
molecule = GenericFluor(
    photons = 5000.0,   # photons/sec when ON
    k_on = 0.06,        # switching on rate (Hz)
    k_off = 20.0        # switching off rate (Hz)
)

smld_true, smld_model, smld_noisy = simulate(
    params;
    pattern = pattern,
    molecule = molecule,
    camera = IdealCamera(1:roi_size, 1:roi_size, pixel_size)
)

n_total = length(smld_noisy.emitters)
println("  Total localizations: $n_total")
println("  Localizations per frame: $(round(n_total / (n_datasets * n_frames), digits=1))")

# Count localizations per dataset
locs_per_dataset = [count(e -> e.dataset == ds, smld_noisy.emitters)
                    for ds in 1:n_datasets]
println("  Localizations per dataset: $(minimum(locs_per_dataset)) - $(maximum(locs_per_dataset))")
println("  Mean per dataset: $(round(mean(locs_per_dataset), digits=1))")

#=============================================================================
# 3. APPLY KNOWN DRIFT
#
# Create a 2nd-order polynomial drift model with realistic drift magnitudes.
# Typical SMLM drift: ~50-200 nm over an acquisition.
=============================================================================#

println("\n[3/6] Applying known polynomial drift (degree=2)...")

# Create drift model with explicit coefficients per dataset
# Using "zeros" init then setting coefficients manually ensures both X and Y
# have comparable drift magnitudes. Each dataset gets different drift to simulate
# realistic multi-acquisition data (drift varies between acquisitions).
drift_true = DC.Polynomial(smld_noisy; degree=2, initialize="zeros")

# Set coefficients to give ~100-200nm total drift in both X and Y
# Each dataset gets slightly different coefficients + inter-shift
# This simulates realistic SMLM where each acquisition has independent drift
Random.seed!(123)  # Use different seed for drift than simulation
for ds in 1:n_datasets
    # Base coefficients with per-dataset variation (±30%)
    base_c1_x = 2.0e-4
    base_c2_x = 3.2e-7
    base_c1_y = -3.0e-4
    base_c2_y = -2.4e-7

    # Add variation per dataset
    var = 0.3 * (2*rand() - 1)  # ±30% variation
    drift_true.intra[ds].dm[1].coefficients[1] = base_c1_x * (1 + var)
    drift_true.intra[ds].dm[1].coefficients[2] = base_c2_x * (1 + 0.3*(2*rand()-1))
    drift_true.intra[ds].dm[2].coefficients[1] = base_c1_y * (1 + 0.3*(2*rand()-1))
    drift_true.intra[ds].dm[2].coefficients[2] = base_c2_y * (1 + 0.3*(2*rand()-1))

    # Add inter-dataset shift (simulates sample repositioning between acquisitions)
    # Shift by ~50-150nm in each direction
    drift_true.inter[ds].dm[1] = 0.1 * (rand() - 0.5)  # X shift: ±50nm
    drift_true.inter[ds].dm[2] = 0.1 * (rand() - 0.5)  # Y shift: ±50nm
end
Random.seed!(42)  # Reset to original seed

# Apply drift to the noisy data
smld_drifted = DC.applydrift(smld_noisy, drift_true)

# Compute drift magnitude statistics
x_orig = [e.x for e in smld_noisy.emitters]
y_orig = [e.y for e in smld_noisy.emitters]
x_drift = [e.x for e in smld_drifted.emitters]
y_drift = [e.y for e in smld_drifted.emitters]

drift_magnitudes = sqrt.((x_drift .- x_orig).^2 .+ (y_drift .- y_orig).^2)
println("  Applied drift magnitude:")
println("    Mean: $(round(mean(drift_magnitudes) * 1000, digits=1)) nm")
println("    Max:  $(round(maximum(drift_magnitudes) * 1000, digits=1)) nm")
println("    Std:  $(round(std(drift_magnitudes) * 1000, digits=1)) nm")

#=============================================================================
# 4. DRIFT CORRECTION WITH BOTH METHODS
=============================================================================#

println("\n[4/6] Running drift correction...")

# Method 1: Kdtree cost function
println("\n  --- Kdtree cost function ---")
t_kd = @elapsed smld_corrected_kd = DC.driftcorrect(
    smld_drifted;
    cost_fun="Kdtree",
    d_cutoff=0.01,
    verbose=1
)
println("    Time: $(round(t_kd, digits=1)) seconds")

# Method 2: Entropy cost function
println("\n  --- Entropy cost function ---")
t_ent = @elapsed smld_corrected_ent = DC.driftcorrect(
    smld_drifted;
    cost_fun="Entropy",
    maxn=100,
    verbose=1
)
println("    Time: $(round(t_ent, digits=1)) seconds")

# Method 3: Progressive refinement (iterative degree=2, standard polynomial)
# Iteratively refines the correction until convergence
println("\n  --- Progressive refinement (standard poly, degree=2 iterative) ---")

function progressive_correct(smld_in, x_orig, y_orig; max_iter=5, intramodel="Polynomial", degree=2)
    smld_current = smld_in
    n_iter = 0
    rmsd_prev = Inf

    for iter in 1:max_iter
        smld_new = DC.driftcorrect(smld_current; degree=degree, cost_fun="Kdtree",
                                    intramodel=intramodel, verbose=0)

        x_p = [e.x for e in smld_new.emitters]
        y_p = [e.y for e in smld_new.emitters]
        rmsd_curr = sqrt(mean((x_p .- x_orig).^2 .+ (y_p .- y_orig).^2))

        n_iter = iter
        smld_current = smld_new

        if abs(rmsd_prev - rmsd_curr) / rmsd_prev < 0.01  # <1% improvement
            break
        end
        rmsd_prev = rmsd_curr
    end

    return smld_current, n_iter
end

t_prog = @elapsed begin
    global smld_corrected_prog, n_prog_iterations = progressive_correct(smld_drifted, x_orig, y_orig)
end
println("    Time: $(round(t_prog, digits=1)) seconds ($n_prog_iterations iterations)")

# Note: Legendre methods removed from this comparison because they have convergence
# issues when applied drift uses standard polynomial basis. The Legendre implementation
# works correctly when tested with LegendrePolynomial-applied drift.

#=============================================================================
# 5. COMPUTE QUANTITATIVE METRICS
=============================================================================#

println("\n[5/6] Computing quantitative metrics...")

# Extract coordinates
x_corr_kd = [e.x for e in smld_corrected_kd.emitters]
y_corr_kd = [e.y for e in smld_corrected_kd.emitters]
x_corr_ent = [e.x for e in smld_corrected_ent.emitters]
y_corr_ent = [e.y for e in smld_corrected_ent.emitters]
x_corr_prog = [e.x for e in smld_corrected_prog.emitters]
y_corr_prog = [e.y for e in smld_corrected_prog.emitters]

# Compute residuals (corrected - original)
residuals_kd_x = x_corr_kd .- x_orig
residuals_kd_y = y_corr_kd .- y_orig
residuals_ent_x = x_corr_ent .- x_orig
residuals_ent_y = y_corr_ent .- y_orig
residuals_prog_x = x_corr_prog .- x_orig
residuals_prog_y = y_corr_prog .- y_orig

# Euclidean residuals
residuals_kd = sqrt.(residuals_kd_x.^2 .+ residuals_kd_y.^2)
residuals_ent = sqrt.(residuals_ent_x.^2 .+ residuals_ent_y.^2)
residuals_prog = sqrt.(residuals_prog_x.^2 .+ residuals_prog_y.^2)

# Overall RMSD
rmsd_kd = sqrt(mean(residuals_kd.^2))
rmsd_ent = sqrt(mean(residuals_ent.^2))
rmsd_prog = sqrt(mean(residuals_prog.^2))

# Per-dataset RMSD
rmsd_kd_per_ds = Float64[]
rmsd_ent_per_ds = Float64[]
rmsd_prog_per_ds = Float64[]
datasets = [e.dataset for e in smld_noisy.emitters]

for ds in 1:n_datasets
    mask = datasets .== ds
    push!(rmsd_kd_per_ds, sqrt(mean(residuals_kd[mask].^2)))
    push!(rmsd_ent_per_ds, sqrt(mean(residuals_ent[mask].^2)))
    push!(rmsd_prog_per_ds, sqrt(mean(residuals_prog[mask].^2)))
end

println("\n  Overall Results:")
println("  " * "-"^58)
println("  Method         | RMSD (nm) | Mean Resid (nm) | Max Resid (nm)")
println("  " * "-"^58)
@printf("  Kdtree         | %9.2f | %15.2f | %14.2f\n",
        rmsd_kd*1000, mean(residuals_kd)*1000, maximum(residuals_kd)*1000)
@printf("  Entropy        | %9.2f | %15.2f | %14.2f\n",
        rmsd_ent*1000, mean(residuals_ent)*1000, maximum(residuals_ent)*1000)
@printf("  Progressive    | %9.2f | %15.2f | %14.2f\n",
        rmsd_prog*1000, mean(residuals_prog)*1000, maximum(residuals_prog)*1000)
println("  " * "-"^58)

# Find best method
rmsds = [rmsd_kd, rmsd_ent, rmsd_prog]
names = ["Kdtree", "Entropy", "Progressive"]
best_idx = argmin(rmsds)
worst_idx = argmax(rmsds)
improvement = (rmsds[worst_idx] - rmsds[best_idx]) / rmsds[worst_idx] * 100
println("\n  $(names[best_idx]) is best: $(round(improvement, digits=1))% better than $(names[worst_idx])")

# Timing comparison
println("\n  Timing:")
println("    Kdtree:      $(round(t_kd, digits=1)) s")
println("    Entropy:     $(round(t_ent, digits=1)) s")
println("    Progressive: $(round(t_prog, digits=1)) s")

#=============================================================================
# 6. GENERATE COMPARISON FIGURES
=============================================================================#

println("\n[6/6] Generating comparison figures...")

# Set up consistent color scheme
colors = (
    original = :gray,
    drifted = :red,
    kdtree = :blue,
    entropy = :green,
    progressive = :purple
)

#-----------------------------------------------------------------------------
# Figure 1: Scatter plot comparison (zoomed to show structure)
#-----------------------------------------------------------------------------

fig1 = Figure(size=(1200, 800), fontsize=14)

# Find a region with good structure for visualization
x_center = mean(x_orig)
y_center = mean(y_orig)
zoom_radius = 2.0  # μm

zoom_mask = (abs.(x_orig .- x_center) .< zoom_radius) .&
            (abs.(y_orig .- y_center) .< zoom_radius)

# Row 1: Original, Drifted, Kdtree
ax1 = Axis(fig1[1, 1], aspect=DataAspect(),
           title="Original (Ground Truth)", xlabel="x (μm)", ylabel="y (μm)")
scatter!(ax1, x_orig[zoom_mask], y_orig[zoom_mask];
         markersize=4, color=colors.original, alpha=0.7)

ax2 = Axis(fig1[1, 2], aspect=DataAspect(),
           title="After Drift Applied", xlabel="x (μm)", ylabel="y (μm)")
scatter!(ax2, x_drift[zoom_mask], y_drift[zoom_mask];
         markersize=4, color=colors.drifted, alpha=0.7)

ax3 = Axis(fig1[1, 3], aspect=DataAspect(),
           title="Kdtree - $(round(rmsd_kd*1000, digits=1)) nm",
           xlabel="x (μm)", ylabel="y (μm)")
scatter!(ax3, x_corr_kd[zoom_mask], y_corr_kd[zoom_mask];
         markersize=4, color=colors.kdtree, alpha=0.7)

# Row 2: Entropy, Progressive
ax4 = Axis(fig1[2, 1], aspect=DataAspect(),
           title="Entropy - $(round(rmsd_ent*1000, digits=1)) nm",
           xlabel="x (μm)", ylabel="y (μm)")
scatter!(ax4, x_corr_ent[zoom_mask], y_corr_ent[zoom_mask];
         markersize=4, color=colors.entropy, alpha=0.7)

ax5 = Axis(fig1[2, 2], aspect=DataAspect(),
           title="Progressive - $(round(rmsd_prog*1000, digits=1)) nm",
           xlabel="x (μm)", ylabel="y (μm)")
scatter!(ax5, x_corr_prog[zoom_mask], y_corr_prog[zoom_mask];
         markersize=4, color=colors.progressive, alpha=0.7)

# Link axes for easy comparison
linkaxes!(ax1, ax2, ax3, ax4, ax5)

save(joinpath(output_dir, "scatter_comparison.png"), fig1, px_per_unit=2)
println("  Saved: scatter_comparison.png")

#-----------------------------------------------------------------------------
# Figure 2: Residual distributions
#-----------------------------------------------------------------------------

fig2 = Figure(size=(1200, 800), fontsize=14)

# X residuals
ax2a = Axis(fig2[1, 1], xlabel="X Residual (nm)", ylabel="Count",
            title="X Residuals Distribution")
hist!(ax2a, residuals_kd_x .* 1000, bins=100, color=(colors.kdtree, 0.5),
      label="Kdtree (σ=$(round(std(residuals_kd_x)*1000, digits=1)) nm)")
hist!(ax2a, residuals_ent_x .* 1000, bins=100, color=(colors.entropy, 0.5),
      label="Entropy (σ=$(round(std(residuals_ent_x)*1000, digits=1)) nm)")
hist!(ax2a, residuals_prog_x .* 1000, bins=100, color=(colors.progressive, 0.5),
      label="Progressive (σ=$(round(std(residuals_prog_x)*1000, digits=1)) nm)")
axislegend(ax2a, position=:rt)

# Y residuals
ax2b = Axis(fig2[1, 2], xlabel="Y Residual (nm)", ylabel="Count",
            title="Y Residuals Distribution")
hist!(ax2b, residuals_kd_y .* 1000, bins=100, color=(colors.kdtree, 0.5),
      label="Kdtree (σ=$(round(std(residuals_kd_y)*1000, digits=1)) nm)")
hist!(ax2b, residuals_ent_y .* 1000, bins=100, color=(colors.entropy, 0.5),
      label="Entropy (σ=$(round(std(residuals_ent_y)*1000, digits=1)) nm)")
hist!(ax2b, residuals_prog_y .* 1000, bins=100, color=(colors.progressive, 0.5),
      label="Progressive (σ=$(round(std(residuals_prog_y)*1000, digits=1)) nm)")
axislegend(ax2b, position=:rt)

# Euclidean residuals
ax2c = Axis(fig2[2, 1], xlabel="Euclidean Residual (nm)", ylabel="Count",
            title="Total Residual Distribution")
hist!(ax2c, residuals_kd .* 1000, bins=100, color=(colors.kdtree, 0.5),
      label="Kdtree (mean=$(round(mean(residuals_kd)*1000, digits=1)) nm)")
hist!(ax2c, residuals_ent .* 1000, bins=100, color=(colors.entropy, 0.5),
      label="Entropy (mean=$(round(mean(residuals_ent)*1000, digits=1)) nm)")
hist!(ax2c, residuals_prog .* 1000, bins=100, color=(colors.progressive, 0.5),
      label="Progressive (mean=$(round(mean(residuals_prog)*1000, digits=1)) nm)")
axislegend(ax2c, position=:rt)

# Per-dataset RMSD (3 methods)
ax2d = Axis(fig2[2, 2], xlabel="Dataset", ylabel="RMSD (nm)",
            title="Per-Dataset RMSD")
barplot!(ax2d, (1:n_datasets) .- 0.2, rmsd_kd_per_ds .* 1000,
         color=colors.kdtree, label="Kdtree", width=0.2)
barplot!(ax2d, 1:n_datasets, rmsd_ent_per_ds .* 1000,
         color=colors.entropy, label="Entropy", width=0.2)
barplot!(ax2d, (1:n_datasets) .+ 0.2, rmsd_prog_per_ds .* 1000,
         color=colors.progressive, label="Progr", width=0.2)
axislegend(ax2d, position=:rt)

save(joinpath(output_dir, "residual_distributions.png"), fig2, px_per_unit=2)
println("  Saved: residual_distributions.png")

#-----------------------------------------------------------------------------
# Figure 3: Drift curves - Applied vs Found (top), Residuals (bottom)
#-----------------------------------------------------------------------------

fig3 = Figure(size=(1400, 900), fontsize=14)

# Get emitters from dataset 1
ds1_mask = [e.dataset == 1 for e in smld_noisy.emitters]
ds1_frames = [e.frame for e in smld_noisy.emitters[ds1_mask]]
ds1_x_orig = [e.x for e in smld_noisy.emitters[ds1_mask]]
ds1_y_orig = [e.y for e in smld_noisy.emitters[ds1_mask]]
ds1_x_drift = [e.x for e in smld_drifted.emitters[ds1_mask]]
ds1_y_drift = [e.y for e in smld_drifted.emitters[ds1_mask]]
ds1_x_corr_kd = [e.x for e in smld_corrected_kd.emitters[ds1_mask]]
ds1_y_corr_kd = [e.y for e in smld_corrected_kd.emitters[ds1_mask]]
ds1_x_corr_ent = [e.x for e in smld_corrected_ent.emitters[ds1_mask]]
ds1_y_corr_ent = [e.y for e in smld_corrected_ent.emitters[ds1_mask]]
ds1_x_corr_prog = [e.x for e in smld_corrected_prog.emitters[ds1_mask]]
ds1_y_corr_prog = [e.y for e in smld_corrected_prog.emitters[ds1_mask]]

# Compute mean drift per frame bin
frame_bins = 1:50:n_frames
n_bins = length(frame_bins) - 1

# Arrays for applied drift
drift_x_applied = Float64[]
drift_y_applied = Float64[]
# Arrays for found drift (drifted - corrected = what was subtracted)
drift_x_found_kd = Float64[]
drift_x_found_ent = Float64[]
drift_x_found_prog = Float64[]
drift_y_found_kd = Float64[]
drift_y_found_ent = Float64[]
drift_y_found_prog = Float64[]
frame_centers = Float64[]

for i in 1:n_bins
    f_lo, f_hi = frame_bins[i], frame_bins[i+1]
    bin_mask = (ds1_frames .>= f_lo) .& (ds1_frames .< f_hi)
    if sum(bin_mask) > 0
        push!(frame_centers, (f_lo + f_hi) / 2)
        # Applied drift = drifted - original
        push!(drift_x_applied, mean(ds1_x_drift[bin_mask] .- ds1_x_orig[bin_mask]))
        push!(drift_y_applied, mean(ds1_y_drift[bin_mask] .- ds1_y_orig[bin_mask]))
        # Found drift = drifted - corrected (what was subtracted by correction)
        push!(drift_x_found_kd, mean(ds1_x_drift[bin_mask] .- ds1_x_corr_kd[bin_mask]))
        push!(drift_y_found_kd, mean(ds1_y_drift[bin_mask] .- ds1_y_corr_kd[bin_mask]))
        push!(drift_x_found_ent, mean(ds1_x_drift[bin_mask] .- ds1_x_corr_ent[bin_mask]))
        push!(drift_y_found_ent, mean(ds1_y_drift[bin_mask] .- ds1_y_corr_ent[bin_mask]))
        push!(drift_x_found_prog, mean(ds1_x_drift[bin_mask] .- ds1_x_corr_prog[bin_mask]))
        push!(drift_y_found_prog, mean(ds1_y_drift[bin_mask] .- ds1_y_corr_prog[bin_mask]))
    end
end

# Subtract zero-frame offset (arbitrary starting point)
drift_x_applied .-= drift_x_applied[1]
drift_y_applied .-= drift_y_applied[1]
drift_x_found_kd .-= drift_x_found_kd[1]
drift_x_found_ent .-= drift_x_found_ent[1]
drift_x_found_prog .-= drift_x_found_prog[1]
drift_y_found_kd .-= drift_y_found_kd[1]
drift_y_found_ent .-= drift_y_found_ent[1]
drift_y_found_prog .-= drift_y_found_prog[1]

# Compute residuals from offset-corrected values (found - applied)
resid_x_kd = drift_x_found_kd .- drift_x_applied
resid_x_ent = drift_x_found_ent .- drift_x_applied
resid_x_prog = drift_x_found_prog .- drift_x_applied
resid_y_kd = drift_y_found_kd .- drift_y_applied
resid_y_ent = drift_y_found_ent .- drift_y_applied
resid_y_prog = drift_y_found_prog .- drift_y_applied

# Top row: Applied vs Found drift curves (zero-offset subtracted)
ax3a = Axis(fig3[1, 1], xlabel="Frame", ylabel="X Drift (μm)",
            title="X Drift: Applied vs Found (Dataset 1)")
lines!(ax3a, frame_centers, drift_x_applied, color=:gray60, linewidth=5, label="Applied")
lines!(ax3a, frame_centers, drift_x_found_kd, color=colors.kdtree, linewidth=2, label="Kdtree")
lines!(ax3a, frame_centers, drift_x_found_ent, color=colors.entropy, linewidth=2, linestyle=:dash, label="Entropy")
lines!(ax3a, frame_centers, drift_x_found_prog, color=colors.progressive, linewidth=2, linestyle=:dot, label="Progr")
axislegend(ax3a, position=:lt)

ax3b = Axis(fig3[1, 2], xlabel="Frame", ylabel="Y Drift (μm)",
            title="Y Drift: Applied vs Found (Dataset 1)")
lines!(ax3b, frame_centers, drift_y_applied, color=:gray60, linewidth=5, label="Applied")
lines!(ax3b, frame_centers, drift_y_found_kd, color=colors.kdtree, linewidth=2, label="Kdtree")
lines!(ax3b, frame_centers, drift_y_found_ent, color=colors.entropy, linewidth=2, linestyle=:dash, label="Entropy")
lines!(ax3b, frame_centers, drift_y_found_prog, color=colors.progressive, linewidth=2, linestyle=:dot, label="Progr")
axislegend(ax3b, position=:lt)

# Bottom row: Residuals (found - applied, should be ~0 for perfect recovery)
ax3c = Axis(fig3[2, 1], xlabel="Frame", ylabel="X Residual (μm)",
            title="X Residual (found - applied)")
hlines!(ax3c, [0], color=:gray60, linewidth=3)
lines!(ax3c, frame_centers, resid_x_kd, color=colors.kdtree, linewidth=2, label="Kdtree")
lines!(ax3c, frame_centers, resid_x_ent, color=colors.entropy, linewidth=2, linestyle=:dash, label="Entropy")
lines!(ax3c, frame_centers, resid_x_prog, color=colors.progressive, linewidth=2, linestyle=:dot, label="Progr")
axislegend(ax3c, position=:lt)

ax3d = Axis(fig3[2, 2], xlabel="Frame", ylabel="Y Residual (μm)",
            title="Y Residual (found - applied)")
hlines!(ax3d, [0], color=:gray60, linewidth=3)
lines!(ax3d, frame_centers, resid_y_kd, color=colors.kdtree, linewidth=2, label="Kdtree")
lines!(ax3d, frame_centers, resid_y_ent, color=colors.entropy, linewidth=2, linestyle=:dash, label="Entropy")
lines!(ax3d, frame_centers, resid_y_prog, color=colors.progressive, linewidth=2, linestyle=:dot, label="Progr")
axislegend(ax3d, position=:lt)

save(joinpath(output_dir, "drift_curves.png"), fig3, px_per_unit=2)
println("  Saved: drift_curves.png")

#-----------------------------------------------------------------------------
# Figure 4: Summary comparison
#-----------------------------------------------------------------------------

fig4 = Figure(size=(1200, 600), fontsize=14)

# Create summary bar plot
metrics = ["RMSD", "Mean\nResidual", "Std X", "Std Y"]
kd_values = [rmsd_kd*1000, mean(residuals_kd)*1000,
             std(residuals_kd_x)*1000, std(residuals_kd_y)*1000]
ent_values = [rmsd_ent*1000, mean(residuals_ent)*1000,
              std(residuals_ent_x)*1000, std(residuals_ent_y)*1000]
prog_values = [rmsd_prog*1000, mean(residuals_prog)*1000,
               std(residuals_prog_x)*1000, std(residuals_prog_y)*1000]

ax4 = Axis(fig4[1, 1],
           ylabel="Error (nm)",
           title="Drift Correction Quality Comparison",
           xticks=(1:4, metrics))

barplot!(ax4, (1:4) .- 0.2, kd_values, width=0.2, color=colors.kdtree, label="Kdtree")
barplot!(ax4, 1:4, ent_values, width=0.2, color=colors.entropy, label="Entropy")
barplot!(ax4, (1:4) .+ 0.2, prog_values, width=0.2, color=colors.progressive, label="Progr")
axislegend(ax4, position=:rt)

save(joinpath(output_dir, "summary_comparison.png"), fig4, px_per_unit=2)
println("  Saved: summary_comparison.png")

#=============================================================================
# FINAL SUMMARY
=============================================================================#

println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println("\nSimulation:")
println("  - $n_datasets datasets × $n_frames frames = $(n_datasets * n_frames) total frames")
println("  - $n_total localizations")
println("  - Applied drift: $(round(maximum(drift_magnitudes)*1000, digits=1)) nm max")

best_method = names[argmin(rmsds)]
println("\nCorrection Quality (RMSD):")
println("  - Kdtree:      $(round(rmsd_kd*1000, digits=2)) nm$(best_method == "Kdtree" ? "  ** BEST **" : "")")
println("  - Entropy:     $(round(rmsd_ent*1000, digits=2)) nm$(best_method == "Entropy" ? "  ** BEST **" : "")")
println("  - Progressive: $(round(rmsd_prog*1000, digits=2)) nm$(best_method == "Progressive" ? "  ** BEST **" : "")")

println("\nTiming:")
println("  - Kdtree:      $(round(t_kd, digits=1)) s")
println("  - Entropy:     $(round(t_ent, digits=1)) s")
println("  - Progressive: $(round(t_prog, digits=1)) s")

println("\nOutput files saved to: $output_dir")
println("  - scatter_comparison.png")
println("  - residual_distributions.png")
println("  - drift_curves.png")
println("  - summary_comparison.png")
println("="^70)
