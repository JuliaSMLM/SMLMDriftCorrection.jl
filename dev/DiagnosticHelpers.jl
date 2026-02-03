# DiagnosticHelpers.jl - Shared utilities for drift correction diagnostics
#
# This module provides functions for:
# - Test data generation
# - Metrics computation (RMSD, entropy)
# - Trajectory comparison
# - Plotting (CairoMakie + SMLMRender)
# - Output management

module DiagnosticHelpers

using SMLMData
using SMLMSim
using SMLMDriftCorrection
using SMLMRender
using CairoMakie
using DataFrames
using Statistics
using Printf
using Random

const DC = SMLMDriftCorrection

export generate_test_smld, apply_test_drift
export compute_rmsd, compute_per_frame_rmsd, compute_position_residuals
export compute_entropy_metrics
export compare_trajectories, inter_shift_comparison
export plot_trajectory_comparison, plot_render_comparison, plot_overlay_comparison
export plot_residuals, plot_rmsd_vs_frame, plot_residual_scatter
export plot_coefficient_comparison, plot_inter_shift_comparison
export plot_trajectory_cumulative, plot_boundary_analysis
export plot_drift_comparison
export ensure_output_dir, save_figure, save_stats_md, save_stats

# =============================================================================
# Default parameters
# =============================================================================

const DEFAULT_PARAMS = (
    density = 5.0,           # emitters/μm²
    n_frames = 1000,         # per dataset
    seed = 42,
    degree = 2,              # polynomial degree
    drift_scale = 0.1,       # μm (intra-drift amplitude)
    inter_scale = 0.2,       # μm (inter-dataset offsets)
    zoom = 50,               # 100nm/50 = 2nm render pixels
    colormap = :inferno,
    maxn = 200,
)

# =============================================================================
# Data generation
# =============================================================================

"""
    generate_test_smld(; kwargs...)

Generate test SMLD data with Nmer pattern using SMLMSim.

# Blinking Model (GenericFluor)
- `k_on = 0.075 s⁻¹`: molecule turns ON ~3 times per 40 seconds (2000 frames)
- `k_off = 50.0 s⁻¹`: molecule stays ON for ~20ms (1-2 frames at 50fps)

This means molecules rarely re-blink within a single dataset. Entropy-based
drift correction works by aligning REPEATED observations of the same structure:
- For INTER-dataset alignment: multiple datasets image the same Nmer patterns
- For INTRA-dataset drift: need high density or many frames for constraint

# Keyword Arguments
- `density`: emitters per μm² (default: 5.0)
- `n_datasets`: number of datasets (default: 1)
- `n_frames`: frames per dataset (default: 1000)
- `pattern`: SMLMSim pattern (default: Nmer2D(n=6, d=0.2) - hexamer, 200nm diameter)
- `seed`: random seed (default: 42)

# Returns
The noisy SMLD (smld_noisy) from SMLMSim.simulate()
"""
function generate_test_smld(;
        density::Real = DEFAULT_PARAMS.density,
        n_datasets::Int = 1,
        n_frames::Int = DEFAULT_PARAMS.n_frames,
        pattern = Nmer2D(n=6, d=0.2),  # hexamer, 200nm diameter (matches examples)
        seed::Int = DEFAULT_PARAMS.seed)

    params = StaticSMLMParams(
        Float64(density),  # ρ: emitters per μm²
        0.13,              # σ_psf: PSF width in μm (130nm)
        50,                # minphotons: minimum photons for detection
        n_datasets,
        n_frames,
        50.0,              # framerate: 50 fps (20ms/frame)
        2,                 # ndims: 2D
        [0.0, 1.0]         # zrange (not used for 2D)
    )

    # Set seed for reproducibility
    Random.seed!(seed)

    (smld, sim_info) = simulate(
        params;
        pattern = pattern,
        molecule = GenericFluor(; photons=5000.0, k_on=0.075, k_off=50.0),  # ~3 blinks per emitter in 40s
        camera = IdealCamera(1:256, 1:256, 0.1)  # 100nm pixels, 25.6μm FOV
    )

    return smld
end

"""
    apply_test_drift(smld, scenario; kwargs...)

Apply drift to SMLD based on scenario type.

# Arguments
- `smld`: source SMLD data
- `scenario`: :single, :continuous, or :registered

# Keyword Arguments
- `degree`: polynomial degree (default: 2)
- `drift_scale`: intra-drift amplitude in μm (default: 0.1)
- `inter_scale`: inter-dataset offset scale in μm (default: 0.2)
- `seed`: random seed (default: 42)

# Returns
NamedTuple with:
- `smld_drifted`: SMLD with drift applied
- `model_true`: the drift model that was applied
"""
function apply_test_drift(smld, scenario::Symbol;
        degree::Int = DEFAULT_PARAMS.degree,
        drift_scale::Real = DEFAULT_PARAMS.drift_scale,
        inter_scale::Real = DEFAULT_PARAMS.inter_scale,
        seed::Int = DEFAULT_PARAMS.seed)

    Random.seed!(seed)

    if scenario == :single
        # Single dataset: intra-drift only
        model = DC.LegendrePolynomial(smld; degree=degree, initialize="random", rscale=drift_scale)
        # Zero out inter-shifts for single dataset
        model.inter[1].dm .= 0.0

    elseif scenario == :continuous
        # Continuous: drift accumulates across datasets
        # Generate random intra polynomials, then chain inter-shifts for continuity
        model = DC.LegendrePolynomial(smld; degree=degree, initialize="random", rscale=drift_scale)

        # Set inter-shifts to ensure continuity at boundaries
        # total(ds, frame) = intra[ds](frame) + inter[ds]
        # Requirement: total(1, 1) = 0  →  inter[1] = -intra[1](1)
        # Requirement: total(n, 1) = total(n-1, nframes)
        n_frames = smld.n_frames
        ndims = 2
        for dim in 1:ndims
            model.inter[1].dm[dim] = -DC.evaluate_at_frame(model.intra[1].dm[dim], 1)
            for ds = 2:smld.n_datasets
                model.inter[ds].dm[dim] = model.inter[ds-1].dm[dim] +
                    DC.evaluate_at_frame(model.intra[ds-1].dm[dim], n_frames) -
                    DC.evaluate_at_frame(model.intra[ds].dm[dim], 1)
            end
        end

    elseif scenario == :registered
        # Registered: independent inter-offsets + intra-drift
        model = DC.LegendrePolynomial(smld; degree=degree, initialize="random", rscale=drift_scale)
        # Add independent inter-dataset shifts (dataset 1 stays at 0)
        for ds = 2:smld.n_datasets
            model.inter[ds].dm .= inter_scale .* randn(2)
        end
        model.inter[1].dm .= 0.0

    else
        error("Unknown scenario: $scenario. Use :single, :continuous, or :registered")
    end

    smld_drifted = DC.applydrift(smld, model)
    return (smld_drifted=smld_drifted, model_true=model)
end

# =============================================================================
# Metrics
# =============================================================================

"""
    compute_rmsd(smld_orig, smld_corrected) -> Float64

Compute RMSD in nanometers between original and corrected positions.
"""
function compute_rmsd(smld_orig, smld_corrected)
    N = length(smld_orig.emitters)
    @assert N == length(smld_corrected.emitters) "Emitter count mismatch"

    sum_sq = 0.0
    for i in 1:N
        dx = smld_corrected.emitters[i].x - smld_orig.emitters[i].x
        dy = smld_corrected.emitters[i].y - smld_orig.emitters[i].y
        sum_sq += dx^2 + dy^2
    end

    rmsd_um = sqrt(sum_sq / N)
    return rmsd_um * 1000.0  # Convert to nm
end

"""
    compute_per_frame_rmsd(smld_orig, smld_corrected) -> NamedTuple

Compute RMSD per frame.

# Returns
NamedTuple with:
- `frames`: frame numbers
- `rmsd`: RMSD values (nm) per frame
"""
function compute_per_frame_rmsd(smld_orig, smld_corrected)
    n_frames = smld_orig.n_frames
    n_datasets = smld_orig.n_datasets

    # Pre-allocate
    total_frames = n_frames * n_datasets
    frames = Vector{Int}(undef, total_frames)
    rmsd_vals = Vector{Float64}(undef, total_frames)

    for ds in 1:n_datasets
        for f in 1:n_frames
            global_frame = (ds - 1) * n_frames + f
            frames[global_frame] = global_frame

            # Find emitters in this frame/dataset
            sum_sq = 0.0
            count = 0
            for i in eachindex(smld_orig.emitters)
                e_orig = smld_orig.emitters[i]
                if e_orig.frame == f && e_orig.dataset == ds
                    e_corr = smld_corrected.emitters[i]
                    dx = e_corr.x - e_orig.x
                    dy = e_corr.y - e_orig.y
                    sum_sq += dx^2 + dy^2
                    count += 1
                end
            end

            rmsd_vals[global_frame] = count > 0 ? sqrt(sum_sq / count) * 1000.0 : 0.0
        end
    end

    return (frames=frames, rmsd=rmsd_vals)
end

"""
    compute_position_residuals(smld_orig, smld_corrected) -> NamedTuple

Compute position residuals in nm.

# Returns
NamedTuple with:
- `x_nm`: X residuals (nm)
- `y_nm`: Y residuals (nm)
- `total_nm`: Total error magnitude (nm)
"""
function compute_position_residuals(smld_orig, smld_corrected)
    N = length(smld_orig.emitters)

    x_nm = Vector{Float64}(undef, N)
    y_nm = Vector{Float64}(undef, N)
    total_nm = Vector{Float64}(undef, N)

    for i in 1:N
        dx = (smld_corrected.emitters[i].x - smld_orig.emitters[i].x) * 1000.0
        dy = (smld_corrected.emitters[i].y - smld_orig.emitters[i].y) * 1000.0
        x_nm[i] = dx
        y_nm[i] = dy
        total_nm[i] = sqrt(dx^2 + dy^2)
    end

    return (x_nm=x_nm, y_nm=y_nm, total_nm=total_nm)
end

"""
    compute_entropy_metrics(smld; maxn=200) -> NamedTuple

Compute entropy metrics for SMLD data.

# Returns
NamedTuple with:
- `ub_entropy`: Upper bound entropy
- `entropy_hd`: H_D entropy from localization uncertainties
"""
function compute_entropy_metrics(smld; maxn::Int = DEFAULT_PARAMS.maxn)
    x = Float64[e.x for e in smld.emitters]
    y = Float64[e.y for e in smld.emitters]
    σ_x = Float64[e.σ_x for e in smld.emitters]
    σ_y = Float64[e.σ_y for e in smld.emitters]

    ub_ent = DC.ub_entropy(x, y, σ_x, σ_y; maxn=maxn)
    ent_hd = DC.entropy_HD(σ_x, σ_y)

    return (ub_entropy=ub_ent, entropy_hd=ent_hd)
end

# =============================================================================
# Trajectory comparison
# =============================================================================

"""
    compare_trajectories(model_true, model_recovered) -> NamedTuple

Compare true vs recovered drift trajectories.

# Returns
NamedTuple with:
- `frames`: frame numbers
- `true_x`, `true_y`: true drift
- `recovered_x`, `recovered_y`: recovered drift
- `diff_x`, `diff_y`: differences
"""
function compare_trajectories(model_true, model_recovered)
    traj_true = DC.drift_trajectory(model_true)
    traj_rec = DC.drift_trajectory(model_recovered)

    return (
        frames = traj_true.frames,
        true_x = traj_true.x,
        true_y = traj_true.y,
        recovered_x = traj_rec.x,
        recovered_y = traj_rec.y,
        diff_x = traj_rec.x .- traj_true.x,
        diff_y = traj_rec.y .- traj_true.y,
        dataset = traj_true.dataset
    )
end

"""
    inter_shift_comparison(model_true, model_recovered) -> DataFrame

Compare true vs recovered inter-dataset shifts.
"""
function inter_shift_comparison(model_true, model_recovered)
    n_datasets = model_true.ndatasets

    df = DataFrame(
        dataset = 1:n_datasets,
        true_x = [model_true.inter[ds].dm[1] for ds in 1:n_datasets],
        true_y = [model_true.inter[ds].dm[2] for ds in 1:n_datasets],
        recovered_x = [model_recovered.inter[ds].dm[1] for ds in 1:n_datasets],
        recovered_y = [model_recovered.inter[ds].dm[2] for ds in 1:n_datasets],
    )

    df.error_x = df.recovered_x .- df.true_x
    df.error_y = df.recovered_y .- df.true_y
    df.error_total_nm = sqrt.(df.error_x.^2 .+ df.error_y.^2) .* 1000.0

    return df
end

# =============================================================================
# Plotting functions
# =============================================================================

"""
    plot_drift_comparison(traj; title="Drift Comparison")

Plot true vs recovered drift in three panels:
1. X drift vs Frame
2. Y drift vs Frame
3. X drift vs Y drift (2D trajectory)

This is the key diagnostic for validating drift recovery.
"""
function plot_drift_comparison(traj; title::String="Drift Comparison: Ground Truth vs Recovered")
    fig = Figure(size=(1400, 450))

    # Panel 1: X vs Frame
    ax1 = Axis(fig[1, 1],
        xlabel = "Frame",
        ylabel = "X drift (μm)",
        title = "X Drift vs Frame")

    # Plot true first (thin), recovered second (thick) so recovered visible on top
    lines!(ax1, traj.frames, traj.true_x, color=:blue, linewidth=1.5, label="Ground Truth")
    lines!(ax1, traj.frames, traj.recovered_x, color=:red, linewidth=3, linestyle=:dash, label="Recovered")
    axislegend(ax1, position=:lt)

    # Panel 2: Y vs Frame
    ax2 = Axis(fig[1, 2],
        xlabel = "Frame",
        ylabel = "Y drift (μm)",
        title = "Y Drift vs Frame")

    lines!(ax2, traj.frames, traj.true_y, color=:blue, linewidth=1.5, label="Ground Truth")
    lines!(ax2, traj.frames, traj.recovered_y, color=:red, linewidth=3, linestyle=:dash, label="Recovered")
    axislegend(ax2, position=:lt)

    # Panel 3: X vs Y (2D trajectory)
    ax3 = Axis(fig[1, 3],
        xlabel = "X drift (μm)",
        ylabel = "Y drift (μm)",
        title = "Drift Trajectory (X vs Y)",
        aspect = DataAspect())

    lines!(ax3, traj.true_x, traj.true_y, color=:blue, linewidth=1.5, label="Ground Truth")
    lines!(ax3, traj.recovered_x, traj.recovered_y, color=:red, linewidth=3, linestyle=:dash, label="Recovered")

    # Mark start/end points
    scatter!(ax3, [traj.true_x[1]], [traj.true_y[1]], color=:blue, markersize=12, marker=:circle, label="Start (GT)")
    scatter!(ax3, [traj.true_x[end]], [traj.true_y[end]], color=:blue, markersize=12, marker=:star5)
    scatter!(ax3, [traj.recovered_x[1]], [traj.recovered_y[1]], color=:red, markersize=10, marker=:circle)
    scatter!(ax3, [traj.recovered_x[end]], [traj.recovered_y[end]], color=:red, markersize=10, marker=:star5)

    axislegend(ax3, position=:lt)

    Label(fig[0, :], title, fontsize=18)

    return fig
end

"""
    plot_trajectory_comparison(traj; title_suffix="")

Plot true vs recovered X/Y drift trajectories.
"""
function plot_trajectory_comparison(traj; title_suffix::String="")
    fig = Figure(size=(1200, 500))

    ax1 = Axis(fig[1, 1],
        xlabel = "Frame",
        ylabel = "X drift (μm)",
        title = "X Drift" * title_suffix)

    ax2 = Axis(fig[1, 2],
        xlabel = "Frame",
        ylabel = "Y drift (μm)",
        title = "Y Drift" * title_suffix)

    # Color by dataset
    datasets = unique(traj.dataset)
    colors = Makie.wong_colors()

    # Plot true first (thinner), then recovered second (thicker) so recovered is visible on top
    for (i, ds) in enumerate(datasets)
        mask = traj.dataset .== ds
        c = colors[mod1(i, length(colors))]

        # True: thin solid line
        lines!(ax1, traj.frames[mask], traj.true_x[mask], color=c, linestyle=:solid, linewidth=1.5, label= i==1 ? "True" : "")
        lines!(ax2, traj.frames[mask], traj.true_y[mask], color=c, linestyle=:solid, linewidth=1.5)
    end
    for (i, ds) in enumerate(datasets)
        mask = traj.dataset .== ds
        c = colors[mod1(i, length(colors))]

        # Recovered: thick dashed line (plotted second so visible on top)
        lines!(ax1, traj.frames[mask], traj.recovered_x[mask], color=c, linestyle=:dash, linewidth=3, label= i==1 ? "Recovered" : "")
        lines!(ax2, traj.frames[mask], traj.recovered_y[mask], color=c, linestyle=:dash, linewidth=3)
    end

    axislegend(ax1, position=:lt)

    return fig
end

"""
    plot_render_comparison(smld_orig, smld_drifted, smld_corrected; zoom=20, roi=nothing)

Create 3-panel render comparison: Original | Drifted | Corrected.
"""
function plot_render_comparison(smld_orig, smld_drifted, smld_corrected;
        zoom::Int = DEFAULT_PARAMS.zoom,
        roi = nothing)

    # Render all three
    render_kwargs = (colormap=DEFAULT_PARAMS.colormap, zoom=zoom)
    if roi !== nothing
        render_kwargs = (render_kwargs..., roi=roi)
    end

    img_orig, _ = render(smld_orig; render_kwargs...)
    img_drifted, _ = render(smld_drifted; render_kwargs...)
    img_corrected, _ = render(smld_corrected; render_kwargs...)

    # Create figure
    fig = Figure(size=(1800, 600))

    ax1 = Axis(fig[1, 1], aspect=DataAspect(), title="Original")
    ax2 = Axis(fig[1, 2], aspect=DataAspect(), title="Drifted")
    ax3 = Axis(fig[1, 3], aspect=DataAspect(), title="Corrected")

    image!(ax1, rotr90(img_orig))
    image!(ax2, rotr90(img_drifted))
    image!(ax3, rotr90(img_corrected))

    hidedecorations!(ax1)
    hidedecorations!(ax2)
    hidedecorations!(ax3)

    return fig
end

"""
    save_render_suite(smld_drifted, smld_corrected, scenario; roi=nothing, include_dataset_colors=false)

Save render images:
- Histogram at 10x, color by absolute_frame (time across datasets)
- Circles at 50x, color by absolute_frame
- Gaussian at 20x, intensity colormap

If `include_dataset_colors=true` (for multi-dataset scenarios), also saves:
- Histogram and circles colored by dataset (to visualize alignment)
"""
function save_render_suite(smld_drifted, smld_corrected, scenario::Symbol;
                           roi=nothing, include_dataset_colors::Bool=false, prefix::String="")
    dir = ensure_output_dir(scenario; clean=false)

    # Common ROI kwargs
    roi_kwargs = roi !== nothing ? (roi=roi,) : ()

    # 1. Histogram at 10x, color by absolute_frame (time across datasets)
    result = render(smld_drifted; strategy=HistogramRender(), zoom=10,
                    color_by=:absolute_frame, colormap=:turbo, roi_kwargs...)
    fname = "$(prefix)render_drifted_histogram_10x.png"
    save_image(joinpath(dir, fname), result[1])
    println("  Saved: $fname")

    result = render(smld_corrected; strategy=HistogramRender(), zoom=10,
                    color_by=:absolute_frame, colormap=:turbo, roi_kwargs...)
    fname = "$(prefix)render_corrected_histogram_10x.png"
    save_image(joinpath(dir, fname), result[1])
    println("  Saved: $fname")

    # 2. Circles at 50x, color by absolute_frame (time)
    result = render(smld_drifted; strategy=CircleRender(), zoom=50,
                    color_by=:absolute_frame, colormap=:turbo, roi_kwargs...)
    fname = "$(prefix)render_drifted_circles_50x.png"
    save_image(joinpath(dir, fname), result[1])
    println("  Saved: $fname")

    result = render(smld_corrected; strategy=CircleRender(), zoom=50,
                    color_by=:absolute_frame, colormap=:turbo, roi_kwargs...)
    fname = "$(prefix)render_corrected_circles_50x.png"
    save_image(joinpath(dir, fname), result[1])
    println("  Saved: $fname")

    # 3. Gaussian at 20x, intensity colormap
    result = render(smld_drifted; strategy=GaussianRender(), zoom=20,
                    colormap=:inferno, roi_kwargs...)
    fname = "$(prefix)render_drifted_gaussian_20x.png"
    save_image(joinpath(dir, fname), result[1])
    println("  Saved: $fname")

    result = render(smld_corrected; strategy=GaussianRender(), zoom=20,
                    colormap=:inferno, roi_kwargs...)
    fname = "$(prefix)render_corrected_gaussian_20x.png"
    save_image(joinpath(dir, fname), result[1])
    println("  Saved: $fname")

    # 4. Dataset-colored renders (for multi-dataset scenarios)
    if include_dataset_colors
        result = render(smld_drifted; strategy=HistogramRender(), zoom=10,
                        color_by=:dataset, colormap=:tab10, roi_kwargs...)
        fname = "$(prefix)render_drifted_histogram_dataset.png"
        save_image(joinpath(dir, fname), result[1])
        println("  Saved: $fname")

        result = render(smld_corrected; strategy=HistogramRender(), zoom=10,
                        color_by=:dataset, colormap=:tab10, roi_kwargs...)
        fname = "$(prefix)render_corrected_histogram_dataset.png"
        save_image(joinpath(dir, fname), result[1])
        println("  Saved: $fname")

        result = render(smld_drifted; strategy=CircleRender(), zoom=50,
                        color_by=:dataset, colormap=:tab10, roi_kwargs...)
        fname = "$(prefix)render_drifted_circles_dataset.png"
        save_image(joinpath(dir, fname), result[1])
        println("  Saved: $fname")

        result = render(smld_corrected; strategy=CircleRender(), zoom=50,
                        color_by=:dataset, colormap=:tab10, roi_kwargs...)
        fname = "$(prefix)render_corrected_circles_dataset.png"
        save_image(joinpath(dir, fname), result[1])
        println("  Saved: $fname")
    end
end

export save_render_suite

"""
    analyze_intra_drift_per_dataset(model_true, model_recovered; n_test_frames=5)

Compare intra-drift recovery for each dataset, isolating from inter-shift errors.
Returns DataFrame with per-dataset intra-drift error statistics.
"""
function analyze_intra_drift_per_dataset(model_true, model_recovered; n_test_frames::Int=5)
    n_datasets = model_true.ndatasets
    n_frames = model_true.n_frames

    # Test frames evenly spaced
    test_frames = round.(Int, range(1, n_frames, length=n_test_frames))

    results = DataFrame(
        dataset = Int[],
        max_intra_error_nm = Float64[],
        mean_intra_error_nm = Float64[]
    )

    for ds = 1:n_datasets
        diffs = Float64[]
        for f in test_frames
            true_drift = DC.drift_at_frame(model_true, ds, f)
            rec_drift = DC.drift_at_frame(model_recovered, ds, f)

            # Remove inter-shift to isolate intra-drift
            true_intra_x = true_drift[1] - model_true.inter[ds].dm[1]
            true_intra_y = true_drift[2] - model_true.inter[ds].dm[2]
            rec_intra_x = rec_drift[1] - model_recovered.inter[ds].dm[1]
            rec_intra_y = rec_drift[2] - model_recovered.inter[ds].dm[2]

            diff = sqrt((true_intra_x - rec_intra_x)^2 + (true_intra_y - rec_intra_y)^2) * 1000
            push!(diffs, diff)
        end

        push!(results, (
            dataset = ds,
            max_intra_error_nm = maximum(diffs),
            mean_intra_error_nm = mean(diffs)
        ))
    end

    return results
end

export analyze_intra_drift_per_dataset

"""
    plot_overlay_comparison(smld_drifted, smld_corrected; zoom=20)

Create red/green overlay comparing drifted vs corrected.
"""
function plot_overlay_comparison(smld_drifted, smld_corrected;
        zoom::Int = DEFAULT_PARAMS.zoom,
        roi = nothing)

    render_kwargs = (zoom=zoom,)
    if roi !== nothing
        render_kwargs = (render_kwargs..., roi=roi)
    end

    # Use render_overlay with red=drifted, green=corrected
    img, _ = render([smld_drifted, smld_corrected],
                    colors=[:red, :green];
                    render_kwargs...)

    fig = Figure(size=(800, 800))
    ax = Axis(fig[1, 1], aspect=DataAspect(),
              title="Overlay: Red=Drifted, Green=Corrected")
    image!(ax, rotr90(img))
    hidedecorations!(ax)

    return fig
end

"""
    plot_residuals(residuals; title="Position Error Distribution")

Plot histogram of position residuals.
"""
function plot_residuals(residuals; title::String="Position Error Distribution")
    fig = Figure(size=(1000, 400))

    ax1 = Axis(fig[1, 1],
        xlabel = "X residual (nm)",
        ylabel = "Count",
        title = "X Residuals")

    ax2 = Axis(fig[1, 2],
        xlabel = "Y residual (nm)",
        ylabel = "Count",
        title = "Y Residuals")

    ax3 = Axis(fig[1, 3],
        xlabel = "Total error (nm)",
        ylabel = "Count",
        title = "Total Error")

    hist!(ax1, residuals.x_nm, bins=50, color=:steelblue)
    hist!(ax2, residuals.y_nm, bins=50, color=:steelblue)
    hist!(ax3, residuals.total_nm, bins=50, color=:orange)

    # Add mean lines
    vlines!(ax1, [mean(residuals.x_nm)], color=:red, linewidth=2)
    vlines!(ax2, [mean(residuals.y_nm)], color=:red, linewidth=2)
    vlines!(ax3, [mean(residuals.total_nm)], color=:red, linewidth=2)

    Label(fig[0, :], title, fontsize=20)

    return fig
end

"""
    plot_residual_scatter(residuals; title="Residual Scatter")

Plot 2D scatter of X vs Y residuals.
"""
function plot_residual_scatter(residuals; title::String="Residual Scatter")
    fig = Figure(size=(600, 600))

    ax = Axis(fig[1, 1],
        xlabel = "X residual (nm)",
        ylabel = "Y residual (nm)",
        title = title,
        aspect = DataAspect())

    scatter!(ax, residuals.x_nm, residuals.y_nm, markersize=2, alpha=0.3, color=:steelblue)

    # Add crosshairs at origin
    hlines!(ax, [0], color=:gray, linewidth=1)
    vlines!(ax, [0], color=:gray, linewidth=1)

    # Add circle at RMSD
    rmsd = sqrt(mean(residuals.x_nm.^2 .+ residuals.y_nm.^2))
    θ = range(0, 2π, length=100)
    lines!(ax, rmsd .* cos.(θ), rmsd .* sin.(θ), color=:red, linewidth=2, label="RMSD = $(round(rmsd, digits=1)) nm")
    axislegend(ax, position=:rt)

    return fig
end

"""
    plot_rmsd_vs_frame(per_frame; title="RMSD vs Frame")

Plot RMSD over time.
"""
function plot_rmsd_vs_frame(per_frame; title::String="RMSD vs Frame")
    fig = Figure(size=(1000, 400))

    ax = Axis(fig[1, 1],
        xlabel = "Frame",
        ylabel = "RMSD (nm)",
        title = title)

    lines!(ax, per_frame.frames, per_frame.rmsd, color=:steelblue, linewidth=1)

    # Add mean line
    mean_rmsd = mean(per_frame.rmsd)
    hlines!(ax, [mean_rmsd], color=:red, linewidth=2, linestyle=:dash,
            label="Mean = $(round(mean_rmsd, digits=1)) nm")
    axislegend(ax, position=:rt)

    return fig
end

"""
    plot_coefficient_comparison(model_true, model_recovered; title="Legendre Coefficients")

Plot true vs recovered Legendre coefficients (single dataset).
"""
function plot_coefficient_comparison(model_true, model_recovered; title::String="Legendre Coefficients")
    degree = model_true.intra[1].dm[1].degree

    # Extract coefficients for dataset 1
    true_x = model_true.intra[1].dm[1].coefficients
    true_y = model_true.intra[1].dm[2].coefficients
    rec_x = model_recovered.intra[1].dm[1].coefficients
    rec_y = model_recovered.intra[1].dm[2].coefficients

    fig = Figure(size=(800, 400))

    ax1 = Axis(fig[1, 1],
        xlabel = "Coefficient index",
        ylabel = "Value (μm)",
        title = "X coefficients",
        xticks = 1:degree)

    ax2 = Axis(fig[1, 2],
        xlabel = "Coefficient index",
        ylabel = "Value (μm)",
        title = "Y coefficients",
        xticks = 1:degree)

    barplot!(ax1, 1:degree, true_x, color=:steelblue, label="True")
    barplot!(ax1, (1:degree) .+ 0.3, rec_x, color=:orange, label="Recovered")

    barplot!(ax2, 1:degree, true_y, color=:steelblue)
    barplot!(ax2, (1:degree) .+ 0.3, rec_y, color=:orange)

    axislegend(ax1, position=:rt)

    Label(fig[0, :], title, fontsize=20)

    return fig
end

"""
    plot_inter_shift_comparison(df; title="Inter-dataset Shifts")

Plot bar chart of true vs recovered inter-dataset shifts.
"""
function plot_inter_shift_comparison(df; title::String="Inter-dataset Shifts")
    n = nrow(df)

    fig = Figure(size=(1000, 500))

    ax1 = Axis(fig[1, 1],
        xlabel = "Dataset",
        ylabel = "X shift (μm)",
        title = "X Shifts",
        xticks = 1:n)

    ax2 = Axis(fig[1, 2],
        xlabel = "Dataset",
        ylabel = "Y shift (μm)",
        title = "Y Shifts",
        xticks = 1:n)

    # Group bar plot
    barplot!(ax1, df.dataset .- 0.15, df.true_x, color=:steelblue, width=0.3, label="True")
    barplot!(ax1, df.dataset .+ 0.15, df.recovered_x, color=:orange, width=0.3, label="Recovered")

    barplot!(ax2, df.dataset .- 0.15, df.true_y, color=:steelblue, width=0.3)
    barplot!(ax2, df.dataset .+ 0.15, df.recovered_y, color=:orange, width=0.3)

    axislegend(ax1, position=:lt)

    Label(fig[0, :], title, fontsize=20)

    return fig
end

"""
    plot_trajectory_cumulative(model; title="Cumulative Drift Trajectory")

Plot cumulative drift trajectory for continuous mode.
"""
function plot_trajectory_cumulative(model; title::String="Cumulative Drift Trajectory")
    traj = DC.drift_trajectory(model; cumulative=true)

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
    datasets = unique(traj.dataset)
    colors = Makie.wong_colors()

    for (i, ds) in enumerate(datasets)
        mask = traj.dataset .== ds
        c = colors[mod1(i, length(colors))]
        lines!(ax1, traj.frames[mask], traj.x[mask], color=c, linewidth=2, label="DS $ds")
        lines!(ax2, traj.frames[mask], traj.y[mask], color=c, linewidth=2)
    end

    axislegend(ax1, position=:lt)

    Label(fig[0, :], title, fontsize=20)

    return fig
end

"""
    plot_boundary_analysis(model_true, model_recovered; title="Boundary Continuity")

Analyze drift continuity at dataset boundaries for continuous mode.
"""
function plot_boundary_analysis(model_true, model_recovered; title::String="Boundary Continuity")
    n_frames = model_true.n_frames
    n_datasets = model_true.ndatasets

    # Compute boundary jumps
    true_jumps_x = Float64[]
    true_jumps_y = Float64[]
    rec_jumps_x = Float64[]
    rec_jumps_y = Float64[]
    boundaries = Int[]

    for ds = 1:(n_datasets-1)
        push!(boundaries, ds)

        # True model
        end_true = DC.drift_at_frame(model_true, ds, n_frames)
        start_true = DC.drift_at_frame(model_true, ds+1, 1)
        push!(true_jumps_x, start_true[1] - end_true[1])
        push!(true_jumps_y, start_true[2] - end_true[2])

        # Recovered model
        end_rec = DC.drift_at_frame(model_recovered, ds, n_frames)
        start_rec = DC.drift_at_frame(model_recovered, ds+1, 1)
        push!(rec_jumps_x, start_rec[1] - end_rec[1])
        push!(rec_jumps_y, start_rec[2] - end_rec[2])
    end

    fig = Figure(size=(800, 400))

    ax = Axis(fig[1, 1],
        xlabel = "Boundary (DS n → n+1)",
        ylabel = "Jump (μm)",
        title = title,
        xticks = boundaries)

    barplot!(ax, boundaries .- 0.2, true_jumps_x, color=:steelblue, width=0.15, label="True X")
    barplot!(ax, boundaries .- 0.05, true_jumps_y, color=:lightblue, width=0.15, label="True Y")
    barplot!(ax, boundaries .+ 0.1, rec_jumps_x, color=:orange, width=0.15, label="Rec X")
    barplot!(ax, boundaries .+ 0.25, rec_jumps_y, color=:lightsalmon, width=0.15, label="Rec Y")

    axislegend(ax, position=:rt)

    return fig
end

# =============================================================================
# Output management
# =============================================================================

"""
    ensure_output_dir(scenario::Symbol; clean=true) -> String

Ensure output directory exists and return its path.
If `clean=true` (default), removes existing files in the directory.
"""
function ensure_output_dir(scenario::Symbol; clean::Bool=true)
    dir = joinpath(@__DIR__, "output", string(scenario))
    if clean && isdir(dir)
        # Remove all files in directory
        for f in readdir(dir)
            rm(joinpath(dir, f), force=true)
        end
    end
    mkpath(dir)
    return dir
end

"""
    save_figure(fig, scenario, name)

Save figure to output directory.
"""
function save_figure(fig, scenario::Symbol, name::String)
    dir = ensure_output_dir(scenario; clean=false)  # Don't clean when saving
    path = joinpath(dir, name)
    save(path, fig)
    println("  Saved: $path")
    return path
end

"""
    save_stats_md(stats::Dict, scenario::Symbol; notes::String="")

Save statistics to markdown file with nice formatting.
"""
function save_stats_md(stats::Dict, scenario::Symbol; notes::String="", filename::String="stats.md")
    dir = ensure_output_dir(scenario; clean=false)  # Don't clean when saving
    path = joinpath(dir, filename)

    open(path, "w") do io
        println(io, "# Drift Correction Diagnostics: $(uppercase(string(scenario)))")
        println(io)

        # Key results section
        println(io, "## Key Results")
        println(io)
        if haskey(stats, "rmsd_nm")
            println(io, "| Metric | Value |")
            println(io, "|--------|-------|")
            @printf(io, "| **RMSD** | %.2f nm |\n", stats["rmsd_nm"])
            if haskey(stats, "mean_error_nm")
                @printf(io, "| Mean Error | %.2f nm |\n", stats["mean_error_nm"])
            end
            if haskey(stats, "max_error_nm")
                @printf(io, "| Max Error | %.2f nm |\n", stats["max_error_nm"])
            end
            println(io)
        end

        # Expected performance note - context-dependent
        if haskey(stats, "locs_per_frame") && stats["locs_per_frame"] < 1.0
            println(io, "> **Note**: Sparse data ($(round(stats["locs_per_frame"], digits=1)) locs/frame).")
            println(io, "> For sparse single-dataset data, RMSD < 100 nm is acceptable.")
            println(io)
        else
            println(io, "> **Expected Performance**: With sufficient data density, RMSD should be < 50 nm.")
            println(io)
        end

        # Entropy section - handle both old and new field names
        if haskey(stats, "entropy_original") || haskey(stats, "entropy_before")
            println(io, "## Entropy Metrics")
            println(io)
            println(io, "| Metric | Value |")
            println(io, "|--------|-------|")

            # New detailed entropy fields
            if haskey(stats, "entropy_original")
                @printf(io, "| Entropy Original | %.4f |\n", stats["entropy_original"])
            end
            if haskey(stats, "entropy_drifted")
                @printf(io, "| Entropy Drifted | %.4f |\n", stats["entropy_drifted"])
            elseif haskey(stats, "entropy_before")
                @printf(io, "| Entropy Before | %.4f |\n", stats["entropy_before"])
            end
            if haskey(stats, "entropy_corrected")
                @printf(io, "| Entropy Corrected | %.4f |\n", stats["entropy_corrected"])
            elseif haskey(stats, "entropy_after")
                @printf(io, "| Entropy After | %.4f |\n", stats["entropy_after"])
            end
            if haskey(stats, "entropy_delta_vs_original")
                @printf(io, "| **Delta (corr - orig)** | %.4f |\n", stats["entropy_delta_vs_original"])
            end
            if haskey(stats, "entropy_reduction_pct")
                @printf(io, "| Reduction %% | %.1f%% |\n", stats["entropy_reduction_pct"])
            end
            println(io)
        end

        # Sanity check section
        if haskey(stats, "sanity_check_rmsd_nm")
            println(io, "## Sanity Check")
            println(io)
            @printf(io, "Apply + correct with same model: **%.6f nm** (should be ~0)\n", stats["sanity_check_rmsd_nm"])
            println(io)
        end

        # Per-dataset table
        if haskey(stats, "per_dataset_rmsd_nm")
            println(io, "## Per-Dataset Results")
            println(io)
            rmsd_arr = stats["per_dataset_rmsd_nm"]
            n_ds = length(rmsd_arr)

            # Build header based on available columns
            header = "| Dataset | RMSD (nm) |"
            divider = "|---------|-----------|"
            has_inter = haskey(stats, "per_dataset_inter_error_nm")
            has_boundary = haskey(stats, "per_dataset_boundary_error_nm")

            if has_inter
                header *= " Inter Error (nm) |"
                divider *= "------------------|"
            end
            if has_boundary
                header *= " Boundary Error (nm) |"
                divider *= "---------------------|"
            end

            println(io, header)
            println(io, divider)

            for i in 1:n_ds
                row = @sprintf("| %d | %.2f |", i, rmsd_arr[i])
                if has_inter
                    row *= @sprintf(" %.2f |", stats["per_dataset_inter_error_nm"][i])
                end
                if has_boundary
                    row *= @sprintf(" %.2f |", stats["per_dataset_boundary_error_nm"][i])
                end
                println(io, row)
            end
            println(io)
        end

        # Simulation parameters
        println(io, "## Simulation Parameters")
        println(io)
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        for key in ["n_emitters", "n_frames", "n_datasets", "degree", "drift_scale_um", "inter_scale_um", "seed"]
            if haskey(stats, key)
                v = stats[key]
                if v isa AbstractFloat
                    @printf(io, "| %s | %.4f |\n", key, v)
                else
                    @printf(io, "| %s | %s |\n", key, string(v))
                end
            end
        end
        println(io)

        # Notes section if provided
        if !isempty(notes)
            println(io, "## Notes")
            println(io)
            println(io, notes)
            println(io)
        end

        # All stats for reference (scalar values only, arrays shown in tables above)
        println(io, "## All Statistics")
        println(io)
        println(io, "```")
        for (key, value) in sort(collect(stats), by=x->string(x[1]))
            # Skip array values (already shown in per-dataset table)
            if value isa AbstractVector
                continue
            elseif value isa AbstractFloat
                @printf(io, "%-30s: %.6f\n", key, value)
            elseif value isa Integer
                @printf(io, "%-30s: %d\n", key, value)
            else
                @printf(io, "%-30s: %s\n", key, string(value))
            end
        end
        println(io, "```")
    end

    println("  Saved: $path")
    return path
end

# Keep old function for backwards compatibility
function save_stats(stats::Dict, scenario::Symbol)
    save_stats_md(stats, scenario)
end

"""
    save_dataframe(df::DataFrame, scenario::Symbol, name::String)

Save DataFrame to CSV-like text file.
"""
function save_dataframe(df::DataFrame, scenario::Symbol, name::String)
    dir = ensure_output_dir(scenario; clean=false)
    path = joinpath(dir, name)

    open(path, "w") do io
        # Header
        println(io, join(names(df), "\t"))
        # Data
        for row in eachrow(df)
            vals = [v isa AbstractFloat ? @sprintf("%.4f", v) : string(v) for v in row]
            println(io, join(vals, "\t"))
        end
    end

    println("  Saved: $path")
    return path
end

export save_dataframe

end # module
