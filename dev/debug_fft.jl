# debug_fft.jl - Debug FFT cross-correlation for drift correction
#
# Generates debug images to diagnose FFT alignment issues:
# - Histogram images for each dataset
# - Cross-correlation images with peak markers
# - Saves to output/fft_debug/

using Pkg
Pkg.activate(@__DIR__)

include("DiagnosticHelpers.jl")
using .DiagnosticHelpers
using SMLMDriftCorrection
using SMLMData
using CairoMakie
using Printf
using Statistics
using FourierTools

const DC = SMLMDriftCorrection

# Import internal functions for debugging
import SMLMDriftCorrection: histimage2D, crosscorr2D, filter_by_dataset

"""
Debug FFT alignment by saving all intermediate images.
"""
function debug_fft(;
        density::Real = 2.5,
        n_datasets::Int = 5,
        n_frames::Int = 2000,
        degree::Int = 2,
        drift_scale::Real = 0.2,
        inter_scale::Real = 0.3,
        seed::Int = 42,
        histbinsize::Real = 0.05)

    println("=" ^ 60)
    println("FFT DEBUG")
    println("=" ^ 60)

    # Create output directory
    debug_dir = joinpath(@__DIR__, "output", "fft_debug")
    rm(debug_dir; force=true, recursive=true)
    mkpath(debug_dir)
    println("Output: $debug_dir")

    # =========================================================================
    # 1. Generate test data (same as registered mode)
    # =========================================================================
    println("\n[1/4] Generating test data...")

    smld_orig = generate_test_smld(;
        density = density,
        n_datasets = n_datasets,
        n_frames = n_frames,
        seed = seed
    )

    println("  Emitters: $(length(smld_orig.emitters))")
    println("  Datasets: $(smld_orig.n_datasets)")

    # =========================================================================
    # 2. Apply drift (registered mode)
    # =========================================================================
    println("\n[2/4] Applying drift...")

    result_drift = apply_test_drift(smld_orig, :registered;
        degree = degree,
        drift_scale = drift_scale,
        inter_scale = inter_scale,
        seed = seed
    )
    smld_drifted = result_drift.smld_drifted
    model_true = result_drift.model_true

    println("  True inter-shifts:")
    for ds = 1:n_datasets
        x, y = model_true.inter[ds].dm[1], model_true.inter[ds].dm[2]
        @printf("    DS %d: (%.4f, %.4f) μm = (%.1f, %.1f) nm\n", ds, x, y, x*1000, y*1000)
    end

    # =========================================================================
    # 3. Generate and save histogram images
    # =========================================================================
    println("\n[3/4] Generating histogram images...")

    # Get ROI from camera
    smld_ref = filter_by_dataset(smld_drifted, 1)
    ROI = Float64[
        smld_ref.camera.pixel_edges_x[1],
        smld_ref.camera.pixel_edges_x[end],
        smld_ref.camera.pixel_edges_y[1],
        smld_ref.camera.pixel_edges_y[end]
    ]
    println("  ROI: $ROI")
    println("  histbinsize: $histbinsize μm")

    # Generate histogram for each dataset
    histimages = Dict{Int, Matrix{Int}}()
    for ds = 1:n_datasets
        smld_ds = filter_by_dataset(smld_drifted, ds)
        x = [e.x for e in smld_ds.emitters]
        y = [e.y for e in smld_ds.emitters]
        im = histimage2D(x, y; ROI=ROI, histbinsize=histbinsize)
        histimages[ds] = im

        # Save histogram image
        fig = Figure(size=(800, 800))
        ax = Axis(fig[1, 1], aspect=DataAspect(),
                  title="Dataset $ds Histogram ($(size(im,1))×$(size(im,2)))")
        heatmap!(ax, im', colormap=:turbo)
        save(joinpath(debug_dir, "hist_ds$(ds).png"), fig)
        println("  Saved: hist_ds$(ds).png ($(length(smld_ds.emitters)) emitters)")
    end

    # =========================================================================
    # 4. Compute and save cross-correlations
    # =========================================================================
    println("\n[4/4] Computing cross-correlations...")

    im_ref = histimages[1]
    println("  Reference image size: $(size(im_ref))")

    open(joinpath(debug_dir, "fft_results.txt"), "w") do io
        println(io, "FFT Debug Results")
        println(io, "=" ^ 40)
        println(io)
        println(io, "True shifts (μm):")
        for ds = 1:n_datasets
            x, y = model_true.inter[ds].dm[1], model_true.inter[ds].dm[2]
            println(io, "  DS $ds: ($x, $y)")
        end
        println(io)
        println(io, "FFT-detected shifts:")
    end

    for ds = 2:n_datasets
        im_ds = histimages[ds]

        # Compute cross-correlation (with zero-padding)
        cc = crosscorr2D(Float64.(im_ref), Float64.(im_ds))

        # Calculate center (where shift=0 should appear)
        mid1 = size(cc, 1) ÷ 2 + 1
        mid2 = size(cc, 2) ÷ 2 + 1

        # Find peak
        peak_idx = argmax(cc)
        peak_val = cc[peak_idx]

        # Calculate shift in pixels (negated: ccorr peaks at -shift)
        shift_px = [mid1 - peak_idx[1], mid2 - peak_idx[2]]

        # Convert to μm
        shift_um = histbinsize .* shift_px

        # True shift
        true_shift = [model_true.inter[ds].dm[1], model_true.inter[ds].dm[2]]
        error_nm = sqrt(sum((shift_um .- true_shift).^2)) * 1000

        println("  DS $ds:")
        @printf("    CC size: %s, center: (%d, %d)\n", size(cc), mid1, mid2)
        @printf("    Peak at: (%d, %d), value: %.2f\n", peak_idx[1], peak_idx[2], peak_val)
        @printf("    Shift (px): (%.1f, %.1f)\n", shift_px[1], shift_px[2])
        @printf("    Shift (μm): (%.4f, %.4f)\n", shift_um[1], shift_um[2])
        @printf("    True  (μm): (%.4f, %.4f)\n", true_shift[1], true_shift[2])
        @printf("    Error: %.1f nm\n", error_nm)

        # Save to results file
        open(joinpath(debug_dir, "fft_results.txt"), "a") do io
            println(io, "  DS $ds:")
            @printf(io, "    Detected: (%.4f, %.4f) μm\n", shift_um[1], shift_um[2])
            @printf(io, "    True:     (%.4f, %.4f) μm\n", true_shift[1], true_shift[2])
            @printf(io, "    Error:    %.1f nm\n", error_nm)
        end

        # Save CC image with markers
        fig = Figure(size=(1000, 800))

        # Full CC image
        ax1 = Axis(fig[1, 1], aspect=DataAspect(),
                   title="Cross-correlation DS1 vs DS$ds")
        hm = heatmap!(ax1, cc', colormap=:turbo)
        # Mark center (shift=0)
        scatter!(ax1, [mid1], [mid2], color=:white, marker=:circle,
                markersize=15, strokewidth=2, strokecolor=:black)
        # Mark peak
        scatter!(ax1, [peak_idx[1]], [peak_idx[2]], color=:red, marker=:star5,
                markersize=20, strokewidth=2, strokecolor=:white)
        Colorbar(fig[1, 2], hm)

        # Zoomed view around center
        zoom_range = 100  # pixels
        x_range = max(1, mid1-zoom_range):min(size(cc,1), mid1+zoom_range)
        y_range = max(1, mid2-zoom_range):min(size(cc,2), mid2+zoom_range)
        cc_zoom = cc[x_range, y_range]

        ax2 = Axis(fig[2, 1], aspect=DataAspect(),
                   title="Zoomed (±$zoom_range px around center)")
        hm2 = heatmap!(ax2, cc_zoom', colormap=:turbo)
        # Mark center in zoomed view
        center_in_zoom = [zoom_range+1, zoom_range+1]
        scatter!(ax2, [center_in_zoom[1]], [center_in_zoom[2]], color=:white,
                marker=:circle, markersize=15, strokewidth=2, strokecolor=:black)
        # Mark peak in zoomed view if visible
        peak_in_zoom = [peak_idx[1] - (mid1-zoom_range), peak_idx[2] - (mid2-zoom_range)]
        if 1 <= peak_in_zoom[1] <= 2*zoom_range+1 && 1 <= peak_in_zoom[2] <= 2*zoom_range+1
            scatter!(ax2, [peak_in_zoom[1]], [peak_in_zoom[2]], color=:red,
                    marker=:star5, markersize=20, strokewidth=2, strokecolor=:white)
        end
        Colorbar(fig[2, 2], hm2)

        # Add text annotation
        Label(fig[0, :], "White circle = center (shift=0), Red star = detected peak\n" *
              @sprintf("Shift: (%.4f, %.4f) μm, True: (%.4f, %.4f) μm, Error: %.1f nm",
                      shift_um[1], shift_um[2], true_shift[1], true_shift[2], error_nm),
              fontsize=14)

        save(joinpath(debug_dir, "cc_ds1_vs_ds$(ds).png"), fig)
        println("    Saved: cc_ds1_vs_ds$(ds).png")
    end

    println("\n" * "=" ^ 60)
    println("FFT DEBUG COMPLETE")
    println("Output: $debug_dir")
    println("=" ^ 60)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    debug_fft()
end
