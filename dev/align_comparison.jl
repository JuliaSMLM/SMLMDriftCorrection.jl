# align_comparison.jl — Compare shift vs affine alignment on GATTA ruler pairs
#
# Loads two independent DNA PAINT acquisitions of the same 20nm GATTA ruler,
# aligns them with both :shift and :affine, and renders two-color overlays
# (before/after) to visually assess alignment quality.
#
# Usage: julia --project=dev dev/align_comparison.jl

using SMLMData
using SMLMDriftCorrection
using SMLMRender
using CairoMakie
using JLD2

DC = SMLMDriftCorrection

# ============================================================================
# Data paths — 2025-10-23 GATTA 20R ruler pair (primary paper dataset)
# ============================================================================

const DATA_ROOT = "/mnt/nas/adapt/projects/smart-microscope/data/DNA paint ruler/papers-vortex-sr/juliasmlm"
const ACQ1 = "20R-ruler-0.1exp-TIRF-onlyZFocusLockDT1-1ch--2025-10-23_11-29-25"
const ACQ2 = "20R-ruler-0.1exp-TIRF-onlyZFocusLockDT1-1ch-N2--2025-10-23_13-19-34"
const OUTDIR = joinpath(@__DIR__, "output", "align_comparison")

mkpath(OUTDIR)

# ============================================================================
# Load data
# ============================================================================

"""Load smld_final.h5 (JLD2/SMLMData format) into an SMLD struct."""
function load_smld_h5(path::String)
    d = JLD2.load(path)
    x = Float64.(d["emitters/x"])
    y = Float64.(d["emitters/y"])
    σ_x = Float64.(d["emitters/sigma_x"])
    σ_y = Float64.(d["emitters/sigma_y"])
    photons = Float64.(d["emitters/photons"])
    bg = Float64.(d["emitters/bg"])
    σ_photons = Float64.(d["emitters/sigma_photons"])
    σ_bg = Float64.(d["emitters/sigma_bg"])
    frame = Int.(d["emitters/frame"])
    dataset = Int.(d["emitters/dataset"])
    id = Int.(d["emitters/id"])
    track_id = Int.(d["emitters/track_id"])

    px_edges_x = Float64.(d["camera/pixel_edges_x"])
    px_edges_y = Float64.(d["camera/pixel_edges_y"])
    pixelsize = px_edges_x[2] - px_edges_x[1]

    N = length(x)
    # Emitter2DFit field order: x, y, photons, bg, σ_x, σ_y, σ_xy, σ_photons, σ_bg, frame, dataset, track_id, id
    emitters = [Emitter2DFit(x[i], y[i], photons[i], bg[i], σ_x[i], σ_y[i], 0.0, σ_photons[i], σ_bg[i],
                             frame[i], dataset[i], track_id[i], id[i]) for i in 1:N]

    n_datasets = d["metadata/n_datasets"]
    n_frames = d["metadata/n_frames"]
    camera = IdealCamera(px_edges_x, px_edges_y)
    return BasicSMLD(emitters, camera, n_frames, n_datasets)
end

println("Loading acquisitions...")
smld1 = load_smld_h5(joinpath(DATA_ROOT, ACQ1, "smld_final.h5"))
smld2 = load_smld_h5(joinpath(DATA_ROOT, ACQ2, "smld_final.h5"))
println("  ACQ1: $(length(smld1.emitters)) locs")
println("  ACQ2: $(length(smld2.emitters)) locs")

# ============================================================================
# Align: shift only
# ============================================================================

println("\n--- Shift alignment ---")
(aligned_shift, info_shift) = align_smld([smld1, smld2]; transform=:shift, verbose=1)

# ============================================================================
# Align: affine
# ============================================================================

# ============================================================================
# Align: affine — use maxn=50 for speed on large datasets
# ============================================================================

println("\n--- Affine alignment ---")
@time (aligned_affine, info_affine) = align_smld([smld1, smld2]; transform=:affine, maxn=50, verbose=1)

t = info_affine.transforms[2]
println("  Affine transform:")
println("    rotation = $(round(rad2deg(t.θ); digits=4))°")
println("    scale    = $(round(t.scale; digits=6))")
println("    shift    = ($(round(t.tx; digits=4)), $(round(t.ty; digits=4))) μm")

# ============================================================================
# Render helper
# ============================================================================

function render_overlay(smld_a, smld_b, zoom; roi=nothing)
    if isnothing(roi)
        (img, info) = render([smld_a, smld_b]; colors=[:magenta, :green], zoom=zoom)
    else
        # Multi-SMLD render doesn't support roi — filter emitters manually
        cam_px = smld_a.camera.pixel_edges_x[2] - smld_a.camera.pixel_edges_x[1]
        xmin = cam_px * (first(roi[1]) - 1)
        xmax = cam_px * last(roi[1])
        ymin = cam_px * (first(roi[2]) - 1)
        ymax = cam_px * last(roi[2])
        mask_a = [(e.x >= xmin && e.x <= xmax && e.y >= ymin && e.y <= ymax) for e in smld_a.emitters]
        mask_b = [(e.x >= xmin && e.x <= xmax && e.y >= ymin && e.y <= ymax) for e in smld_b.emitters]
        smld_a_roi = DC.filter_emitters(smld_a, mask_a)
        smld_b_roi = DC.filter_emitters(smld_b, mask_b)
        (img, info) = render([smld_a_roi, smld_b_roi]; colors=[:magenta, :green], pixel_size=cam_px * 1000 / zoom)
    end
    return img
end

function save_panel(img, path)
    save_image(path, img)
    println("  saved: $path")
end

# ============================================================================
# Determine a good ROI — find a dense region for zoomed view
# ============================================================================

# Use full FOV at moderate zoom for overview
zoom_overview = 10
zoom_detail = 40

# Compute centroid of smld1 to pick a central ROI
x1 = [e.x for e in smld1.emitters]
y1 = [e.y for e in smld1.emitters]
cx = (minimum(x1) + maximum(x1)) / 2
cy = (minimum(y1) + maximum(y1)) / 2

# Camera pixel size from edge array
cam_px = smld1.camera.pixel_edges_x[2] - smld1.camera.pixel_edges_x[1]  # μm per camera pixel
cx_px = round(Int, cx / cam_px)
cy_px = round(Int, cy / cam_px)
half_roi = 128
roi_detail = (max(1, cx_px - half_roi):cx_px + half_roi,
              max(1, cy_px - half_roi):cy_px + half_roi)

println("\nRendering overlays (overview zoom=$(zoom_overview), detail zoom=$(zoom_detail))...")
println("  Detail ROI (camera px): x=$(roi_detail[1]), y=$(roi_detail[2])")
println("  Centroid: ($(round(cx; digits=2)), $(round(cy; digits=2))) μm")

# ============================================================================
# Render: before alignment (raw)
# ============================================================================

println("\n--- Before alignment ---")
img_before_overview = render_overlay(smld1, smld2, zoom_overview)
save_panel(img_before_overview, joinpath(OUTDIR, "01_before_overview.png"))

img_before_detail = render_overlay(smld1, smld2, zoom_detail; roi=roi_detail)
save_panel(img_before_detail, joinpath(OUTDIR, "02_before_detail.png"))

# ============================================================================
# Render: after shift alignment
# ============================================================================

println("\n--- After shift alignment ---")
img_shift_overview = render_overlay(aligned_shift[1], aligned_shift[2], zoom_overview)
save_panel(img_shift_overview, joinpath(OUTDIR, "03_shift_overview.png"))

img_shift_detail = render_overlay(aligned_shift[1], aligned_shift[2], zoom_detail; roi=roi_detail)
save_panel(img_shift_detail, joinpath(OUTDIR, "04_shift_detail.png"))

# ============================================================================
# Render: after affine alignment
# ============================================================================

println("\n--- After affine alignment ---")
img_affine_overview = render_overlay(aligned_affine[1], aligned_affine[2], zoom_overview)
save_panel(img_affine_overview, joinpath(OUTDIR, "05_affine_overview.png"))

img_affine_detail = render_overlay(aligned_affine[1], aligned_affine[2], zoom_detail; roi=roi_detail)
save_panel(img_affine_detail, joinpath(OUTDIR, "06_affine_detail.png"))

# ============================================================================
# CairoMakie comparison figure
# ============================================================================

println("\n--- Building comparison figure ---")

fig = Figure(size=(1800, 1200))

# Row 1: Overview
ax1 = Axis(fig[1, 1]; title="Before alignment", aspect=DataAspect(),
           yreversed=true, titlesize=16)
ax2 = Axis(fig[1, 2]; title="Shift aligned", aspect=DataAspect(),
           yreversed=true, titlesize=16)
ax3 = Axis(fig[1, 3]; title="Affine aligned", aspect=DataAspect(),
           yreversed=true, titlesize=16)

image!(ax1, rotr90(img_before_overview))
image!(ax2, rotr90(img_shift_overview))
image!(ax3, rotr90(img_affine_overview))

hidexdecorations!.([ax1, ax2, ax3])
hideydecorations!.([ax1, ax2, ax3])

Label(fig[1, 0], "Overview\n($(zoom_overview)×)", fontsize=14, rotation=pi/2)

# Row 2: Detail
ax4 = Axis(fig[2, 1]; aspect=DataAspect(), yreversed=true)
ax5 = Axis(fig[2, 2]; aspect=DataAspect(), yreversed=true)
ax6 = Axis(fig[2, 3]; aspect=DataAspect(), yreversed=true)

image!(ax4, rotr90(img_before_detail))
image!(ax5, rotr90(img_shift_detail))
image!(ax6, rotr90(img_affine_detail))

hidexdecorations!.([ax4, ax5, ax6])
hideydecorations!.([ax4, ax5, ax6])

Label(fig[2, 0], "Detail\n($(zoom_detail)×)", fontsize=14, rotation=pi/2)

# Annotation
shift_str = "Δ = ($(round(info_shift.shifts[2][1]; digits=3)), $(round(info_shift.shifts[2][2]; digits=3))) μm"
affine_str = "θ=$(round(rad2deg(t.θ); digits=3))°, s=$(round(t.scale; digits=5))\nΔ=($(round(t.tx; digits=3)), $(round(t.ty; digits=3))) μm"

Label(fig[3, 2], shift_str; fontsize=12, color=:gray50)
Label(fig[3, 3], affine_str; fontsize=12, color=:gray50)

# Color legend
Label(fig[0, 1:3], "GATTA 20R ruler — 2025-10-23 — magenta: ACQ1, green: ACQ2 (white = overlap)",
      fontsize=14, color=:gray30)

outpath = joinpath(OUTDIR, "comparison.png")
save(outpath, fig; px_per_unit=2)
println("\nComparison figure saved: $outpath")

println("\nDone!")
