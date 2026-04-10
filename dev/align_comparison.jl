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
# Affine: FFT-only (Fourier-Mellin, < 2s)
# ============================================================================

println("\n--- Affine alignment (FFT only, Fourier-Mellin) ---")
@time t_fft = DC.find_affine_fft(smld1, smld2; histbinsize=0.05)
println("  FFT affine: θ=$(round(rad2deg(t_fft.θ); digits=4))°, s=$(round(t_fft.scale; digits=6)), shift=($(round(t_fft.tx; digits=4)), $(round(t_fft.ty; digits=4))) μm")

# Apply FFT affine to full data
aligned_affine = [smld1, deepcopy(smld2)]
DC.apply_affine_transform!(aligned_affine[2], t_fft)

t = t_fft
info_affine = (; transforms=[DC.AffineTransform2D(0.0, 1.0, 0.0, 0.0), t_fft])

# ============================================================================
# Render helper
# ============================================================================

# ============================================================================
# Render helpers
# ============================================================================

function filter_roi(smld, xr, yr)
    mask = [(e.x >= xr[1] && e.x <= xr[2] && e.y >= yr[1] && e.y <= yr[2]) for e in smld.emitters]
    DC.filter_emitters(smld, mask)
end

# ============================================================================
# Find a dense 3×3 μm region for detail views
# ============================================================================

println("\nFinding dense region for detail renders...")
roi_indices = DC.find_dense_roi(smld1, 5000)
x_roi = [smld1.emitters[i].x for i in roi_indices]
y_roi = [smld1.emitters[i].y for i in roi_indices]
roi_cx = (minimum(x_roi) + maximum(x_roi)) / 2
roi_cy = (minimum(y_roi) + maximum(y_roi)) / 2
roi_half = 1.5  # μm → 3×3 μm detail region
xr = (roi_cx - roi_half, roi_cx + roi_half)
yr = (roi_cy - roi_half, roi_cy + roi_half)
println("  Detail ROI: x=[$(round(xr[1];digits=2)), $(round(xr[2];digits=2))], y=[$(round(yr[1];digits=2)), $(round(yr[2];digits=2))] μm")

# ============================================================================
# Render: overview (full FOV, zoom=10)
# ============================================================================

println("\n--- Rendering overview (zoom=10) ---")

render([smld1, smld2]; colors=[:magenta, :green], zoom=10,
       filename=joinpath(OUTDIR, "01_before_overview.png"))
println("  saved: 01_before_overview.png")

render([aligned_shift[1], aligned_shift[2]]; colors=[:magenta, :green], zoom=10,
       filename=joinpath(OUTDIR, "02_shift_overview.png"))
println("  saved: 02_shift_overview.png")

render([aligned_affine[1], aligned_affine[2]]; colors=[:magenta, :green], zoom=10,
       filename=joinpath(OUTDIR, "03_affine_overview.png"))
println("  saved: 03_affine_overview.png")

# ============================================================================
# Render: detail (3×3 μm ROI, 5nm pixels)
# ============================================================================

println("\n--- Rendering detail (5nm/px, 3×3μm ROI) ---")

render([filter_roi(smld1, xr, yr), filter_roi(smld2, xr, yr)];
       colors=[:magenta, :green], pixel_size=5.0,
       filename=joinpath(OUTDIR, "04_before_detail.png"))
println("  saved: 04_before_detail.png")

render([filter_roi(aligned_shift[1], xr, yr), filter_roi(aligned_shift[2], xr, yr)];
       colors=[:magenta, :green], pixel_size=5.0,
       filename=joinpath(OUTDIR, "05_shift_detail.png"))
println("  saved: 05_shift_detail.png")

render([filter_roi(aligned_affine[1], xr, yr), filter_roi(aligned_affine[2], xr, yr)];
       colors=[:magenta, :green], pixel_size=5.0,
       filename=joinpath(OUTDIR, "06_affine_detail.png"))
println("  saved: 06_affine_detail.png")

println("\nDone!")
