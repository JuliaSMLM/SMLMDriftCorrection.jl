# align_comparison.jl — Compare shift vs affine alignment on 1ch vs 2ch DNA PAINT
#
# Usage: julia --project=dev dev/align_comparison.jl

using SMLMData
using SMLMDriftCorrection
using SMLMRender
using JLD2

DC = SMLMDriftCorrection

# ============================================================================
# Data: 1ch vs 2ch polarization channels, 2025-10-23 GATTA 20R ruler
# Same sample, different optical paths → expect small rotation/scale difference
# ============================================================================

const DATA_DIR = joinpath(homedir(), "julia_shared_dev/papers/papers-vortex-sr/data/processed")
const CH1_PATH = joinpath(DATA_DIR, "20R-ruler-0.1exp-TIRF-onlyZFocusLockDT1-1ch--2025-10-23_11-29-25/loc.h5")
const CH2_PATH = joinpath(DATA_DIR, "20R-ruler-0.1exp-TIRF-onlyZFocusLockDT1-2ch--2025-10-23_12-04-53/loc.h5")
const OUTDIR = joinpath(@__DIR__, "output", "align_comparison")
mkpath(OUTDIR)

# ============================================================================
# Load DME-format loc.h5 → SMLD
# ============================================================================

function load_dme_h5(path; pixel_size=0.078)
    d = JLD2.load(path)
    x = Float64.(d["res/x"]) .* pixel_size   # pixels → μm
    y = Float64.(d["res/y"]) .* pixel_size
    crlb = d["res/crlb"]  # (N, 5): σ_x, σ_y, σ_photon, σ_bg, σ_z
    σ_x = Float64.(crlb[:, 1])
    σ_y = Float64.(crlb[:, 2])
    photons = Float64.(d["res/photon"])
    bg = Float64.(d["res/bg"])
    frame = Int.(d["res/frames"])

    N = length(x)
    emitters = [Emitter2DFit(x[i], y[i], photons[i], bg[i], σ_x[i], σ_y[i],
                             0.0, 0.0, 0.0, frame[i], 1, 0, i) for i in 1:N]

    # Build camera from data extent
    nx = round(Int, (maximum(x) - minimum(x)) / pixel_size) + 10
    ny = round(Int, (maximum(y) - minimum(y)) / pixel_size) + 10
    x_edges = collect(range(floor(minimum(x); digits=1), step=pixel_size, length=nx+1))
    y_edges = collect(range(floor(minimum(y); digits=1), step=pixel_size, length=ny+1))
    camera = IdealCamera(x_edges, y_edges)
    return BasicSMLD(emitters, camera, maximum(frame), 1)
end

# Load both, then rebuild with shared camera (required for findshift)
println("Loading 1ch and 2ch...")
smld1_raw = load_dme_h5(CH1_PATH)
smld2_raw = load_dme_h5(CH2_PATH)

# Build common camera covering both FOVs
pixel_size = 0.078
all_x = vcat([e.x for e in smld1_raw.emitters], [e.x for e in smld2_raw.emitters])
all_y = vcat([e.y for e in smld1_raw.emitters], [e.y for e in smld2_raw.emitters])
x_min = floor(minimum(all_x); digits=1)
y_min = floor(minimum(all_y); digits=1)
nx = round(Int, (ceil(maximum(all_x); digits=1) - x_min) / pixel_size) + 2
ny = round(Int, (ceil(maximum(all_y); digits=1) - y_min) / pixel_size) + 2
shared_camera = IdealCamera(
    collect(range(x_min, step=pixel_size, length=nx+1)),
    collect(range(y_min, step=pixel_size, length=ny+1)))

smld1 = BasicSMLD(smld1_raw.emitters, shared_camera, smld1_raw.n_frames, 1)
smld2 = BasicSMLD(smld2_raw.emitters, shared_camera, smld2_raw.n_frames, 1)
println("  1ch: $(length(smld1.emitters)) locs")
println("  2ch: $(length(smld2.emitters)) locs")

# ============================================================================
# Align
# ============================================================================

println("\n--- Fourier-Mellin FFT affine ---")
@time t_fft = DC.find_affine_fft(smld1, smld2; histbinsize=0.05)
println("  θ = $(round(rad2deg(t_fft.θ); digits=4))°")
println("  scale = $(round(t_fft.scale; digits=6))")
println("  shift = ($(round(t_fft.tx; digits=4)), $(round(t_fft.ty; digits=4))) μm")

println("\n--- Entropy shift alignment ---")
@time (aligned_shift, info_shift) = DC.align_smld([smld1, smld2]; transform=:shift, verbose=1)

# Apply FFT affine to full data for rendering
aligned_affine = [smld1, deepcopy(smld2)]
DC.apply_affine_transform!(aligned_affine[2], t_fft)

# ============================================================================
# Render helpers
# ============================================================================

function filter_roi(smld, xr, yr)
    mask = [(e.x >= xr[1] && e.x <= xr[2] && e.y >= yr[1] && e.y <= yr[2]) for e in smld.emitters]
    DC.filter_emitters(smld, mask)
end

# ============================================================================
# Find dense ROI for detail views
# ============================================================================

println("\nFinding dense region...")
roi_indices = DC.find_dense_roi(smld1, 10000)
x_roi = [smld1.emitters[i].x for i in roi_indices]
y_roi = [smld1.emitters[i].y for i in roi_indices]
roi_cx = (minimum(x_roi) + maximum(x_roi)) / 2
roi_cy = (minimum(y_roi) + maximum(y_roi)) / 2
roi_half = 1.5  # μm → 3×3 μm
xr = (roi_cx - roi_half, roi_cx + roi_half)
yr = (roi_cy - roi_half, roi_cy + roi_half)
println("  ROI center: ($(round(roi_cx; digits=2)), $(round(roi_cy; digits=2))) μm")

# ============================================================================
# Render
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

println("\n--- Rendering detail (5nm/px) ---")
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
