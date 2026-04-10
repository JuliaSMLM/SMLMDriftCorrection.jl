# align_comparison.jl — Compare shift vs affine alignment: 1ch vs 2ch cell data
#
# HeLa cell DNA PAINT, 2026-04-02. 1ch and 2ch are different polarization
# channels imaging the same cell — expect rotation/scale from optical path.
#
# Usage: julia --project=dev dev/align_comparison.jl

using SMLMData
using SMLMDriftCorrection
using SMLMRender
using JLD2

DC = SMLMDriftCorrection

# ============================================================================
# Data paths
# ============================================================================

const BASE = joinpath(homedir(), "julia_shared_dev/papers/papers-vortex-sr/data/processed_cells/2026-04-02",
    "Cell1_HELA_noEGF_150pM - right channel ref same as 1ch in 2ch analysis")
# Use dme (drift-corrected, NOT channel-aligned) — DCCh is already aligned
const OUTDIR = joinpath(@__DIR__, "output", "align_comparison")
mkpath(OUTDIR)

# ============================================================================
# Load DME DCCh H5 → SMLD (drift-corrected, dataset-comparable)
# ============================================================================

function load_dme_dcch(path; pixel_size=0.078)
    d = JLD2.load(path)
    x = Float64.(d["res/x"]) .* pixel_size   # pixels → μm
    y = Float64.(d["res/y"]) .* pixel_size
    crlb_raw = d["res/crlb"]
    # Handle both (N, d) and (d, N) orientations
    crlb = ndims(crlb_raw) == 2 && size(crlb_raw, 1) < size(crlb_raw, 2) ? collect(crlb_raw') : crlb_raw
    σ_x = Float64.(crlb[:, 1]) .* pixel_size  # → μm
    σ_y = Float64.(crlb[:, 2]) .* pixel_size
    photons = haskey(d, "res/photon") ? Float64.(d["res/photon"]) : ones(length(x))
    frame = Int.(d["res/frames"])
    N = length(x)
    emitters = [Emitter2DFit(x[i], y[i], photons[i], 0.0, σ_x[i], σ_y[i],
                             0.0, 0.0, 0.0, frame[i], 1, 0, i) for i in 1:N]
    return emitters
end

# Use dme (drift-corrected per channel, NOT channel-aligned)
println("Loading 1ch and 2ch (HeLa cell, drift-corrected, no channel alignment)...")
em1 = load_dme_dcch(joinpath(BASE, "dme_1ch.h5"))
em2 = load_dme_dcch(joinpath(BASE, "dme_2ch.h5"))

# Build shared camera covering both FOVs
pixel_size = 0.078
all_x = vcat([e.x for e in em1], [e.x for e in em2])
all_y = vcat([e.y for e in em1], [e.y for e in em2])
x0 = floor(minimum(all_x) - 0.5; digits=1)
y0 = floor(minimum(all_y) - 0.5; digits=1)
x1 = ceil(maximum(all_x) + 0.5; digits=1)
y1 = ceil(maximum(all_y) + 0.5; digits=1)
nx = round(Int, (x1 - x0) / pixel_size) + 1
ny = round(Int, (y1 - y0) / pixel_size) + 1
cam = IdealCamera(collect(range(x0, step=pixel_size, length=nx+1)),
                  collect(range(y0, step=pixel_size, length=ny+1)))

smld1 = BasicSMLD(em1, cam, maximum(e.frame for e in em1), 1)
smld2 = BasicSMLD(em2, cam, maximum(e.frame for e in em2), 1)
println("  1ch: $(length(smld1.emitters)) locs")
println("  2ch: $(length(smld2.emitters)) locs")

# ============================================================================
# Fourier-Mellin affine (fast, ~2s)
# ============================================================================

println("\n--- Fourier-Mellin FFT affine ---")
@time t_fft = DC.find_affine_fft(smld1, smld2; histbinsize=0.05)
println("  θ = $(round(rad2deg(t_fft.θ); digits=4))°")
println("  scale = $(round(t_fft.scale; digits=6))")
println("  shift = ($(round(t_fft.tx; digits=4)), $(round(t_fft.ty; digits=4))) μm")

# ============================================================================
# align_smld with :shift, :fft method
# ============================================================================

println("\n--- align_smld (shift, fft) ---")
@time (aligned_shift, _) = DC.align_smld([smld1, smld2]; transform=:shift, method=:fft, verbose=1)

# ============================================================================
# Apply FFT affine to full data
# ============================================================================

aligned_affine = [smld1, deepcopy(smld2)]
DC.apply_affine_transform!(aligned_affine[2], t_fft)

# ============================================================================
# Render
# ============================================================================

function filter_roi(smld, xr, yr)
    mask = [(e.x >= xr[1] && e.x <= xr[2] && e.y >= yr[1] && e.y <= yr[2]) for e in smld.emitters]
    DC.filter_emitters(smld, mask)
end

# Find dense ROI
println("\nFinding dense region...")
roi_idx = DC.find_dense_roi(smld1, 20000)
xr_locs = [smld1.emitters[i].x for i in roi_idx]
yr_locs = [smld1.emitters[i].y for i in roi_idx]
cx = (minimum(xr_locs) + maximum(xr_locs)) / 2
cy = (minimum(yr_locs) + maximum(yr_locs)) / 2
half = 2.0  # 4×4 μm ROI
xr = (cx - half, cx + half)
yr = (cy - half, cy + half)
println("  ROI: x=[$(round(xr[1];digits=1)), $(round(xr[2];digits=1))], y=[$(round(yr[1];digits=1)), $(round(yr[2];digits=1))] μm")

println("\n--- Rendering overview (zoom=10) ---")
render([smld1, smld2]; colors=[:magenta, :green], zoom=10,
       filename=joinpath(OUTDIR, "01_before_overview.png"))
println("  01_before_overview.png")

render([aligned_shift[1], aligned_shift[2]]; colors=[:magenta, :green], zoom=10,
       filename=joinpath(OUTDIR, "02_shift_overview.png"))
println("  02_shift_overview.png")

render([aligned_affine[1], aligned_affine[2]]; colors=[:magenta, :green], zoom=10,
       filename=joinpath(OUTDIR, "03_affine_overview.png"))
println("  03_affine_overview.png")

println("\n--- Rendering detail (5nm/px, 4×4μm ROI) ---")
render([filter_roi(smld1, xr, yr), filter_roi(smld2, xr, yr)];
       colors=[:magenta, :green], pixel_size=5.0,
       filename=joinpath(OUTDIR, "04_before_detail.png"))
println("  04_before_detail.png")

render([filter_roi(aligned_shift[1], xr, yr), filter_roi(aligned_shift[2], xr, yr)];
       colors=[:magenta, :green], pixel_size=5.0,
       filename=joinpath(OUTDIR, "05_shift_detail.png"))
println("  05_shift_detail.png")

render([filter_roi(aligned_affine[1], xr, yr), filter_roi(aligned_affine[2], xr, yr)];
       colors=[:magenta, :green], pixel_size=5.0,
       filename=joinpath(OUTDIR, "06_affine_detail.png"))
println("  06_affine_detail.png")

println("\nDone!")
