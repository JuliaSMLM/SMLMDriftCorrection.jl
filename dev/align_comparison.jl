# align_comparison.jl — Shift vs affine alignment on 1ch vs 2ch cell data
#
# HeLa DNA PAINT, 2026-04-02. 1ch and 2ch polarization channels,
# drift-corrected per channel (dme), NOT channel-aligned.
#
# Usage: julia --project=dev dev/align_comparison.jl

using SMLMData
using SMLMDriftCorrection
using SMLMRender
using JLD2

DC = SMLMDriftCorrection

# ============================================================================
# Data
# ============================================================================

const BASE = joinpath(homedir(), "julia_shared_dev/papers/papers-vortex-sr/data/processed_cells/2026-04-02",
    "Cell1_HELA_noEGF_150pM - right channel ref same as 1ch in 2ch analysis")
const OUTDIR = joinpath(@__DIR__, "output", "align_comparison")
mkpath(OUTDIR)

function load_dme(path; pixel_size=0.078)
    d = JLD2.load(path)
    x = Float64.(d["res/x"]) .* pixel_size
    y = Float64.(d["res/y"]) .* pixel_size
    crlb_raw = d["res/crlb"]
    crlb = ndims(crlb_raw) == 2 && size(crlb_raw, 1) < size(crlb_raw, 2) ? collect(crlb_raw') : crlb_raw
    σ_x = Float64.(crlb[:, 1]) .* pixel_size
    σ_y = Float64.(crlb[:, 2]) .* pixel_size
    photons = haskey(d, "res/photon") ? Float64.(d["res/photon"]) : ones(length(x))
    frame = Int.(d["res/frames"])
    N = length(x)
    [Emitter2DFit(x[i], y[i], photons[i], 0.0, σ_x[i], σ_y[i],
                  0.0, 0.0, 0.0, frame[i], 1, 0, i) for i in 1:N]
end

println("Loading..."); flush(stdout)
em1 = load_dme(joinpath(BASE, "dme_1ch.h5"))
em2 = load_dme(joinpath(BASE, "dme_2ch.h5"))

# Shared camera
ps = 0.078
all_x = vcat([e.x for e in em1], [e.x for e in em2])
all_y = vcat([e.y for e in em1], [e.y for e in em2])
x0 = max(0.0, floor(minimum(all_x) - 0.5; digits=1))
y0 = max(0.0, floor(minimum(all_y) - 0.5; digits=1))
x1 = ceil(maximum(all_x) + 0.5; digits=1)
y1 = ceil(maximum(all_y) + 0.5; digits=1)
nx = round(Int, (x1 - x0) / ps) + 1
ny = round(Int, (y1 - y0) / ps) + 1
cam = IdealCamera(collect(range(x0, step=ps, length=nx+1)),
                  collect(range(y0, step=ps, length=ny+1)))
smld1 = BasicSMLD(em1, cam, 1, 1)
smld2 = BasicSMLD(em2, cam, 1, 1)
println("  1ch: $(length(em1)), 2ch: $(length(em2))"); flush(stdout)

# ============================================================================
# Align
# ============================================================================

println("\n--- Shift (FFT) ---"); flush(stdout)
@time (aligned_shift, info_shift) = DC.align_smld([smld1, smld2];
    transform=:shift, method=:fft, verbose=1)

println("\n--- Affine (FFT shift-field) ---"); flush(stdout)
@time (aligned_affine, info_affine) = DC.align_smld([smld1, smld2];
    transform=:affine, method=:fft, verbose=1)

# ============================================================================
# Render: full FOV + 3x3 grid of detail regions
# ============================================================================

function filter_roi(smld, xr, yr)
    mask = [(e.x >= xr[1] && e.x <= xr[2] && e.y >= yr[1] && e.y <= yr[2]) for e in smld.emitters]
    DC.filter_emitters(smld, mask)
end

println("\n--- Rendering full FOV ---"); flush(stdout)
for (tag, s1, s2) in [("before", smld1, smld2),
                       ("shift", aligned_shift[1], aligned_shift[2]),
                       ("affine", aligned_affine[1], aligned_affine[2])]
    render([s1, s2]; colors=[:magenta, :green], zoom=10,
           filename=joinpath(OUTDIR, "full_$(tag).png"))
    println("  full_$(tag).png"); flush(stdout)
end

println("\n--- Rendering detail grid (4×4μm, 5nm/px) ---"); flush(stdout)
grid_labels = ["TL", "TC", "TR", "CL", "CC", "CR", "BL", "BC", "BR"]
grid_x = [2.0, 8.0, 14.0, 2.0, 8.0, 14.0, 2.0, 8.0, 14.0]
grid_y = [2.0, 2.0, 2.0, 8.0, 8.0, 8.0, 14.0, 14.0, 14.0]
side = 4.0

for (label, gx, gy) in zip(grid_labels, grid_x, grid_y)
    xr = (gx, gx + side); yr = (gy, gy + side)
    for (tag, s1, s2) in [("before", smld1, smld2),
                           ("shift", aligned_shift[1], aligned_shift[2]),
                           ("affine", aligned_affine[1], aligned_affine[2])]
        r1 = filter_roi(s1, xr, yr); r2 = filter_roi(s2, xr, yr)
        (length(r1.emitters) < 50 || length(r2.emitters) < 50) && continue
        render([r1, r2]; colors=[:magenta, :green], pixel_size=5.0,
               filename=joinpath(OUTDIR, "$(label)_$(tag).png"))
    end
    println("  $label done"); flush(stdout)
end

println("\nDone!")
