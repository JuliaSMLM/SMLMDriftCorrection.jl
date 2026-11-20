# align_smld(transform=:affine): info.diagnostic is the correction that was applied.
using Test, SMLMDriftCorrection, SMLMSim, Random

const DC = SMLMDriftCorrection

@testset "align_smld affine: diagnostic replays the applied correction" begin
    Random.seed!(11)
    cam = IdealCamera(1:128, 1:128, 0.1)
    fl = GenericFluor(; photons = 5000.0, k_on = 0.5, k_off = 50.0)
    cfg = StaticSMLMConfig(12.0, 0.13, 50, 1, 300, 50.0, 2, [0.0, 1.0])
    (reference, _) = simulate(cfg; pattern = Nmer2D(n = 1, d = 0.0), molecule = fl, camera = cam)

    # Target: the reference under a known rotation, scale and shift about the field center.
    θ, s, t = deg2rad(1.0), 1.01, (0.12, -0.08)
    x0, y0 = 6.4, 6.4
    target = deepcopy(reference)
    for e in target.emitters
        dx, dy = e.x - x0, e.y - y0
        e.x = x0 + s * (cos(θ) * dx - sin(θ) * dy) + t[1]
        e.y = y0 + s * (sin(θ) * dx + cos(θ) * dy) + t[2]
    end

    (aligned, info) = align_smld([reference, target]; transform = :affine)
    (a, b, c, d, e, f) = info.diagnostic.affine_coeffs[2]
    g = info.diagnostic.global_shifts[2]

    # The alignment itself recovers the reference.
    resid = maximum(hypot(al.x - r.x, al.y - r.y)
                    for (al, r) in zip(aligned[2].emitters, reference.emitters))
    @test resid < 0.020

    # Replaying the recorded correction on the input gives the returned positions:
    # subtract the global shift, then the affine.
    worst = 0.0
    for (in_e, out_e) in zip(target.emitters, aligned[2].emitters)
        x = in_e.x - g[1]
        y = in_e.y - g[2]
        xr = x - (a * x + b * y + c)
        yr = y - (d * x + e * y + f)
        worst = max(worst, abs(xr - out_e.x), abs(yr - out_e.y))
    end
    @test worst < 1.0e-9   # μm; rounding only

    # The reference is returned unchanged with a zero record.
    @test info.diagnostic.affine_coeffs[1] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    @test info.diagnostic.global_shifts[1] == [0.0, 0.0]
end
