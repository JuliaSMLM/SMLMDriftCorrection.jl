# Capture range of the intra fit with the background floor. The floor (`_BG_C`) ends a pair's pull at a few σ,
# which keeps sparse data from running away but, fitted from zero at the true σ alone, also stops the fit short of
# a large drift on dense data: on this SMLMSim fixture a 600 nm RMSD smooth drift was left almost uncorrected. The
# coarse first stage (σ × `_INTRA_COARSE_SCALE`) restores the range that 0.2.10 had without a floor.
using Test, SMLMDriftCorrection, SMLMSim, Random, Statistics

const DC = SMLMDriftCorrection

# A smooth drift (Legendre P1 + P2 per axis) scaled to RMSD R (μm) over the localizations.
function capture_drift(rng, frames, n_frames, R)
    a = randn(rng, 4)
    t = 2 .* (frames .- 1) ./ max(1, n_frames - 1) .- 1
    p2 = (3 .* t .^ 2 .- 1) ./ 2
    dx = a[1] .* t .+ a[2] .* p2
    dy = a[3] .* t .+ a[4] .* p2
    s = sqrt(mean((dx .- mean(dx)) .^ 2 .+ (dy .- mean(dy)) .^ 2))
    return R .* dx ./ s, R .* dy ./ s
end

@testset "Intra capture: 600 nm RMSD smooth drift on dense simulated data" begin
    Random.seed!(2)
    cam = IdealCamera(1:128, 1:128, 0.1)
    (truth, _) = simulate(StaticSMLMConfig(4.0, 0.13, 30, 1, 2000, 50.0, 2, [0.0, 1.0]);
                          pattern = Nmer2D(n = 8, d = 0.1),
                          molecule = GenericFluor(; photons = 5000.0, k_on = 0.02, k_off = 50.0), camera = cam)
    x0 = Float64[e.x for e in truth.emitters]
    y0 = Float64[e.y for e in truth.emitters]
    frames = Int[e.frame for e in truth.emitters]
    for draw in 1:3
        dx, dy = capture_drift(MersenneTwister(600 + draw), frames, truth.n_frames, 0.6)
        em = deepcopy(truth.emitters)
        for (i, e) in enumerate(em)
            e.x += oftype(e.x, dx[i])
            e.y += oftype(e.y, dy[i])
        end
        drifted = BasicSMLD(em, cam, truth.n_frames, 1, Dict{String,Any}())
        Random.seed!(700 + draw)
        (corrected, _) = driftcorrect(drifted; degree = 2)
        ex = Float64[e.x for e in corrected.emitters] .- x0
        ey = Float64[e.y for e in corrected.emitters] .- y0
        residual_nm = 1000 * sqrt(mean((ex .- mean(ex)) .^ 2 .+ (ey .- mean(ey)) .^ 2))
        @test residual_nm < 10
    end
end
