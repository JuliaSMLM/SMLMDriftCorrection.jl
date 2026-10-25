# Registered inter alignment on the docs Quick Start geometry (seed 42), 2 to 6 datasets.
using Test, SMLMDriftCorrection, SMLMSim, Random

const DC = SMLMDriftCorrection

# -----------------------------------------------------------------------
# Every dataset is aligned against one snapshot of the others while all move, so an
# unscaled update overcounts the relative correction by N/(N-1); with two datasets the
# relative error then flips sign each pass and never shrinks (5.8 nm on this geometry).
# Two datasets align DS2 to DS1 only; N >= 3 damps a small step by (N-1)/N.
# -----------------------------------------------------------------------
const QS_SHIFTS = [[0.0, 0.0], [0.05, -0.03], [-0.04, 0.02], [0.03, 0.045], [-0.02, -0.05], [0.06, 0.01]]

function quickstart_errors(nds)
    Random.seed!(42)
    params = StaticSMLMConfig(7.5, 0.13, 50, nds, 1000, 50.0, 2, [0.0, 1.0])
    (smld_noisy, _) = simulate(params; pattern = Nmer2D(n = 6, d = 0.2),
        molecule = GenericFluor(; photons = 35000.0, k_on = 0.25, k_off = 50.0),
        camera = IdealCamera(1:32, 1:32, 0.1))
    drift_true = DC.LegendrePolynomial(smld_noisy; degree = 2, initialize = "zeros")
    for ds in 1:nds
        drift_true.intra[ds].dm[1].coefficients .= [0.10, 0.03]
        drift_true.intra[ds].dm[2].coefficients .= [0.08, -0.02]
        drift_true.inter[ds].dm .= QS_SHIFTS[ds]
    end
    smld_drifted = DC.applydrift(smld_noisy, drift_true)
    (smld_c, info) = DC.driftcorrect(smld_drifted)
    inter_err = [1000 * hypot(((info.model.inter[k].dm .- info.model.inter[1].dm) .- QS_SHIFTS[k])...)
                 for k in 2:nds]
    rmsd = 1000 * sqrt(sum((a.x - b.x)^2 + (a.y - b.y)^2
                           for (a, b) in zip(smld_c.emitters, smld_noisy.emitters)) / length(smld_c.emitters))
    return (inter = maximum(inter_err), rmsd = rmsd)
end

@testset "Quick Start geometry: registered inter error by dataset count" begin
    r2 = quickstart_errors(2)
    println("quickstart 2 datasets: max inter err = $(round(r2.inter, digits=2)) nm, RMSD = $(round(r2.rmsd, digits=2)) nm")
    @test r2.inter < 1.5
    @test r2.rmsd < 1.5
    r3 = quickstart_errors(3)
    println("quickstart 3 datasets: max inter err = $(round(r3.inter, digits=2)) nm, RMSD = $(round(r3.rmsd, digits=2)) nm")
    @test r3.inter < 1.0
    @test r3.rmsd < 1.0
    # more datasets were already accurate (0.84 and 0.42 nm before the fix); they must not get worse
    r4 = quickstart_errors(4)
    println("quickstart 4 datasets: max inter err = $(round(r4.inter, digits=2)) nm, RMSD = $(round(r4.rmsd, digits=2)) nm")
    @test r4.inter <= 0.85
    r6 = quickstart_errors(6)
    println("quickstart 6 datasets: max inter err = $(round(r6.inter, digits=2)) nm, RMSD = $(round(r6.rmsd, digits=2)) nm")
    @test r6.inter <= 0.45
end
