#!/usr/bin/env julia
# Regression check: captures numerical fingerprint of driftcorrect output.
# Run before and after optimization to verify identical results.
#
# Usage: julia --project=dev --threads=4 dev/regression_check.jl

using SMLMDriftCorrection
using SMLMSim
using Random
using Statistics

const DC = SMLMDriftCorrection

function fingerprint(smld, info, label)
    x = [e.x for e in smld.emitters]
    y = [e.y for e in smld.emitters]
    σ_x = [e.σ_x for e in smld.emitters]
    println("  [$label] n=$(length(x))")
    println("  [$label] sum_x=$(sum(x))")
    println("  [$label] sum_y=$(sum(y))")
    println("  [$label] mean_x=$(mean(x))")
    println("  [$label] mean_y=$(mean(y))")
    println("  [$label] std_x=$(std(x))")
    println("  [$label] std_y=$(std(y))")
    println("  [$label] entropy=$(info.entropy)")
    println("  [$label] iterations=$(info.iterations)")
    println("  [$label] converged=$(info.converged)")

    # Model fingerprint
    m = info.model
    for ds in 1:m.ndatasets
        for dim in 1:m.intra[ds].ndims
            coeffs = m.intra[ds].dm[dim].coefficients
            println("  [$label] intra_ds$(ds)_dim$(dim)=$(coeffs)")
        end
        println("  [$label] inter_ds$(ds)=$(m.inter[ds].dm)")
    end
end

# Deterministic data generation
Random.seed!(12345)

# 2D test
params = StaticSMLMConfig(10.0, 0.13, 30, 3, 1000, 50.0, 2, [0.0, 1.0])
smld, _ = simulate(params;
    pattern=Nmer2D(n=6, d=0.2),
    molecule=GenericFluor(; photons=5000.0, k_on=0.02, k_off=50.0),
    camera=IdealCamera(1:64, 1:64, 0.1))

# Apply known drift
Random.seed!(99)
drift_model = DC.LegendrePolynomial(smld; degree=2, initialize="random", rscale=0.1)
smld_drifted = DC.applydrift(smld, drift_model)

println("Data: $(length(smld_drifted.emitters)) locs, $(smld.n_datasets) datasets")

# Test all quality tiers with fixed seed
println("\n=== FFT ===")
Random.seed!(42)
(c_fft, i_fft) = driftcorrect(smld_drifted; quality=:fft)
fingerprint(c_fft, i_fft, "fft")

println("\n=== SINGLEPASS ===")
Random.seed!(42)
(c_sp, i_sp) = driftcorrect(smld_drifted; quality=:singlepass)
fingerprint(c_sp, i_sp, "sp")

println("\n=== ITERATIVE (3 iter) ===")
Random.seed!(42)
(c_it, i_it) = driftcorrect(smld_drifted; quality=:iterative, max_iterations=3)
fingerprint(c_it, i_it, "iter")

println("\nDONE")
