#!/usr/bin/env julia
# Quick profile run - just allocation analysis + Profile flamegraph
using SMLMDriftCorrection
using SMLMSim
using Random
using Profile

const DC = SMLMDriftCorrection
Random.seed!(42)

params = StaticSMLMConfig(10.0, 0.13, 30, 5, 5000, 50.0, 2, [0.0, 1.0])
smld, _ = simulate(params;
    pattern=Nmer2D(n=6, d=0.2),
    molecule=GenericFluor(; photons=5000.0, k_on=0.02, k_off=50.0),
    camera=IdealCamera(1:64, 1:64, 0.1))

drift_model = DC.LegendrePolynomial(smld; degree=2, initialize="random", rscale=0.1)
smld_drifted = DC.applydrift(smld, drift_model)

println("Data: $(length(smld_drifted.emitters)) locs, $(smld.n_datasets) DS")

# Warmup
driftcorrect(smld_drifted; quality=:singlepass, verbose=0)

# Allocation analysis
println("\n=== ALLOCATION ANALYSIS ===")

alloc_full = @timed driftcorrect(smld_drifted; quality=:singlepass, verbose=0)
println("Full singlepass: $(round(alloc_full.bytes / 1e6, digits=1)) MB, $(round(alloc_full.time, digits=3))s")

model = DC.LegendrePolynomial(smld_drifted; degree=2)
for nn in 1:smld.n_datasets
    DC.initialize_random!(model.intra[nn], 0.01, smld.n_frames)
end

alloc_intra = @timed DC.findintra!(model.intra[1], smld_drifted, 1, 200)
println("findintra! (1 DS, $(sum(e.dataset==1 for e in smld_drifted.emitters)) locs): $(round(alloc_intra.bytes / 1e6, digits=1)) MB, $(round(alloc_intra.time, digits=3))s")

precomp = DC.correctdrift(smld_drifted, model)
alloc_inter = @timed DC.findinter!(model, smld_drifted, 2, [1], 200; precomputed_corrected=precomp)
println("findinter! (1 DS): $(round(alloc_inter.bytes / 1e6, digits=1)) MB, $(round(alloc_inter.time, digits=3))s")

alloc_cd = @timed DC.correctdrift(smld_drifted, model)
println("correctdrift: $(round(alloc_cd.bytes / 1e6, digits=1)) MB, $(round(alloc_cd.time, digits=3))s")

smld_c = DC.correctdrift(smld_drifted, model)
alloc_ent = @timed DC._compute_entropy(smld_c, 200)
println("_compute_entropy: $(round(alloc_ent.bytes / 1e6, digits=1)) MB, $(round(alloc_ent.time, digits=3))s")

# Profile
println("\n=== PROFILE (flat, top entries) ===")
Profile.clear()
@profile driftcorrect(smld_drifted; quality=:singlepass, verbose=0)
Profile.print(maxdepth=15, mincount=10, noisefloor=2, sortedby=:count)
