## Demonstrate applying and correcting drift 

using Revise
using SMLMDriftCorrection
DC = SMLMDriftCorrection
using SMLMData
using SMLMSim
using CairoMakie
#using GLMakie
#using Statistics

# make an Nmer dataset
# Simulation parameters use physical units
# smld structures are in units of pixels and frames
println("original noisy data ...")
smld_true, smld_model, smld_noisy = simulate(;
    ρ=1.0,                # emitters per μm²
    σ_psf=0.13,           # PSF width in μm (130nm)
    minphotons=50,        # minimum photons for detection
    ndatasets=10,         # number of independent datasets
    nframes=1000,         # frames per dataset
    framerate=50.0,       # frames per second
    pattern=Nmer2D(n=6, d=0.2),  # hexamer with 200nm diameter
    molecule=GenericFluor(; q=[0 50; 1e-2 0]),  # rates in 1/s
    camera=IdealCamera(1:256, 1:256, 0.1)  # pixelsize in μm
)

## Setup drift model 
drift_true = DC.Polynomial(smld_noisy; degree=2, initialize="random",
                                       rscale=0.1)
println("drifted data ...")
smld_drift = DC.applydrift(smld_noisy, drift_true)

## Correct Drift (Kdtree cost function)
println("cost=Kdtree")
smld_correctedKd = DC.driftcorrect(smld_drift; verbose=1)

## Correct drift (Entropy cost function --- slow compared to Kdtree)
println("cost=Entropy")
smld_correctedE = DC.driftcorrect(smld_drift; cost_fun="Entropy", maxn=100,
                                              verbose=1)

## Correct drift (Entropy cost function --- slow compared to
## Kdtree + findshift2D (inter-datset pair correlation)
println("cost=Kdtree + findshift2D")
smld_correctedECC = DC.driftcorrect(smld_drift; cost_fun="Entropy", maxn=100,
                                    histbinsize=0.05, verbose=1)

println("cost=Kd/Entr + findshift2D")
smld_correctedKCC = DC.driftcorrect(smld_drift; cost_fun_intra="Kdtree",
    cost_fun_inter="Entropy", maxn=100, histbinsize=0.05, verbose=1)

smld_noisy_x = [e.x for e in smld_noisy.emitters]
smld_noisy_y = [e.y for e in smld_noisy.emitters]
smld_drift_x = [e.x for e in smld_drift.emitters]
smld_drift_y = [e.y for e in smld_drift.emitters]
smld_correctedKd_x = [e.x for e in smld_correctedKd.emitters]
smld_correctedKd_y = [e.y for e in smld_correctedKd.emitters]
smld_correctedE_x = [e.x for e in smld_correctedE.emitters]
smld_correctedE_y = [e.y for e in smld_correctedE.emitters]
smld_correctedECC_x = [e.x for e in smld_correctedECC.emitters]
smld_correctedECC_y = [e.y for e in smld_correctedECC.emitters]
smld_correctedKCC_x = [e.x for e in smld_correctedKCC.emitters]
smld_correctedKCC_y = [e.y for e in smld_correctedKCC.emitters]


f = Figure()
ax1 = Axis(f[1, 1], aspect=DataAspect(), title="original noisy data")
scatter!(smld_noisy_x, smld_noisy_y; markersize=5)
ax2 = Axis(f[1, 2], aspect=DataAspect(), title="drifted data")
scatter!(smld_drift_x, smld_drift_y; markersize=5)
ax3 = Axis(f[1, 3], aspect=DataAspect(), title="cost=Kdtree + findshift2D")
scatter!(smld_correctedKCC_x, smld_correctedKCC_y; markersize=5)
ax4 = Axis(f[2, 1], aspect=DataAspect(), title="cost=Kdtree")
scatter!(smld_correctedKd_x, smld_correctedKd_y; markersize=5)
ax5 = Axis(f[2, 2], aspect=DataAspect(), title="cost=Entropy")
scatter!(smld_correctedE_x, smld_correctedE_y; markersize=5)
ax6 = Axis(f[2, 3], aspect=DataAspect(), title="cost=Kd/Entr + findshift2D")
scatter!(smld_correctedECC_x, smld_correctedECC_y; markersize=5)
linkxaxes!(ax1, ax2)
linkxaxes!(ax1, ax3)
linkxaxes!(ax1, ax4)
linkxaxes!(ax1, ax5)
linkxaxes!(ax1, ax6)
linkyaxes!(ax1, ax2)
linkyaxes!(ax1, ax3)
linkyaxes!(ax1, ax4)
linkyaxes!(ax1, ax5)
linkyaxes!(ax1, ax6)
display(f)
