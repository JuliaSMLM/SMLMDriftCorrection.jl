## Demonstrate applying and correcting drift 

using Revise
using SMLMDriftCorrection
DC = SMLMDriftCorrection
using SMLMData
using SMLMSim
using GLMakie
#using PlotlyJS
using Statistics

# make an Nmer dataset
# Simulation parameters use physical units
# smld structures are in units of pixels and frames
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
drift_true = DC.Polynomial(smld_noisy; degree = 2, 
               initialize = "random", rscale=0.1)
smld_drift = DC.applydrift(smld_noisy, drift_true)

## Correct Drift (Kdtree cost function)
smld_correctedKd = DC.driftcorrect(smld_drift; verbose = 1)

#plt1=PlotlyJS.plot(scattergl(x=smld_noisy.x, y=smld_noisy.y, mode="markers"), Layout(title="original"))
#display(plt1)

#plt2=PlotlyJS.plot(scattergl(x=smld_drift.x, y=smld_drift.y, mode="markers"), Layout(title="drifted"))
#display(plt2)

#plt3=PlotlyJS.plot(scattergl(x=smld_correctedKd.x, y=smld_correctedKd.y, mode="markers"), Layout(title="cost=Kdtree"))
#display(plt3)

## Correct drift (Entropy cost function --- slow compared to Kdtree)
smld_correctedE = DC.driftcorrect(smld_drift; cost_fun="Entropy", maxn=100, verbose=1)

#plt4=PlotlyJS.plot(scattergl(x=smld_correctedE.x, y=smld_correctedE.y, mode="markers"), Layout(title="cost=Entropy"))
#display(plt4)

## Correct drift (Entropy cost function --- slow compared to Kdtree + findshift2D (inter-datset pair correlation)
smld_correctedECC = DC.driftcorrect(smld_drift; cost_fun="Entropy", maxn=100, histbinsize=0.05, verbose=1)

#plt5=PlotlyJS.plot(scattergl(x=smld_correctedECC.x, y=smld_correctedECC.y, mode="markers"), Layout(title="cost=Entropy + findshift2D"))
#display(plt5)

smld_correctedKCC = DC.driftcorrect(smld_drift; cost_fun_intra="Kdtree", cost_fun_inter="Entropy", maxn=100, histbinsize=0.05, verbose=1)

f = Figure()
ax1 = Axis(f[1, 1], aspect=DataAspect(), title="original")
scatter!(smld_noisy.x, smld_noisy.y; markersize=5)
ax2 = Axis(f[1, 2], aspect=DataAspect(), title="drifted")
scatter!(smld_drift.x, smld_drift.y; markersize=5)
ax3 = Axis(f[1, 3], aspect=DataAspect(), title="cost=Kdtree + findshift2D")
scatter!(smld_correctedKCC.x, smld_correctedKCC.y; markersize=5)
ax4 = Axis(f[2, 1], aspect=DataAspect(), title="cost=Kdtree")
scatter!(smld_correctedKd.x, smld_correctedKd.y; markersize=5)
ax5 = Axis(f[2, 2], aspect=DataAspect(), title="cost=Entropy")
scatter!(smld_correctedE.x, smld_correctedE.y; markersize=5)
ax6 = Axis(f[2, 3], aspect=DataAspect(), title="cost=Entropy + findshift2D")
scatter!(smld_correctedECC.x, smld_correctedECC.y; markersize=5)
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
