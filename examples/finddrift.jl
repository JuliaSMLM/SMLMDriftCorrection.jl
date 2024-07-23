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
γ = 1e5 # Fluorophore emission rate
q = [0 50
   5e-2 0] # Fluorophore blinking rates
n = 6 # Nmer rank
d = 0.05 # Nmer diameter
#ρ = 0.5 # density of Nmers 
ρ = 0.05 # density of Nmers # make smaller for entropy cost function
xsize = 25.6 # image size
ysize = 25.6
nframes = 2000 # number of frames
framerate = 50.0 # framerate
σ_psf = .13 # psf sigma used for uncertainty calcs
minphotons = 500 # minimum number of photons per frame accepted

# Simulation sequence
f = SMLMSim.GenericFluor(γ, q)
pattern = SMLMSim.Nmer2D(; n, d)
smd_true = SMLMSim.uniform2D(ρ, pattern, xsize, ysize)
smd_model = SMLMSim.kineticmodel(smd_true, f, nframes, framerate; ndatasets = 10, minphotons = minphotons)
smd_noisy = SMLMSim.noise(smd_model, σ_psf)

## Setup drift model 
drift_true = DC.Polynomial(smd_noisy; degree = 2, 
               initialize = "random", rscale=0.1)
smd_drift = DC.applydrift(smd_noisy, drift_true)

## Correct Drift (Kdtree cost function)
smld_correctedKd = DC.driftcorrect(smd_drift; verbose = 1)

#plt1=PlotlyJS.plot(scattergl(x=smd_noisy.x, y=smd_noisy.y, mode="markers"), Layout(title="original"))
#display(plt1)

#plt2=PlotlyJS.plot(scattergl(x=smd_drift.x, y=smd_drift.y, mode="markers"), Layout(title="drifted"))
#display(plt2)

#plt3=PlotlyJS.plot(scattergl(x=smld_correctedKd.x, y=smld_correctedKd.y, mode="markers"), Layout(title="cost=Kdtree"))
#display(plt3)

## Correct drift (Entropy cost function --- slow compared to Kdtree)
smld_correctedE = DC.driftcorrect(smd_drift; cost_fun="Entropy", maxn=100, verbose=1)

#plt4=PlotlyJS.plot(scattergl(x=smld_correctedE.x, y=smld_correctedE.y, mode="markers"), Layout(title="cost=Entropy"))
#display(plt4)

## Correct drift (Entropy cost function --- slow compared to Kdtree + findshift2D (inter-datset pair correlation)
smld_correctedECC = DC.driftcorrect(smd_drift; cost_fun="Entropy", maxn=100, histbinsize=0.05, verbose=1)

#plt5=PlotlyJS.plot(scattergl(x=smld_correctedECC.x, y=smld_correctedECC.y, mode="markers"), Layout(title="cost=Entropy + findshift2D"))
#display(plt5)

smld_correctedKCC = DC.driftcorrect(smd_drift; cost_fun_intra="Kdtree", cost_fun_inter="Entropy", maxn=100, histbinsize=0.05, verbose=1)

f = Figure()
ax1 = Axis(f[1, 1], aspect=DataAspect(), title="original")
scatter!(smd_noisy.x, smd_noisy.y; markersize=5)
ax2 = Axis(f[1, 2], aspect=DataAspect(), title="drifted")
scatter!(smd_drift.x, smd_drift.y; markersize=5)
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