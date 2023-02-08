## Demonstrate applying and correctly drift 

using Revise
using SMLMDriftCorrection
DC = SMLMDriftCorrection
using SMLMData
using SMLMSim
using PlotlyJS
using Statistics

# make an n-mer dataset
γ = 1e5 # Fluorophore emission rate
q = [0 50
   5e-2 0] # Fluorophore blinking rates
n = 6 # Nmer rank
d = 0.05 # Nmer diameter
ρ = 0.5 # density of Nmers 
xsize = 25.6 # image size
ysize = 25.6
nframes = 2000 # number of frames
framerate = 50.0 # framerate
σ_psf = .13 # psf sigma used for uncertainty calcs
minphotons = 500 # minimum number of photons per frame accepted

# Simulation sequence
f = SMLMSim.GenericFluor(γ, q)
pattern = SMLMSim.Nmer2D(n, d)
smd_true = SMLMSim.uniform2D(ρ, pattern, xsize, ysize)
smd_model = SMLMSim.kineticmodel(smd_true, f, nframes, framerate; ndatasets = 10, minphotons = minphotons)
smd_noisy = SMLMSim.noise(smd_model, σ_psf)

## Setup drift model 
drift_true = DC.Polynomial(smd_noisy; degree = 2, 
               initialize = "random",rscale=0.1)
smd_drift = DC.applydrift(smd_noisy, drift_true)

## Correct Drift
smld_corrected=DC.driftcorrect(smd_drift)

plt=PlotlyJS.plot(scattergl(x=smd_noisy.x, y=smd_noisy.y, mode="markers"))
display(plt)

plt=PlotlyJS.plot(scattergl(x=smd_drift.x, y=smd_drift.y, mode="markers"))
display(plt)

plt=PlotlyJS.plot(scattergl(x=smld_corrected.x, y=smld_corrected.y, mode="markers"))
display(plt)

