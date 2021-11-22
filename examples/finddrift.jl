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
d = 0.2 # Nmer diameter
ρ = 0.1 # density of Nmers 
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
drift_true = DC.Polynomial(smd_noisy; degree = 1, 
               initialize = "random",rscale=0.1)
smd_drift = DC.applydrift(smd_noisy, drift_true)

## Intra drift 

driftmodel = DC.Polynomial(smd_noisy;degree=1)
d_cutoff = 3*mean([mean(smd_drift.σ_x), mean(smd_drift.σ_y)])
# d_cutoff=.1

# Intra dataset 
for nn=1:smd_drift.ndatasets
   DC.findintra!(driftmodel.intra[nn], smd_drift, nn, d_cutoff)
end

smd_intraonly = DC.correctdrift(smd_drift, driftmodel)

# Correct them all to datatset 1
for nn=2:smd_drift.ndatasets
   refdataset=1
   DC.findinter!(driftmodel, smd_drift, nn, refdataset,  d_cutoff)
end

# Correct them all to all others
for ii=1:2, nn=1:smd_drift.ndatasets
   # DC.findinter!(driftmodel, smd_drift, nn, d_cutoff)
end


smd_found = DC.correctdrift(smd_drift, driftmodel)

drift_true.inter[2]
driftmodel.inter[2]

drift_true.intra[3]
driftmodel.intra[3]


plt=PlotlyJS.plot(scattergl(x=smd_noisy.x, y=smd_noisy.y, mode="markers"))
display(plt)

plt=PlotlyJS.plot(scattergl(x=smd_drift.x, y=smd_drift.y, mode="markers"))
display(plt)

plt=PlotlyJS.plot(scattergl(x=smd_intraonly.x, y=smd_intraonly.y, mode="markers"))
display(plt)

plt=PlotlyJS.plot(scattergl(x=smd_found.x, y=smd_found.y, mode="markers"))
display(plt)

