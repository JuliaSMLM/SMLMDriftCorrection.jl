## Demonstrate applying and correctly drift 

using Revise
using SMLMDriftCorrection
DC=SMLMDriftCorrection
using SMLMData
using SMLMSim
using PlotlyJS

# make an n-mer dataset
γ=1e5 # Fluorophore emission rate
q=[0 50
   1e-2 0] # Fluorophore blinking rates
n=6 # Nmer rank
d=.1 # Nmer diameter
ρ=0.1 # density of Nmers 
xsize=25.6 # image size
ysize=25.6
nframes=2000 # number of frames
framerate=50.0 # framerate
σ_psf=1.3 # psf sigma used for uncertainty calcs
minphotons=500 # minimum number of photons per frame accepted

# Simulation sequence
f=SMLMSim.GenericFluor(γ,q)
pattern=SMLMSim.Nmer2D(n,d)
smd_true=SMLMSim.uniform2D(ρ,pattern,xsize,ysize)
smd_model=SMLMSim.kineticmodel(smd_true,f,nframes,framerate;ndatasets=10,minphotons=minphotons)
smd_noisy=SMLMSim.noise(smd_model,σ_psf)
plt=PlotlyJS.plot(scattergl(x=smd_noisy.x, y=smd_noisy.y, mode="markers"))
display(plt)


# Setup drift model 
driftmodel=DC.Polynomial(smd_noisy; degree=2, initialize="random")
smd_drift=DC.applydrift(smd_noisy,driftmodel)

plt=PlotlyJS.plot(scattergl(x=smd_drift.x, y=smd_drift.y, mode="markers"))
display(plt)

smd_DC=DC.correctdrift(smd_drift,driftmodel)

plt=PlotlyJS.plot(scattergl(x=smd_DC.x, y=smd_DC.y, mode="markers"))
display(plt)


cost=DC.NND(smd_drift)
cost=DC.NND(smd_DC)


θ=DC.model2theta(driftmodel)
dm=DC.theta2model(θ,driftmodel)

plt=PlotlyJS.plot(scattergl(x=smd_DC.x, y=smd_DC.y, mode="markers"))
display(plt)





