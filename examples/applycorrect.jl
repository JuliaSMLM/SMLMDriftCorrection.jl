## Demonstrate applying and correcting drift 

using Revise
using SMLMDriftCorrection
DC=SMLMDriftCorrection
using SMLMData
using SMLMSim
using PlotlyJS

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

plt=PlotlyJS.plot(scattergl(x=smld_noisy.x, y=smld_noisy.y, mode="markers"))
display(plt)

## Set up drift model 
driftmodel=DC.Polynomial(smld_noisy; degree=2, initialize="random")
smld_drift=DC.applydrift(smld_noisy,driftmodel)

plt=PlotlyJS.plot(scattergl(x=smld_drift.x, y=smld_drift.y, mode="markers"))
display(plt)

smld_DC=DC.correctdrift(smld_drift, driftmodel)

plt=PlotlyJS.plot(scattergl(x=smld_DC.x, y=smld_DC.y, mode="markers"))
display(plt)


# cost=DC.NND(smld_drift)
# cost=DC.NND(smld_DC)


# plt=PlotlyJS.plot(scattergl(x=smld_DC.x, y=smld_DC.y, mode="markers"))
# display(plt)

##
# dm_found=DC.finddrift(smld_drift)
