## Demonstrate applying and correctly drift 

using Revise
using SMLMDriftCorrection
DC = SMLMDriftCorrection
using SMLMData
using SMLMSim
using Plots
using PlotlyJS
using Statistics

# make an n-mer dataset
γ = 1e5 # Fluorophore emission rate
q = [0 50
   1e-2 0] # Fluorophore blinking rates
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
smd_model = SMLMSim.kineticmodel(smd_true, f, nframes, framerate; ndatasets = 100, minphotons = minphotons)
smd_noisy = SMLMSim.noise(smd_model, σ_psf)

## Setup drift model 
drift_true = DC.Polynomial(smd_noisy; degree = 2, 
               initialize = "random",rscale=0.1)
smld_drift = DC.applydrift(smd_noisy, drift_true)

#correct intra
Threads.@threads for nn = 1:smld_drift.ndatasets
    DC.findintra!(driftmodel.intra[nn], smld_drift, nn, d_cutoff)
end

plt=PlotlyJS.plot(scattergl(x=smld_drift.x, y=smld_drift.y, mode="markers"))
display(plt)

## Cost matrix, drift matrix 
d_cutoff=.1
m=8 #distance off diagonal
driftmodel = DC.Polynomial(smld_drift; degree = 2)
nd=smld_drift.ndatasets
costmatrix=zeros(nd,nd)
shiftmatrix=Array{Vector{Float64}}(undef,nd,nd) #[x,y]
for ii=1:nd,jj=ii+1:(min(nd,ii+m))
    println(ii," ",jj)
    costmatrix[ii,jj]=DC.findinter!(driftmodel, smld_drift, ii, jj, d_cutoff)   
    shiftmatrix[ii,jj]=driftmodel.inter[ii].dm
end

plotly()
Plots.heatmap(costmatrix)


