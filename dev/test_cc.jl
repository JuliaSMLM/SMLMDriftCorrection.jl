using SMLMDriftCorrection
DC = SMLMDriftCorrection
using SMLMSim

# make an Nmer dataset
γ = 1e5 # Fluorophore emission rate
q = [0 50
    1e-2 0] # Fluorophore blinking rates
n = 6 # Nmer rank
d = 0.1 # Nmer diameter
ρ = 0.1 # density of Nmers
xsize = 25.6 # image size
ysize = 25.6
nframes = 2000 # number of frames
framerate = 50.0 # framerate
σ_psf = 1.3 # psf sigma used for uncertainty calcs
minphotons = 500 # minimum number of photons per frame accepted
# Simulation sequence
f = SMLMSim.GenericFluor(γ, q)
pattern = SMLMSim.Nmer2D(; n, d)
smd_true = SMLMSim.uniform2D(ρ, pattern, xsize, ysize)
smd_model = SMLMSim.kineticmodel(smd_true, f, nframes, framerate; ndatasets=10, minphotons=minphotons)
smd_noisy = SMLMSim.noise(smd_model, σ_psf)
## Set up drift model
driftmodel = DC.Polynomial(smd_noisy; degree=2, initialize="random")
# Apply drift to the noisy dataset using the drift model
smd_drift = DC.applydrift(smd_noisy, driftmodel)
# Apply drift correction [correctdrift] to the drifted dataset using the drift model
smd_DC = DC.correctdrift(smd_drift, driftmodel)
N = length(smd_noisy.x)
rmsd = sqrt(sum((smd_DC.x .- smd_noisy.x) .^ 2 .+ (smd_DC.y .- smd_noisy.y) .^ 2) ./ N)

# Apply drift to the noisy dataset using the drift model
smd_drift = DC.applydrift(smd_noisy, driftmodel)
# Apply drift correction [driftcorrect (Kdtree)] to the drifted dataset
smd_DC = DC.driftcorrect(smd_drift; cost_fun = "Kdtree")
rmsd1 = sqrt(sum((smd_DC.x .- smd_noisy.x) .^ 2 .+ (smd_DC.y .- smd_noisy.y) .^ 2) ./ N)

# Apply drift to the noisy dataset using the drift model
smd_drift = DC.applydrift(smd_noisy, driftmodel)
# Apply drift correction [driftcorrect (Kdtree) + findshift2] to the drifted dataset
smd_DC = DC.driftcorrect(smd_drift; cost_fun = "Kdtree", histbinsize=0.05)
rmsd2 = sqrt(sum((smd_DC.x .- smd_noisy.x) .^ 2 .+ (smd_DC.y .- smd_noisy.y) .^ 2) ./ N)

# Apply drift to the noisy dataset using the drift model
smd_drift = DC.applydrift(smd_noisy, driftmodel)
# Apply drift correction [driftcorrect + findshift2] to the drifted dataset
smd_DC = DC.driftcorrect(smd_drift; cost_fun_inter="None", histbinsize=0.05)
rmsd3 = sqrt(sum((smd_DC.x .- smd_noisy.x) .^ 2 .+ (smd_DC.y .- smd_noisy.y) .^ 2) ./ N)

# Apply drift to the noisy dataset using the drift model
println("Entropy")
smd_drift = DC.applydrift(smd_noisy, driftmodel)
# Apply drift correction [driftcorrect (Entropy)] to the drifted dataset
smd_DC = DC.driftcorrect(smd_drift; cost_fun="Entropy", maxn=100)
rmsd4 = sqrt(sum((smd_DC.x .- smd_noisy.x) .^ 2 .+ (smd_DC.y .- smd_noisy.y) .^ 2) ./ N)

# Apply drift to the noisy dataset using the drift model
println("Entropy + findshift2D")
smd_drift = DC.applydrift(smd_noisy, driftmodel)
# Apply drift correction [driftcorrect (Entropy) + findshift2] to the drifted dataset
smd_DC = DC.driftcorrect(smd_drift; cost_fun="Entropy", maxn=100, histbinsize=0.05)
rmsd5 = sqrt(sum((smd_DC.x .- smd_noisy.x) .^ 2 .+ (smd_DC.y .- smd_noisy.y) .^ 2) ./ N)

println("correctdrift rmsd                                               = $rmsd")
println("driftcorrect intra = Kdtree,  inter = Kdtree rmsd               = $rmsd1")
println("driftcorrect intra = Kdtree,  inter = Kdtree + findshift2 rmsd  = $rmsd2")
println("driftcorrect intra = Kdtree,  inter = findshift2 rmsd           = $rmsd3")
println("driftcorrect intra = Entropy, inter = Entropy rmsd              = $rmsd4")
println("driftcorrect intra = Entropy, inter = Rntropy + findshift2 rmsd = $rmsd5")
#isapprox(rmsd, 0.0; atol=1e-10)
