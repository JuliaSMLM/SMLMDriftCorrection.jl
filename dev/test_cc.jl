using SMLMDriftCorrection
DC = SMLMDriftCorrection
using SMLMSim

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

## Set up drift model
driftmodel = DC.Polynomial(smld_noisy; degree=2, initialize="random")
# Apply drift to the noisy dataset using the drift model
smld_drift = DC.applydrift(smld_noisy, driftmodel)
# Apply drift correction [correctdrift] to the drifted dataset using the drift model
smld_DC = DC.correctdrift(smld_drift, driftmodel)
N = length(smld_noisy.x)
println("N = $N")
rmsd = sqrt(sum((smld_DC.x .- smld_noisy.x) .^ 2 .+ (smld_DC.y .- smld_noisy.y) .^ 2) ./ N)

# Apply drift to the noisy dataset using the drift model
smld_drift = DC.applydrift(smld_noisy, driftmodel)
# Apply drift correction [driftcorrect (Kdtree)] to the drifted dataset
smld_DC = DC.driftcorrect(smld_drift; cost_fun = "Kdtree")
rmsd1 = sqrt(sum((smld_DC.x .- smld_noisy.x) .^ 2 .+ (smld_DC.y .- smld_noisy.y) .^ 2) ./ N)

# Apply drift to the noisy dataset using the drift model
smld_drift = DC.applydrift(smld_noisy, driftmodel)
# Apply drift correction [driftcorrect (Kdtree) + findshift2] to the drifted dataset
smld_DC = DC.driftcorrect(smld_drift; cost_fun = "Kdtree", histbinsize=0.05)
rmsd2 = sqrt(sum((smld_DC.x .- smld_noisy.x) .^ 2 .+ (smld_DC.y .- smld_noisy.y) .^ 2) ./ N)

# Apply drift to the noisy dataset using the drift model
smld_drift = DC.applydrift(smld_noisy, driftmodel)
# Apply drift correction [driftcorrect + findshift2] to the drifted dataset
smld_DC = DC.driftcorrect(smld_drift; cost_fun_inter="None", histbinsize=0.05)
rmsd3 = sqrt(sum((smld_DC.x .- smld_noisy.x) .^ 2 .+ (smld_DC.y .- smld_noisy.y) .^ 2) ./ N)

# Apply drift to the noisy dataset using the drift model
println("Entropy")
smld_drift = DC.applydrift(smld_noisy, driftmodel)
# Apply drift correction [driftcorrect (Entropy)] to the drifted dataset
smld_DC = DC.driftcorrect(smld_drift; cost_fun="Entropy", maxn=100)
rmsd4 = sqrt(sum((smld_DC.x .- smld_noisy.x) .^ 2 .+ (smld_DC.y .- smld_noisy.y) .^ 2) ./ N)

# Apply drift to the noisy dataset using the drift model
println("Entropy + findshift2D")
smld_drift = DC.applydrift(smld_noisy, driftmodel)
# Apply drift correction [driftcorrect (Entropy) + findshift2] to the drifted dataset
smld_DC = DC.driftcorrect(smld_drift; cost_fun="Entropy", maxn=100, histbinsize=0.05)
rmsd5 = sqrt(sum((smld_DC.x .- smld_noisy.x) .^ 2 .+ (smld_DC.y .- smld_noisy.y) .^ 2) ./ N)

println("correctdrift rmsd                                               = $rmsd")
println("driftcorrect intra = Kdtree,  inter = Kdtree rmsd               = $rmsd1")
println("driftcorrect intra = Kdtree,  inter = Kdtree + findshift2 rmsd  = $rmsd2")
println("driftcorrect intra = Kdtree,  inter = findshift2 rmsd           = $rmsd3")
println("driftcorrect intra = Entropy, inter = Entropy rmsd              = $rmsd4")
println("driftcorrect intra = Entropy, inter = Entropy + findshift2 rmsd = $rmsd5")
#isapprox(rmsd, 0.0; atol=1e-10)
