using SMLMDriftCorrection
DC = SMLMDriftCorrection
using SMLMSim
using Test

@testset "SMLMDriftCorrection.jl" begin
    # Write your tests here.

    # make an Nmer dataset
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
    pattern=SMLMSim.Nmer2D(; n,d)
    smd_true=SMLMSim.uniform2D(ρ,pattern,xsize,ysize)
    smd_model=SMLMSim.kineticmodel(smd_true,f,nframes,framerate;ndatasets=10,minphotons=minphotons)
    smd_noisy=SMLMSim.noise(smd_model,σ_psf)
    ## Set up drift model
    N = length(smd_noisy.x)
    driftmodel=DC.Polynomial(smd_noisy; degree=2, initialize="random")
    smd_drift=DC.applydrift(smd_noisy,driftmodel)
    smd_DC=DC.correctdrift(smd_drift,driftmodel)
    rmsd = sqrt(sum((smd_DC.x .- smd_noisy.x).^2 .+ (smd_DC.y .- smd_noisy.y).^2) ./ N)
    @test isapprox(rmsd, 0.0; atol=1e-10)

    smd_DC = DC.driftcorrect(smd_drift)
    rmsd = sqrt(sum((smd_DC.x .- smd_noisy.x).^2 .+ (smd_DC.y .- smd_noisy.y).^2) ./ N)
    print("rmsd (K-d tree) = $rmsd\n")
    @test isapprox(rmsd, 0.0; atol = 1.0)

    smd_DC = DC.driftcorrect(smd_drift; histbinsize=0.25)
    rmsd = sqrt(sum((smd_DC.x .- smd_noisy.x).^2 .+ (smd_DC.y .- smd_noisy.y).^2) ./ N)
    print("rmsd (PairCorr) = $rmsd\n")
    @test isapprox(rmsd, 0.0; atol = 1.0)

smld_true, smld_model, smld_noisy = SMLMSim.sim(;
    ρ=1.0,
    σ_PSF=0.13, #micron 
    minphotons=50,
    ndatasets=10,
    nframes=1000,
    framerate=50.0, # 1/s
    pattern=SMLMSim.Nmer2D(),
    molecule=SMLMSim.GenericFluor(; q=[0 50; 1e-2 0]), #1/s 
    camera=SMLMSim.IdealCamera(; xpixels=256, ypixels=256, pixelsize=0.1) #pixelsize is microns
)

println("N = $(size(smld_noisy.x, 1))")
drift = DC.findshift2D(smld_noisy, smld_noisy; histbinsize=0.25)
@test all(drift .≈ [0.0, 0.0])

smldn = deepcopy(smld_noisy)
smldn.x .+= 4.3
smldn.y .+= -2.8
smldn.x .= max.(0, min.(smldn.x, 256))
smldn.y .= max.(0, min.(smldn.y, 256))
drift = DC.findshift2D(smld_noisy, smldn; histbinsize=0.25)
@test all(drift .≈ [-4.25, 2.75])
end