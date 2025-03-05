using SMLMDriftCorrection
DC = SMLMDriftCorrection
using SMLMSim
using Test

@testset "SMLMDriftCorrection.jl" begin
    # Write your tests here.

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
    N = length(smld_noisy.emitters)
    driftmodel=DC.Polynomial(smld_noisy; degree=2, initialize="random")
    smld_drift=DC.applydrift(smld_noisy,driftmodel)
    smld_DC=DC.correctdrift(smld_drift,driftmodel)

    smld_noisy_x = [e.x for e in smld_noisy.emitters]
    smld_noisy_y = [e.y for e in smld_noisy.emitters]
    smld_DC_x = [e.x for e in smld_DC.emitters]
    smld_DC_y = [e.y for e in smld_DC.emitters]
    rmsd = sqrt(sum((smld_DC_x .- smld_noisy_x).^2 .+ (smld_DC_y .- smld_noisy_y).^2) ./ N)
    @test isapprox(rmsd, 0.0; atol=1e-10)

    smld_DC = DC.driftcorrect(smld_drift)
    rmsd = sqrt(sum((smld_DC.x .- smld_noisy.x).^2 .+ (smld_DC.y .- smld_noisy.y).^2) ./ N)
    print("rmsd (K-d tree) = $rmsd\n")
    @test isapprox(rmsd, 0.0; atol = 1.0)

    smld_DC = DC.driftcorrect(smld_drift; histbinsize=0.25)
    rmsd = sqrt(sum((smld_DC.x .- smld_noisy.x).^2 .+ (smld_DC.y .- smld_noisy.y).^2) ./ N)
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
