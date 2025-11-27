"""
    test_basic.jl

Basic functionality test for SMLMDriftCorrection with current SMLMSim API.
Tests defaults with a simple simulated dataset.
"""

using Pkg
Pkg.activate(".")

using SMLMDriftCorrection
using SMLMSim
using SMLMData

println("="^60)
println("Testing SMLMDriftCorrection Basic Functionality")
println("="^60)

# Setup camera
println("\n[1/4] Setting up camera...")
camera = IdealCamera(128, 128, 0.1)  # 128×128 pixels, 100nm pixel size
println("  Camera: 128×128 pixels, 100nm pixel size")

# Setup simulation parameters
println("\n[2/4] Setting up simulation...")
sim_params = StaticSMLMParams(
    density = 1.5,          # 1.5 patterns per μm²
    σ_psf = 0.13,           # 130nm PSF width
    nframes = 1000,         # 1000 frames per dataset
    framerate = 50.0,       # 50 fps
    ndatasets = 5,          # 5 datasets for inter-dataset correction
    ndims = 2               # 2D simulation
)

pattern = Nmer2D(n=6, d=0.2)  # Hexamer, 200nm diameter

fluor = GenericFluor(
    photons = 5000.0,       # 5k photons/sec
    k_off = 50.0,           # Switch off at 50 Hz
    k_on = 0.001            # Low on-rate for sparse labeling
)

println("  Pattern: 6-mer (200nm diameter)")
println("  Datasets: $(sim_params.ndatasets)")
println("  Frames/dataset: $(sim_params.nframes)")

# Simulate
println("\n[3/4] Running simulation...")
smld_true, smld_model, smld_noisy = simulate(
    sim_params;
    pattern = pattern,
    molecule = fluor,
    camera = camera
)

println("  Total localizations: $(length(smld_noisy.emitters))")
println("  Datasets: $(smld_noisy.n_datasets)")
println("  Frames: $(smld_noisy.n_frames)")

# Test drift correction with defaults
println("\n[4/4] Running drift correction (defaults)...")
println("  Cost function: Kdtree (default)")
println("  Degree: 2 (default)")
println("  d_cutoff: 0.01 μm (default)")

try
    smld_corrected = driftcorrect(smld_noisy; verbose=1)
    println("\n✓ SUCCESS: Drift correction completed!")
    println("  Output localizations: $(length(smld_corrected.emitters))")

    # Quick sanity check
    if length(smld_corrected.emitters) == length(smld_noisy.emitters)
        println("  ✓ Localization count preserved")
    else
        println("  ⚠ WARNING: Localization count changed!")
    end

catch e
    println("\n✗ FAILED: Drift correction encountered an error")
    println("\nError details:")
    showerror(stdout, e, catch_backtrace())
    println("\n")
end

println("\n" * "="^60)
println("Test complete")
println("="^60)
