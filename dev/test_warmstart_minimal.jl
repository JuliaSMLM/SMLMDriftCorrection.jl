# Minimal warmstart convergence test - no SMLMRender dependency
using Pkg
Pkg.activate(temp=true)
Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.add(["SMLMSim", "SMLMData", "Statistics", "Random", "Printf"])

using SMLMDriftCorrection
using SMLMSim
using SMLMData
using Statistics
using Random
using Printf

const DC = SMLMDriftCorrection

function generate_simple_smld(; n_datasets=5, n_frames=5000, density=5.0, seed=42)
    Random.seed!(seed)
    pattern = Nmer2D(n=6, d=0.2)
    fov_size = sqrt(50.0)  # ~50 nmers
    n_emitters = round(Int, density * fov_size^2 / 6)

    smlds = StaticSMLD[]
    for ds in 1:n_datasets
        smld = simulate(SMLMSimParams(
            density=density, σ_psf=0.130, minphotons=200,
            ndatasets=1, nframes=n_frames,
            framerate=20.0, ndims=2, zoom=20,
            pattern=pattern, molecule=GenericFluor(),
            camera=IdealCamera(1:128, 1:128, 0.1)
        )).smld_noisy
        push!(smlds, smld)
    end

    # Combine into one multi-dataset SMLD
    all_emitters = []
    for (ds, smld) in enumerate(smlds)
        for e in smld.emitters
            push!(all_emitters, typeof(e)(e.x, e.y, e.σ_x, e.σ_y, e.frame, ds, e.track_id, e.photons, e.bg))
        end
    end
    return BasicSMLD(all_emitters, smlds[1].camera, n_frames, n_datasets)
end

function compute_rmsd(smld_a, smld_b)
    n = min(length(smld_a.emitters), length(smld_b.emitters))
    sse = 0.0
    for i in 1:n
        ea, eb = smld_a.emitters[i], smld_b.emitters[i]
        sse += (ea.x - eb.x)^2 + (ea.y - eb.y)^2
    end
    return sqrt(sse / n)
end

function max_drift_change(model_a::DC.LegendrePolynomial, model_b::DC.LegendrePolynomial)
    n_datasets = model_a.ndatasets
    n_dims = model_a.intra[1].ndims
    n_frames = model_a.intra[1].dm[1].n_frames

    max_intra = 0.0
    max_inter = 0.0

    for nn in 1:n_datasets
        for d in 1:n_dims
            max_inter = max(max_inter, abs(model_a.inter[nn].dm[d] - model_b.inter[nn].dm[d]))
        end
        for frame in 1:max(1, n_frames÷10):n_frames
            for d in 1:n_dims
                a = DC.evaluate_at_frame(model_a.intra[nn].dm[d], frame)
                b = DC.evaluate_at_frame(model_b.intra[nn].dm[d], frame)
                max_intra = max(max_intra, abs(a - b))
            end
        end
    end

    return (intra=max_intra, inter=max_inter)
end

function main()
    println("="^70)
    println("WARMSTART CONVERGENCE TEST")
    println("="^70)

    # Use SMLMSim's full pipeline directly
    println("Generating test data...")
    params_2d = StaticSMLMConfig(
        10.0,     # density: emitters per μm²
        0.13,     # σ_psf
        30,       # minphotons
        5,        # ndatasets
        5000,     # nframes per dataset
        50.0,     # framerate
        2,        # ndims
        [0.0, 1.0]
    )
    (smld_orig, _) = simulate(
        params_2d;
        pattern=Nmer2D(n=6, d=0.2),
        molecule=GenericFluor(; photons=5000.0, k_on=0.02, k_off=50.0),
        camera=IdealCamera(1:64, 1:64, 0.1)
    )
    println("  $(length(smld_orig.emitters)) emitters, $(smld_orig.n_datasets) datasets")

    # Apply known drift
    Random.seed!(42)
    drift_model = DC.LegendrePolynomial(smld_orig; degree=3, initialize="random", rscale=0.1)
    smld_drifted = DC.applydrift(smld_orig, drift_model)

    # PASS 1
    println("\n--- PASS 1 (cold) ---")
    t1 = @elapsed (smld_p1, info1) = driftcorrect(smld_drifted;
        degree=3, quality=:iterative, dataset_mode=:registered,
        shift_scale=0.05, max_iterations=10, verbose=0)
    rmsd1 = compute_rmsd(smld_orig, smld_p1)
    @printf "Time: %.1fs  converged=%s  iter=%d  RMSD=%.2fnm\n" t1 info1.converged info1.iterations rmsd1*1000

    # PASS 2 (warm-started)
    println("\n--- PASS 2 (warm-start from pass 1) ---")
    t2 = @elapsed (smld_p2, info2) = driftcorrect(smld_drifted;
        degree=3, quality=:iterative, dataset_mode=:registered,
        shift_scale=0.05, max_iterations=10,
        warm_start=info1.model, verbose=0)
    rmsd2 = compute_rmsd(smld_orig, smld_p2)
    @printf "Time: %.1fs  converged=%s  iter=%d  RMSD=%.2fnm\n" t2 info2.converged info2.iterations rmsd2*1000

    diffs = max_drift_change(info1.model, info2.model)
    println("\n--- DIFFERENCE ---")
    @printf "Max intra-trajectory diff: %.2f nm\n" diffs.intra*1000
    @printf "Max inter-shift diff:      %.2f nm\n" diffs.inter*1000
    @printf "RMSD pass1 → pass2:        %.2f nm → %.2f nm (Δ %+.2f)\n" rmsd1*1000 rmsd2*1000 (rmsd2-rmsd1)*1000

    println("\n--- VERDICT ---")
    if max(diffs.intra, diffs.inter) < 0.005
        println("CONVERGED: warm-start finds <5nm additional correction")
    elseif max(diffs.intra, diffs.inter) < 0.020
        println("MARGINAL: warm-start finds 5-20nm additional correction")
    else
        println("NOT CONVERGED: warm-start finds >20nm additional correction")
        println("  → first-pass result was a local minimum")
    end
end

main()
