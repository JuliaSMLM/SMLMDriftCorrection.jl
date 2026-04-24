# scenario_hexabody.jl - Real-data stress test using the RGY hexabody SMLD
#
# Loads the pre-driftcorrect SMLD from the paper-genmab-hexabody pipeline
# (03_frameconnect output, Cell_01_bagol/RGY condition) and runs the current
# SMLMDriftCorrection against it. Compares against the reference post-correct
# SMLD (04_driftcorrect/smld_corrected.jld2) the canonical pipeline produced.
#
# Dataset characteristics (from 03_frameconnect/info.toml, 04_driftcorrect/stats.md):
#   20 datasets × 5000 frames, ~147k FrameConnect tracks, σ_median ≈ 5 nm
#   Max intra drift ≈ 109 nm, max inter shift ≈ 104 nm
#   Reference run: iterative/degree=3/shift_scale=0.05, 410 s, 5 iterations
#
# Usage:
#   julia --project=. dev/scenario_hexabody.jl
#   julia> include("dev/scenario_hexabody.jl"); run_hexabody_diagnostics(; quality=:singlepass)

using Pkg
Pkg.activate(@__DIR__)

using SMLMDriftCorrection
using SMLMData
using JLD2
using Printf
using Statistics

const DC = SMLMDriftCorrection

# Default path into the hexabody pipeline. Overridable so the script stays
# portable if the tree gets moved.
const HEXABODY_BASE = "/home/kalidke/julia_shared_dev/papers/paper-genmab-hexabody/data/results/juliasmlm/2024-05-26_HeLa_SaturatingIgG10min+C1q/HeLa_IgG1-2F8-RGY-AF647_5ugml_10min+C1q/Cell_01_bagol"
const HEXABODY_INPUT = joinpath(HEXABODY_BASE, "03_frameconnect/smld_combined.jld2")
const HEXABODY_REFERENCE = joinpath(HEXABODY_BASE, "04_driftcorrect/smld_corrected.jld2")

"""
    load_hexabody(; input=HEXABODY_INPUT)

Load the pre-driftcorrect RGY hexabody SMLD. JLD2 key is `"smld"`.
"""
function load_hexabody(; input::AbstractString = HEXABODY_INPUT)
    isfile(input) || error("hexabody SMLD not found at $input")
    return JLD2.load(input, "smld")
end

"""
    summarize_smld(smld; label="smld")

Print a one-line summary of an SMLD: datasets, frames, locs, median σ, bbox.
"""
function summarize_smld(smld; label::AbstractString = "smld")
    N = length(smld.emitters)
    σs = Float64[e.σ_x for e in smld.emitters]
    append!(σs, Float64[e.σ_y for e in smld.emitters])
    xs = Float64[e.x for e in smld.emitters]
    ys = Float64[e.y for e in smld.emitters]
    @printf("  %-24s %7d locs | %2d datasets | %5d frames | σ_med = %5.1f nm | bbox [%.1f, %.1f] × [%.1f, %.1f] μm\n",
        label, N, smld.n_datasets, smld.n_frames, 1000 * median(σs),
        minimum(xs), maximum(xs), minimum(ys), maximum(ys))
end

"""
    run_hexabody_diagnostics(; kwargs...)

Run drift correction on the hexabody dataset and report metrics.

# Keyword Arguments
- `input::AbstractString=HEXABODY_INPUT`
- `reference::AbstractString=HEXABODY_REFERENCE` — post-correct SMLD to compare against
- `quality::Symbol=:iterative` — `:fft`, `:singlepass`, `:iterative`
- `degree::Int=3`
- `dataset_mode::Symbol=:registered`
- `maxn::Int=100`
- `max_iterations::Int=10`
- `convergence_tol::Float64=0.001`
- `shift_scale::Float64=0.05`
- `auto_roi::Bool=false`
- `verbose::Int=1`
- `compare_reference::Bool=true`

# Returns
NamedTuple with `smld_corrected`, `info`, `elapsed`, `rmsd_vs_reference`.
"""
function run_hexabody_diagnostics(;
        input::AbstractString = HEXABODY_INPUT,
        reference::AbstractString = HEXABODY_REFERENCE,
        quality::Symbol = :iterative,
        degree::Int = 3,
        dataset_mode::Symbol = :registered,
        maxn::Int = 100,
        max_iterations::Int = 10,
        convergence_tol::Float64 = 0.001,
        shift_scale::Float64 = 0.05,
        auto_roi::Bool = false,
        verbose::Int = 1,
        compare_reference::Bool = true)

    println("=" ^ 72)
    println("RGY HEXABODY DRIFT CORRECTION STRESS TEST")
    println("=" ^ 72)

    println("\n[1/4] Loading pre-driftcorrect SMLD")
    @printf("  path: %s\n", input)
    t_load = @elapsed smld = load_hexabody(; input=input)
    @printf("  loaded in %.2f s\n", t_load)
    summarize_smld(smld; label="input")

    println("\n[2/4] Running driftcorrect(quality=$(quality), degree=$(degree))")
    config = DC.DriftConfig(;
        quality = quality,
        degree = degree,
        dataset_mode = dataset_mode,
        maxn = maxn,
        max_iterations = max_iterations,
        convergence_tol = convergence_tol,
        shift_scale = shift_scale,
        auto_roi = auto_roi,
        verbose = verbose,
    )
    t_correct = @elapsed (smld_corrected, info) = DC.driftcorrect(smld, config)
    @printf("  elapsed: %.2f s   iterations: %d   converged: %s\n",
        t_correct, info.iterations, info.converged)
    @printf("  final entropy: %.1f\n", info.entropy)
    summarize_smld(smld_corrected; label="corrected")

    println("\n[3/4] Inter-dataset shifts (|shift| nm)")
    for ds in 1:smld_corrected.n_datasets
        mag = 1000 * sqrt(sum(info.model.inter[ds].dm .^ 2))
        @printf("  DS%02d: %5.1f nm\n", ds, mag)
    end

    println("\n  Residual correlation (lower is better, 0.01 is reference level):")
    rc = info.residual_correlation
    @printf("    intra mean |corr_x| = %.4f   |corr_y| = %.4f\n",
        rc.intra_summary.mean_abs_corr_x, rc.intra_summary.mean_abs_corr_y)
    @printf("    inter       corr_x  = %+.4f   corr_y  = %+.4f\n",
        rc.inter.corr_x, rc.inter.corr_y)

    rmsd_vs_ref = NaN
    if compare_reference && isfile(reference)
        println("\n[4/4] Comparing against reference post-correct SMLD")
        smld_ref = JLD2.load(reference, "smld")
        summarize_smld(smld_ref; label="reference")
        if length(smld_ref.emitters) == length(smld_corrected.emitters)
            # Assume matching order (same upstream 03_frameconnect file was used)
            n = length(smld_corrected.emitters)
            Δ2 = 0.0
            @inbounds for i in 1:n
                dx = smld_corrected.emitters[i].x - smld_ref.emitters[i].x
                dy = smld_corrected.emitters[i].y - smld_ref.emitters[i].y
                Δ2 += dx * dx + dy * dy
            end
            rmsd_vs_ref = sqrt(Δ2 / n)
            @printf("  RMSD vs. reference: %.2f nm  (N = %d)\n", 1000 * rmsd_vs_ref, n)
        else
            @printf("  Skipping RMSD: emitter counts differ (%d vs %d)\n",
                length(smld_corrected.emitters), length(smld_ref.emitters))
        end
    else
        println("\n[4/4] Reference not found or comparison disabled — skipping")
    end

    println("\n" * "=" ^ 72)
    return (
        smld_corrected = smld_corrected,
        info = info,
        elapsed = t_correct,
        rmsd_vs_reference = rmsd_vs_ref,
    )
end

# When run as a script, execute the default iterative case.
if abspath(PROGRAM_FILE) == @__FILE__
    run_hexabody_diagnostics()
end
