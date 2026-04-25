# pairdist_diagnostic.jl — coordinate-only nanoruler reconstruction-quality
# diagnostic for arbitrating between drift-correction methods.
#
# Background: ref-pipeline RMSD isn't ground truth on real data — the reference
# pipeline can itself be wrong (non-converged, gauge-locked at a stale optimum).
# What matters is whether the *corrected* SMLD yields a tighter, less-tailed
# 20 nm pair-distance distribution and is consistent across frame-parity halves.
#
# This is the lightweight v1: nearest-neighbor pair-distance histogram, peak
# stats over a window around 20 nm, tail mass, and odd/even split-half
# consistency. Coordinate-only, no BaGoL or external clustering required.
# Per @bagol's recommendation: scaffold light first, escalate to run_bagol +
# per-cluster log_evidence only if the lightweight metric is ambiguous.

using NearestNeighbors
using Statistics
using Printf
using SMLMData

"""
    pair_distances(smld; max_dist_um=0.05) -> Vector{Float64}

Return all unique within-radius pairwise distances (μm) in `smld`. Pairs are
collected via KDTree radius search; each pair counted once (i<j).

`max_dist_um` controls the search radius. 50 nm is a good default for 20 nm
nanorulers (captures the within-cluster 20 nm signal and a little
between-cluster background).
"""
function pair_distances(smld::SMLD; max_dist_um::Real = 0.05)
    x = Float64[e.x for e in smld.emitters]
    y = Float64[e.y for e in smld.emitters]
    N = length(x)
    if N < 2
        return Float64[]
    end
    data = Matrix{Float64}(undef, 2, N)
    @inbounds for i in 1:N
        data[1, i] = x[i]
        data[2, i] = y[i]
    end
    kdtree = KDTree(data; leafsize = 10)

    distances = Float64[]
    @inbounds for i in 1:N
        idxs = inrange(kdtree, view(data, :, i), Float64(max_dist_um))
        for j in idxs
            if j > i
                d = hypot(x[i] - x[j], y[i] - y[j])
                push!(distances, d)
            end
        end
    end
    return distances
end

"""
    pairdist_histogram(distances; bin_um=0.001, max_um=0.05) -> (centers_um, counts)

Histogram pair distances. Default 1 nm bins out to 50 nm.
"""
function pairdist_histogram(distances::Vector{<:Real}; bin_um::Real = 0.001, max_um::Real = 0.05)
    n_bins = max(1, ceil(Int, max_um / bin_um))
    counts = zeros(Int, n_bins)
    @inbounds for d in distances
        d < 0 && continue
        bin = floor(Int, d / bin_um) + 1
        if 1 <= bin <= n_bins
            counts[bin] += 1
        end
    end
    centers = [(k - 0.5) * bin_um for k in 1:n_bins]
    return centers, counts
end

"""
    peak_stats(distances; window_um=(0.012, 0.035)) -> NamedTuple

Robust peak statistics for the 20 nm nanoruler peak. Window excludes the
within-site cluster at small distances (`> 12 nm`) and trims the
between-cluster tail (`< 35 nm`).

Returns:
- `n`: count of pair distances inside the window
- `median_nm`, `mean_nm`, `iqr_nm`, `std_nm`: in nanometers
- `peak_nm`: histogram-mode position inside the window (1 nm bins)
"""
function peak_stats(distances::Vector{<:Real}; window_um::Tuple{<:Real,<:Real} = (0.012, 0.035))
    in_window = filter(d -> window_um[1] <= d <= window_um[2], distances)
    if isempty(in_window)
        return (n = 0, median_nm = NaN, mean_nm = NaN,
                iqr_nm = NaN, std_nm = NaN, peak_nm = NaN)
    end
    centers, counts = pairdist_histogram(in_window;
        bin_um = 0.001, max_um = window_um[2] + 0.001)
    # Restrict the mode search to the same window.
    keep = window_um[1] .<= centers .<= window_um[2]
    sub_centers = centers[keep]
    sub_counts  = counts[keep]
    peak_idx = argmax(sub_counts)
    peak_nm = 1000 * sub_centers[peak_idx]
    return (
        n         = length(in_window),
        median_nm = 1000 * median(in_window),
        mean_nm   = 1000 * mean(in_window),
        iqr_nm    = 1000 * (quantile(in_window, 0.75) - quantile(in_window, 0.25)),
        std_nm    = 1000 * std(in_window),
        peak_nm   = peak_nm,
    )
end

"""
    tail_fraction(distances; cutoff_um=0.030, max_um=0.05) -> Float64

Fraction of pair distances inside `(0, max_um]` that lie beyond `cutoff_um`.
For nanoruler data, well-corrected drift produces a distribution dominated
by the 20 nm peak and the small-distance within-site cluster, leaving
relatively little mass beyond 30 nm. Higher tail = more diffuse / smeared.
"""
function tail_fraction(distances::Vector{<:Real}; cutoff_um::Real = 0.030, max_um::Real = 0.05)
    in_range = filter(d -> 0 < d <= max_um, distances)
    isempty(in_range) && return NaN
    return count(d -> d > cutoff_um, in_range) / length(in_range)
end

"""
    split_by_frame_parity(smld) -> (smld_odd, smld_even)

Partition emitters by their frame parity. Same camera, n_frames, n_datasets;
metadata copied. Used for split-half consistency checks.
"""
function split_by_frame_parity(smld::SMLD)
    odd_mask  = [isodd(e.frame)  for e in smld.emitters]
    even_mask = [iseven(e.frame) for e in smld.emitters]
    smld_odd  = typeof(smld)(smld.emitters[odd_mask],  smld.camera,
                             smld.n_frames, smld.n_datasets, copy(smld.metadata))
    smld_even = typeof(smld)(smld.emitters[even_mask], smld.camera,
                             smld.n_frames, smld.n_datasets, copy(smld.metadata))
    return smld_odd, smld_even
end

"""
    pairdist_diagnostic(smld; max_dist_um=0.05, peak_window_um=(0.012, 0.035),
                              tail_cutoff_um=0.030, split_half=true)

Return a NamedTuple with peak stats, tail fraction, and (optionally)
odd/even split-half consistency for the same metrics. Higher peak n with
narrower IQR / std and lower tail fraction → tighter reconstruction. Split-half
agreement on `peak_nm` indicates drift correction is consistent across frames.
"""
function pairdist_diagnostic(smld::SMLD;
        max_dist_um::Real = 0.05,
        peak_window_um::Tuple{<:Real,<:Real} = (0.012, 0.035),
        tail_cutoff_um::Real = 0.030,
        split_half::Bool = true)

    full_d = pair_distances(smld; max_dist_um = max_dist_um)
    full_stats = peak_stats(full_d; window_um = peak_window_um)
    full_tail  = tail_fraction(full_d; cutoff_um = tail_cutoff_um, max_um = max_dist_um)

    base = (
        n_pairs        = length(full_d),
        peak_nm        = full_stats.peak_nm,
        median_nm      = full_stats.median_nm,
        mean_nm        = full_stats.mean_nm,
        iqr_nm         = full_stats.iqr_nm,
        std_nm         = full_stats.std_nm,
        in_window_n    = full_stats.n,
        tail_fraction  = full_tail,
    )

    if !split_half
        return base
    end

    smld_odd, smld_even = split_by_frame_parity(smld)
    odd_d  = pair_distances(smld_odd;  max_dist_um = max_dist_um)
    even_d = pair_distances(smld_even; max_dist_um = max_dist_um)
    odd_stats  = peak_stats(odd_d;  window_um = peak_window_um)
    even_stats = peak_stats(even_d; window_um = peak_window_um)
    return merge(base, (
        odd_peak_nm    = odd_stats.peak_nm,
        odd_iqr_nm     = odd_stats.iqr_nm,
        odd_n          = odd_stats.n,
        even_peak_nm   = even_stats.peak_nm,
        even_iqr_nm    = even_stats.iqr_nm,
        even_n         = even_stats.n,
        split_peak_diff_nm = abs(odd_stats.peak_nm - even_stats.peak_nm),
    ))
end

"""
    print_pairdist_report(label, diag)

Pretty-print a diagnostic NamedTuple alongside a method label.
"""
function print_pairdist_report(label::AbstractString, diag::NamedTuple)
    println("=== $label ===")
    @printf("  pairs <50nm                 : %d\n", diag.n_pairs)
    @printf("  20nm peak window pairs       : %d\n", diag.in_window_n)
    @printf("  peak position                : %6.2f nm  (target 20.00)\n", diag.peak_nm)
    @printf("  median (in window)           : %6.2f nm\n", diag.median_nm)
    @printf("  IQR  (in window)             : %6.2f nm  (smaller = tighter)\n", diag.iqr_nm)
    @printf("  std  (in window)             : %6.2f nm\n", diag.std_nm)
    @printf("  tail fraction (>30nm)        : %6.4f    (smaller = less smear)\n", diag.tail_fraction)
    if haskey(diag, :odd_peak_nm)
        @printf("  split-half odd  peak/IQR     : %5.2f / %5.2f nm  (n=%d)\n",
                diag.odd_peak_nm, diag.odd_iqr_nm, diag.odd_n)
        @printf("  split-half even peak/IQR     : %5.2f / %5.2f nm  (n=%d)\n",
                diag.even_peak_nm, diag.even_iqr_nm, diag.even_n)
        @printf("  split-half peak Δ            : %6.2f nm  (smaller = consistent)\n",
                diag.split_peak_diff_nm)
    end
end
