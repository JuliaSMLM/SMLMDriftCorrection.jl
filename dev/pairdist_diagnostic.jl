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
using Random
using SMLMData

"""
    nn_pair_distances(smld; k=20, max_dist_um=0.05) -> Vector{Float64}

For each emitter, collect at most `k` nearest neighbours within `max_dist_um`
(μm). Returns the union of those NN distances (one entry per emitter–neighbour
edge — every edge counted twice since both endpoints contribute, but cluster
occupancy bias is removed: dense rulers no longer dominate the histogram with
quadratic pair counts).

This is the v2 metric. v1 (`pair_distances`) returned every within-radius
pair, which weights bright clusters by N² and biases the distribution toward
within-site distances. v2 weights every emitter equally.
"""
function nn_pair_distances(smld::SMLD; k::Int = 20, max_dist_um::Real = 0.05)
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

    # Ask for k+1 NN (first is self), then drop self and any neighbour beyond
    # max_dist. We use knn (sorted=true) rather than inrange so we cap the
    # per-emitter contribution at k regardless of cluster density.
    kquery = min(k + 1, N)
    idxs, dists = knn(kdtree, data, kquery, true)
    out = Float64[]
    sizehint!(out, N * k)
    @inbounds for i in 1:N
        di = dists[i]
        # di[1] is the self-match (distance 0). Skip it.
        for j in 2:length(di)
            d = di[j]
            d > max_dist_um && break  # sorted, so no further entries qualify
            push!(out, d)
        end
    end
    return out
end

"""
    pair_distances(smld; max_dist_um=0.05) -> Vector{Float64}

v1 variant — every unique within-radius pair (i<j) collected via KDTree radius
search. Kept for back-compat / contrast with v2 nn_pair_distances. Biased by
cluster occupancy: pairs grow as N² within dense rulers.
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
    shape_stats(distances; max_um=0.05, tail_cutoffs_um=(0.025, 0.030, 0.035, 0.040))

Robust shape stats for a pair-distance set, no peak-finder, no model
assumptions. Reports:
- `n`: pairs ≤ max_um
- `median_nm`, `mean_nm`
- `iqr_nm`, `std_nm`
- `p90_nm`, `p99_nm`
- `tail_NN_nm`: fraction of (0, max_um] mass beyond `NN` nm, one entry per
  cutoff in `tail_cutoffs_um`.

For data dominated by within-site Rayleigh noise + between-site Rice signal,
two drift methods that find different optima will produce the same Rayleigh
component but different Rice tails — the shape stats catch the difference
without trying to localise a noise-broadened peak.
"""
function shape_stats(distances::Vector{<:Real};
        max_um::Real = 0.05,
        tail_cutoffs_um::Tuple = (0.025, 0.030, 0.035, 0.040))
    in_range = filter(d -> 0 < d <= max_um, distances)
    n = length(in_range)
    if n == 0
        zeros_nt = NamedTuple{Tuple([Symbol(:tail_, round(Int, c*1000), :_nm) for c in tail_cutoffs_um])}(
                    ntuple(_ -> NaN, length(tail_cutoffs_um)))
        return merge((n=0, median_nm=NaN, mean_nm=NaN, iqr_nm=NaN,
                     std_nm=NaN, p90_nm=NaN, p99_nm=NaN), zeros_nt)
    end
    base = (
        n         = n,
        median_nm = 1000 * median(in_range),
        mean_nm   = 1000 * mean(in_range),
        iqr_nm    = 1000 * (quantile(in_range, 0.75) - quantile(in_range, 0.25)),
        std_nm    = 1000 * std(in_range),
        p90_nm    = 1000 * quantile(in_range, 0.90),
        p99_nm    = 1000 * quantile(in_range, 0.99),
    )
    tails = NamedTuple{Tuple([Symbol(:tail_, round(Int, c*1000), :_nm) for c in tail_cutoffs_um])}(
        Tuple(count(d -> d > c, in_range) / n for c in tail_cutoffs_um))
    return merge(base, tails)
end

"""
    pairdist_diagnostic(smld; k=20, max_dist_um=0.05, split_half=true)

v2 diagnostic: k-NN-per-emitter pair distances (cancels cluster-occupancy
bias) + robust shape stats over a wider window (no peak finder). Optionally
odd/even frame-parity split-half on the same stats.

Default `k=20` covers a typical full ruler (locs from both binding sites)
without spilling into neighbouring rulers at hs-tirf density. Adjust if your
data is sparser/denser. Same-input cross-method comparison: tighter
reconstruction → smaller median/IQR/tail; better drift consistency → smaller
split-half difference on those stats.
"""
function pairdist_diagnostic(smld::SMLD;
        k::Int = 20,
        max_dist_um::Real = 0.05,
        tail_cutoffs_um::Tuple = (0.025, 0.030, 0.035, 0.040),
        split_half::Bool = true)

    full_d   = nn_pair_distances(smld; k = k, max_dist_um = max_dist_um)
    full     = shape_stats(full_d; max_um = max_dist_um, tail_cutoffs_um = tail_cutoffs_um)
    base = merge((variant = :v2_nn, k = k), full)

    if !split_half
        return base
    end

    smld_odd, smld_even = split_by_frame_parity(smld)
    odd_d  = nn_pair_distances(smld_odd;  k = k, max_dist_um = max_dist_um)
    even_d = nn_pair_distances(smld_even; k = k, max_dist_um = max_dist_um)
    odd  = shape_stats(odd_d;  max_um = max_dist_um, tail_cutoffs_um = tail_cutoffs_um)
    even = shape_stats(even_d; max_um = max_dist_um, tail_cutoffs_um = tail_cutoffs_um)
    return merge(base, (
        odd_median_nm  = odd.median_nm,
        odd_iqr_nm     = odd.iqr_nm,
        odd_n          = odd.n,
        even_median_nm = even.median_nm,
        even_iqr_nm    = even.iqr_nm,
        even_n         = even.n,
        split_median_diff_nm = abs(odd.median_nm - even.median_nm),
        split_iqr_diff_nm    = abs(odd.iqr_nm - even.iqr_nm),
    ))
end

"""
    nn_distances_per_emitter(smld; k=20, max_dist_um=0.05) -> Vector{Vector{Float64}}

Per-emitter k-NN distance lists (one entry per emitter, length 0..k). The
single KDTree query is the expensive step; storing per-emitter lists lets the
paired bootstrap below resample emitters cheaply without rebuilding the tree.

Self-match is excluded; entries are sorted ascending and capped at `max_dist_um`.
"""
function nn_distances_per_emitter(smld::SMLD; k::Int = 20, max_dist_um::Real = 0.05)
    x = Float64[e.x for e in smld.emitters]
    y = Float64[e.y for e in smld.emitters]
    N = length(x)
    out = [Float64[] for _ in 1:N]
    N < 2 && return out

    data = Matrix{Float64}(undef, 2, N)
    @inbounds for i in 1:N
        data[1, i] = x[i]; data[2, i] = y[i]
    end
    kdtree = KDTree(data; leafsize = 10)

    kquery = min(k + 1, N)
    idxs, dists = knn(kdtree, data, kquery, true)

    @inbounds for i in 1:N
        di = dists[i]
        v = out[i]
        sizehint!(v, kquery - 1)
        for j in 2:length(di)  # skip self at index 1
            d = di[j]
            d > max_dist_um && break
            push!(v, d)
        end
    end
    return out
end

"""
    _shape_stats_from_indexed(per_emit, idxs; max_um, tail_cutoffs_um, scratch=Float64[])

Compute shape stats on the concatenation of per_emit[i] for i in idxs.
Reuses `scratch` to amortise allocation across bootstrap draws.
"""
function _shape_stats_from_indexed(per_emit::Vector{Vector{Float64}}, idxs::AbstractVector{<:Integer};
        max_um::Real = 0.05,
        tail_cutoffs_um::Tuple = (0.025, 0.030, 0.035, 0.040),
        scratch::Vector{Float64} = Float64[])
    total = 0
    @inbounds for i in idxs
        total += length(per_emit[i])
    end
    resize!(scratch, total)
    pos = 1
    @inbounds for i in idxs
        v = per_emit[i]
        for d in v
            scratch[pos] = d
            pos += 1
        end
    end
    return shape_stats(scratch; max_um = max_um, tail_cutoffs_um = tail_cutoffs_um)
end

"""
    paired_bootstrap_compare(per_emit_a, per_emit_b; n_boot=500, seed=42, ...)

Paired bootstrap (resample emitter indices) on the Δstats between two methods
operating on the SAME raw input. Both `per_emit_*` arrays must be aligned
(same length, same indexing). Returns CIs on Δmedian, ΔIQR, Δp90, Δp99, and
Δtail30+ (method_b - method_a — positive Δ ⇒ method_a tighter on that stat).

`n_boot` draws of N indices with replacement; quantiles 2.5/50/97.5%.
"""
function paired_bootstrap_compare(per_emit_a::Vector{Vector{Float64}},
        per_emit_b::Vector{Vector{Float64}};
        n_boot::Int = 500, seed::Int = 42,
        max_um::Real = 0.05,
        tail_cutoffs_um::Tuple = (0.025, 0.030, 0.035, 0.040))

    N = length(per_emit_a)
    @assert N == length(per_emit_b) "per_emit arrays must align (same N)"
    rng = MersenneTwister(seed)

    deltas_med    = Vector{Float64}(undef, n_boot)
    deltas_iqr    = Vector{Float64}(undef, n_boot)
    deltas_p90    = Vector{Float64}(undef, n_boot)
    deltas_p99    = Vector{Float64}(undef, n_boot)
    deltas_tail30 = Vector{Float64}(undef, n_boot)

    scratch_a = Float64[]
    scratch_b = Float64[]
    idxs = Vector{Int}(undef, N)

    for b in 1:n_boot
        rand!(rng, idxs, 1:N)
        sa = _shape_stats_from_indexed(per_emit_a, idxs;
                max_um = max_um, tail_cutoffs_um = tail_cutoffs_um, scratch = scratch_a)
        sb = _shape_stats_from_indexed(per_emit_b, idxs;
                max_um = max_um, tail_cutoffs_um = tail_cutoffs_um, scratch = scratch_b)
        # method_b - method_a so positive Δ means method_a is tighter
        deltas_med[b]    = sb.median_nm - sa.median_nm
        deltas_iqr[b]    = sb.iqr_nm    - sa.iqr_nm
        deltas_p90[b]    = sb.p90_nm    - sa.p90_nm
        deltas_p99[b]    = sb.p99_nm    - sa.p99_nm
        deltas_tail30[b] = sb.tail_30_nm - sa.tail_30_nm
    end

    qci(v) = (lo = quantile(v, 0.025), med = quantile(v, 0.5), hi = quantile(v, 0.975))
    return (
        n_boot         = n_boot,
        delta_median   = qci(deltas_med),
        delta_iqr      = qci(deltas_iqr),
        delta_p90      = qci(deltas_p90),
        delta_p99      = qci(deltas_p99),
        delta_tail30   = qci(deltas_tail30),
    )
end

"""
    print_paired_bootstrap_report(label_a, label_b, ci; α=0.05)

Pretty-print paired CIs. Sign of `(ci.lo, ci.hi)`:
  both negative ⇒ method_a (the first one) clearly tighter on this stat (CI doesn't cross zero)
  both positive ⇒ method_b clearly tighter
  spans zero    ⇒ within noise
"""
function print_paired_bootstrap_report(label_a::AbstractString, label_b::AbstractString, ci::NamedTuple)
    println("=== paired Δ ($label_b − $label_a, B=$(ci.n_boot), 95% CI) ===")
    function fmt(name, q)
        marker = (q.lo > 0) ? "⇒ $label_a tighter" :
                 (q.hi < 0) ? "⇒ $label_b tighter" : "(within noise)"
        @printf("  Δ%-12s : %+7.3f  [%+7.3f, %+7.3f]  %s\n",
                name, q.med, q.lo, q.hi, marker)
    end
    fmt("median",   ci.delta_median)
    fmt("IQR",      ci.delta_iqr)
    fmt("p90",      ci.delta_p90)
    fmt("p99",      ci.delta_p99)
    fmt("tail30+",  ci.delta_tail30)
end

"""
    print_pairdist_report(label, diag)

Pretty-print a v2 diagnostic NamedTuple alongside a method label.
"""
function print_pairdist_report(label::AbstractString, diag::NamedTuple)
    println("=== $label ===")
    @printf("  k-NN pairs (k=%d, ≤50nm)       : %d\n", diag.k, diag.n)
    @printf("  median                       : %6.2f nm\n", diag.median_nm)
    @printf("  IQR                          : %6.2f nm  (smaller = tighter)\n", diag.iqr_nm)
    @printf("  std                          : %6.2f nm\n", diag.std_nm)
    @printf("  p90 / p99                    : %6.2f / %6.2f nm\n", diag.p90_nm, diag.p99_nm)
    if haskey(diag, :tail_25_nm)
        @printf("  tail fractions               : >25nm %.4f  >30nm %.4f  >35nm %.4f  >40nm %.4f\n",
                diag.tail_25_nm, diag.tail_30_nm, diag.tail_35_nm, diag.tail_40_nm)
    end
    if haskey(diag, :odd_median_nm)
        @printf("  split-half odd  median/IQR   : %5.2f / %5.2f nm  (n=%d)\n",
                diag.odd_median_nm, diag.odd_iqr_nm, diag.odd_n)
        @printf("  split-half even median/IQR   : %5.2f / %5.2f nm  (n=%d)\n",
                diag.even_median_nm, diag.even_iqr_nm, diag.even_n)
        @printf("  split-half median Δ / IQR Δ  : %6.2f / %6.2f nm  (smaller = consistent)\n",
                diag.split_median_diff_nm, diag.split_iqr_diff_nm)
    end
end
