# Overlap decision helpers of the registered inter alignment: per-localization inlier flags,
# the occupancy score (inverse local density, so a site localized many times counts once)
# with the population k-d tree that counts local density, and the overlap arbiter.

"""
Per-localization inlier flags of a dataset (after applying `inter`): `true` where the
localization lies within `radius` (μm) of a consensus localization in `tree`. The mean
is `_overlap_score`; the flags let two shifts be compared on the localizations they
disagree about (`_overlap_leads`).
"""
function _overlap_inliers(sd::SMLD, inter::Vector{Float64}, tree, radius::Float64, n_dims::Int)
    flags = falses(length(sd.emitters))
    pt = zeros(Float64, n_dims)
    for (i, e) in enumerate(sd.emitters)
        pt[1] = e.x - inter[1]
        pt[2] = e.y - inter[2]
        n_dims == 3 && (pt[3] = e.z - inter[3])
        _, dists = knn(tree, pt, 1)
        flags[i] = dists[1] < radius
    end
    return flags
end

"Separation (μm) within which two localizations count as one site in `_occupancy`."
const _OCC_SEP = 0.030

"""
K-d tree over `N`-dimensional points with a bounding box and population per node, for exact
range counts (`_count_within`). A node whose box lies entirely within `r` of a query point
adds its whole population and one whose box lies entirely beyond `r` is skipped, so only
boundary leaves are checked point by point. A compact aggregate then costs per tree node,
not per pair of localizations (`NearestNeighbors.inrangecount` visits every contained
point). Coordinates are plain `N`-row matrices (`pts` as given, `spts` in tree order) and the
node boxes flat vectors with `N` entries per node, so the query loop does scalar arithmetic
only. Node `id` holds the points `perm[lo[id]:hi[id]]`; `left[id] == 0` marks a leaf.
"""
struct _PopKDTree{N}
    pts::Matrix{Float64}
    spts::Matrix{Float64}
    perm::Vector{Int}                   # tree order -> column of `pts`
    lo::Vector{Int}
    hi::Vector{Int}
    left::Vector{Int}
    right::Vector{Int}
    bmin::Vector{Float64}               # node `id`, dimension `d` at `(id - 1) * N + d`
    bmax::Vector{Float64}
end

"Most points in a `_PopKDTree` leaf."
const _KDTREE_LEAFSIZE = 32

function _PopKDTree(pts::Matrix{Float64})
    N, n = size(pts)
    tree = _PopKDTree{N}(pts, pts, collect(1:n), Int[], Int[], Int[], Int[], Float64[], Float64[])
    n > 0 && _build!(tree, 1, n)
    return _PopKDTree{N}(pts, pts[:, tree.perm], tree.perm, tree.lo, tree.hi, tree.left,
                         tree.right, tree.bmin, tree.bmax)
end
function _PopKDTree(sd::SMLD)
    N = nDims(sd)
    pts = Matrix{Float64}(undef, N, length(sd.emitters))
    for (i, e) in enumerate(sd.emitters)
        pts[1, i] = e.x; pts[2, i] = e.y
        N == 3 && (pts[3, i] = e.z)
    end
    return _PopKDTree(pts)
end

# Median select (Wirth): reorder `perm[l:h]` so that every point before position `k` has
# coordinate `d` <= that of the one at `k`, and every point after it >= (O(n), not a sort;
# `partialsort!` does not partition, which leaves overlapping boxes and a slow query).
function _select!(perm::Vector{Int}, pts::Matrix{Float64}, d::Int, l::Int, h::Int, k::Int)
    @inbounds while l < h
        pivot = pts[d, perm[(l + h) >> 1]]
        i, j = l, h
        while i <= j
            while pts[d, perm[i]] < pivot; i += 1; end
            while pts[d, perm[j]] > pivot; j -= 1; end
            if i <= j
                perm[i], perm[j] = perm[j], perm[i]
                i += 1; j -= 1
            end
        end
        if k <= j
            h = j
        elseif k >= i
            l = i
        else
            break
        end
    end
    return perm
end

# Build the node over tree positions `l:h` and return its id; children split at the median of
# the widest dimension.
function _build!(tree::_PopKDTree{N}, l::Int, h::Int) where {N}
    pts, perm = tree.pts, tree.perm
    id = length(tree.lo) + 1
    push!(tree.lo, l); push!(tree.hi, h); push!(tree.left, 0); push!(tree.right, 0)
    widest, d = -1.0, 1
    for k in 1:N
        lo = hi = pts[k, perm[l]]
        for j in (l + 1):h
            v = pts[k, perm[j]]
            lo = min(lo, v); hi = max(hi, v)
        end
        push!(tree.bmin, lo); push!(tree.bmax, hi)
        hi - lo > widest && ((widest, d) = (hi - lo, k))
    end
    if h - l + 1 > _KDTREE_LEAFSIZE
        m = (l + h) ÷ 2
        _select!(perm, pts, d, l, h, m)
        tree.left[id] = _build!(tree, l, m)
        tree.right[id] = _build!(tree, m + 1, h)
    end
    return id
end

"""
Exact count of the points of `tree` (the query point itself included) within `r` of each
point `tree.pts[:, q]`, `q` in `queries`.
"""
_count_within(tree::_PopKDTree, r::Float64, queries::AbstractVector{Int}) =
    _count_near(tree, r, view(tree.pts, :, queries))

"""
Exact count of the points of `tree` within `r` of each column of `Q` (`N` rows, any points).
"""
function _count_near(tree::_PopKDTree{N}, r::Float64, Q::AbstractMatrix{Float64}) where {N}
    r2 = r * r
    spts, bmin, bmax = tree.spts, tree.bmin, tree.bmax
    cnt = zeros(Int, size(Q, 2))
    isempty(tree.pts) && return cnt
    stack = Int[]
    @inbounds for iq in axes(Q, 2)
        c = 0
        push!(stack, 1)
        while !isempty(stack)
            id = pop!(stack)
            dmin = 0.0; dmax = 0.0
            for d in 1:N
                v = Q[d, iq]
                a = bmin[(id - 1) * N + d] - v; b = v - bmax[(id - 1) * N + d]
                g = max(a, b, 0.0)
                dmin += g * g
                f = max(abs(a), abs(b))   # the farther box face
                dmax += f * f
            end
            dmin > r2 && continue
            if dmax <= r2
                c += tree.hi[id] - tree.lo[id] + 1
            elseif tree.left[id] == 0
                for j in tree.lo[id]:tree.hi[id]
                    s = 0.0
                    for d in 1:N
                        s += (spts[d, j] - Q[d, iq])^2
                    end
                    s <= r2 && (c += 1)
                end
            else
                push!(stack, tree.left[id], tree.right[id])
            end
        end
        cnt[iq] = c
    end
    return cnt
end

"""
Inverse local density weights of the localizations of `sd`: `w_i = 1 / n_i`, with `n_i` the
number of its localizations within `_OCC_SEP` of localization `i` (itself included; 3D uses
z), for the localizations flagged in `need`; `w_i = 0` elsewhere (`n_i` still counts every
localization). Independent of any candidate shift (positions are shift invariant) and of the
emitter order. A compact site (all its localizations within `_OCC_SEP` of each other, sites
farther apart than that) sums to 1 however often it was localized. Every decision compares
candidates, and a localization that is an inlier under all of them or under none adds the
same to each score, so `need` is the set where the flags disagree (`_disagreeing`) and the
cost scales with the disagreement, not with the dataset. Without `need`, all of them.
"""
function _occupancy_weights(sd::SMLD, need::BitVector = trues(length(sd.emitters)))
    w = zeros(length(need))
    queries = findall(need)
    isempty(queries) && return w
    w[queries] .= 1.0 ./ _count_within(_PopKDTree(sd), _OCC_SEP, queries)
    return w
end

"Localizations that some but not all of the inlier `flags` include: where the candidates disagree."
_disagreeing(flags::BitVector...) = reduce(.|, flags) .& .!reduce(.&, flags)

"Absolute tolerance, in effective sites, within which occupancy sums count as equal (`_occ_gt`)."
const _OCC_ATOL = 1e-6

"""
Whether occupancy `a` exceeds `b` by more than rounding: the sums of inverse density weights
are equal in exact arithmetic for equal site sets but differ in the last bits by summation
order, so every occupancy comparison goes through this. The tolerance is absolute, in effective
sites: the sums cover only the localizations where the candidates disagree (`_disagreeing`),
so their size says nothing about the rounding, which stays near 1e-11; a difference under a
millionth of a site is a tie.
"""
_occ_gt(a, b) = a > b + _OCC_ATOL

"""
Occupancy of a dataset's inlier localizations: the sum of their inverse local density weights
`w` (`_occupancy_weights`), the number of effective sites. Monotone in the flags, identical
(up to summation rounding) under any permutation of the emitters, and equal to the number of
covered sites when sites are compact and separated, so a site localized many times counts
once. Overlap decisions compare this, not the inlier count, so a shift that lays frequently
localized sites on the consensus cannot outscore the true shift. The `SMLD` method builds
the weights itself.
"""
function _occupancy(w::Vector{Float64}, flags::BitVector)
    t = 0.0
    @inbounds for i in eachindex(w)
        flags[i] && (t += w[i])
    end
    return t
end
_occupancy(sd::SMLD, flags::BitVector) = _occupancy(_occupancy_weights(sd), flags)

"Significance of a lead in overlap: McNemar's z (`_overlap_leads`)."
const _OVERLAP_Z = 3.0

"""
Whether inlier flags `a` lead `b` significantly: with `nb` localizations inlier only
under `a` and `nc` only under `b`, McNemar's z on the discordant localizations,
`(nb - nc) / sqrt(nb + nc)`, exceeds `z` (default `_OVERLAP_Z`). With weights `w` (`_occupancy_weights`) the discordant
localizations are summed by weight, McNemar on effective sites. Localizations inlier under both or
neither carry no evidence, so a few-localization lead on sparse data does not count.
"""
function _overlap_leads(a::BitVector, b::BitVector; z::Float64 = _OVERLAP_Z,
                        w::Union{Nothing,Vector{Float64}} = nothing)
    nb = w === nothing ? count(a .& .!b) : sum(w[a .& .!b]; init = 0.0)
    nc = w === nothing ? count(b .& .!a) : sum(w[b .& .!a]; init = 0.0)
    return nb + nc > 0 && (nb - nc) > z * sqrt(nb + nc) + _OCC_ATOL
end

"Significance (`_overlap_leads` z) of the winner's site-level lead above which entropy cannot override it."
const _TIE_Z = 2.0

"Relative tolerance within which two entropy costs count as equal (`_arbitrate`)."
const _COST_RTOL = 1e-10

"""
Overlap arbiter over {refined, CC, incumbent} (`cc` and `in_cc` may be `nothing`): the
candidate with the largest `score` of its inlier flags wins the basin (`_cc_primary_inter!`
passes `_occupancy`), ties keeping refined, then CC, then the incumbent; `score` is called
once per flag vector. Where `cost` is given, the candidates within `_CC_VERIFY_RADIUS` of
that winner W, except those W leads at the site level (`_overlap_leads` with weights `w`,
default one per localization, and `z = _TIE_Z`), form a near tie with
W; if it holds two or more distinct shifts, the one with the lowest `cost(shift)` is kept,
evaluated once per distinct shift. Costs within `_COST_RTOL` of the lowest count as equal and
equal costs keep the order incumbent, refined, CC: a flat cost has resolved nothing, so the
incumbent is not displaced by a refined shift that merely equals the CC. A lead of a few
localizations between candidates this close is noise, and the entropy cost the refine
minimizes positions the shift within the basin; a significant site-level lead is not noise.
"""
function _arbitrate(refined, in_ref::BitVector, cc, in_cc, inc, in_inc::BitVector;
                    score = count, cost = nothing, w = nothing)
    sc_ref = score(in_ref)
    sc_cc = cc === nothing ? 0.0 : score(in_cc)
    sc_inc = score(in_inc)
    win, in_win, sc_win = refined, in_ref, sc_ref
    if cc !== nothing && _occ_gt(sc_cc, sc_win)
        win, in_win, sc_win = cc, in_cc, sc_cc
    end
    if _occ_gt(sc_inc, sc_win)
        win, in_win, sc_win = inc, in_inc, sc_inc
    end
    cost === nothing && return win
    tied = Vector{Float64}[]                       # order: incumbent, refined, CC
    for (c, f) in ((inc, in_inc), (refined, in_ref), (cc, in_cc))
        c === nothing && continue
        (c === win || (norm(c .- win) < _CC_VERIFY_RADIUS &&
                       !_overlap_leads(in_win, f; z = _TIE_Z, w = w))) || continue
        push!(tied, c)
    end
    distinct = unique(tied)
    length(distinct) < 2 && return win
    costs = map(cost, distinct)
    lowest = minimum(costs)
    return distinct[findfirst(c -> c <= lowest + _COST_RTOL * abs(lowest), costs)]
end

"""
Lazy inter entropy cost for dataset `d` against the corrected consensus `snap` of
`others`, the objective `findinter!` minimizes (`_inter_entropy_cost`), as a function of
a shift with neighbors built at that shift. Nothing is built unless it is called.
"""
function _arbiter_cost(model::LegendrePolynomial, smld::SMLD, snap::SMLD, d::Int,
                       others::Vector{Int}, maxn::Int)
    setup = Ref{Any}(nothing)
    return function (θ)
        if setup[] === nothing
            scratch = InterShift(nDims(smld), zeros(nDims(smld)))
            setup[] = _inter_entropy_cost(model, smld, snap, d, others, maxn, scratch)
        end
        (cost, state, _) = setup[]
        state.needs_initial_build = true
        return cost(Float64.(θ))
    end
end

"Most localizations per dataset that `_dataset_support` tests (a pseudo-random sample beyond this)."
const _SUPPORT_MAX_TESTED = 5000

"Hash seed of `_dataset_support`'s sample: deterministic, independent of emitter order periodicity."
const _SUPPORT_SALT = 0x5eed_5a3b_1e00_0000

"Displacement (μm) of `_dataset_support`'s chance level: the test is repeated this far off, in 8 directions."
const _SUPPORT_OFFSET = 0.150

"Chance coverage (fraction) that sets `_dataset_support`'s radius on dense data."
const _SUPPORT_CHANCE = 0.25

"""
    _dataset_support(sc::SMLD, labels = dataset of each emitter, n = sc.n_datasets) -> Vector{DatasetSupport}

Whether each dataset of the corrected `sc` co-localizes with the other datasets more than
chance, i.e. whether its inter alignment rests on evidence. `labels` (one per emitter, `1:n`)
can name other units than the datasets: chunked continuous mode passes the chunks, the units
its inter alignment placed. Per dataset, over its tested
localizations `i` (all, or a pseudo-random sample of `M = _SUPPORT_MAX_TESTED` of its `n`, chosen
by a hash of the index, so no periodicity in emitter order can bias it), with occupancy weights
`w_i = (n / M) / n_i` (`n_i` its own localizations within `_OCC_SEP` of `i`, so a site localized
many times counts once; each tested localization stands for `n / M`, so sampling keeps the
effective-site count and `z` does not shrink with `n`):

- `c_i`: whether another dataset has a localization within `radius` of `i`. The radius is
  `_CC_VERIFY_RADIUS` (the inlier test of the inter alignment), reduced on dense data until a
  point placed at random would be covered with chance `_SUPPORT_CHANCE` (from the median
  number of other-dataset localizations within `_SUPPORT_OFFSET`, as a Poisson count): at
  30 nm a dense reference covers every point, aligned or not, and the test says nothing;
- `p_i`: the same test with `i` displaced by `_SUPPORT_OFFSET` in 8 directions in x-y,
  `(k_i + 1/2) / 9` for `k_i` of the 8 covered: the chance level at that local density, with
  the reference's own clustering;
- `colocalized = Σ w_i c_i / W` and `expected = Σ w_i p_i / W` (`W = Σ w_i`, the effective
  sites), `z = Σ w_i (c_i - p_i) / sqrt(Σ w_i p_i (1 - p_i))` (sites taken as independent
  trials), and `supported = z > _TIE_Z`.

A dataset placed at random relative to the others (thrown, or not the same structure) is not
covered more at its own position than 150 nm off, so `z` is near or below 0; a correctly aligned
one that shares localized sites with the others has `z` well above 2. Empty datasets get
`z = NaN` and `supported = false`. Returns an empty vector for fewer than two datasets.
"""
function _dataset_support(sc::SMLD, ds::Vector{Int} = Int[e.dataset for e in sc.emitters],
                          nd::Int = sc.n_datasets)
    out = DatasetSupport[]
    nd < 2 && return out
    D = nDims(sc)
    all_tree = _PopKDTree(sc)
    δ = _SUPPORT_OFFSET
    dirs = [(cospi(k / 4), sinpi(k / 4)) for k in 0:7]
    for d in 1:nd
        idx = findall(==(d), ds)
        n = length(idx)
        if n == 0
            push!(out, (dataset = d, n_locs = 0, colocalized = 0.0, expected = 0.0, z = NaN,
                        radius = NaN, supported = false))
            continue
        end
        own_tree = _PopKDTree(all_tree.pts[:, idx])
        M = min(n, _SUPPORT_MAX_TESTED)
        tested = M == n ? idx : idx[sort(partialsortperm([hash(i, _SUPPORT_SALT) for i in 1:n], 1:M))]
        X = all_tree.pts[:, tested]
        others(r, Q) = _count_near(all_tree, r, Q) .- _count_near(own_tree, r, Q)
        w = (n / M) ./ _count_near(own_tree, _OCC_SEP, X)   # each tested localization stands for n / M
        # radius: chance coverage _SUPPORT_CHANCE at the median local density of the others
        m = median(others(δ, X))
        r = m > 0 ? min(_CC_VERIFY_RADIUS, δ * (-log1p(-_SUPPORT_CHANCE) / m)^(1 / D)) :
                    _CC_VERIFY_RADIUS
        c = others(r, X) .> 0
        k = zeros(Int, size(X, 2))
        Xk = similar(X)
        for (ux, uy) in dirs
            Xk .= X
            Xk[1, :] .+= δ * ux
            Xk[2, :] .+= δ * uy
            k .+= others(r, Xk) .> 0
        end
        p = (k .+ 0.5) ./ (length(dirs) + 1)
        W = sum(w)
        obs = sum(w[c]; init = 0.0)
        ex = sum(w .* p)
        z = (obs - ex) / sqrt(sum(@. w * p * (1 - p)))
        push!(out, (dataset = d, n_locs = n, colocalized = obs / W, expected = ex / W, z = z,
                    radius = r, supported = z > _TIE_Z))
    end
    return out
end

"""
Log one warning naming the datasets (`unit = "dataset"`) or chunks (`"chunk"`) with
localizations whose alignment rests on no evidence (`_dataset_support`): their positions
relative to the others are not established, so treat them as unaligned.
"""
function _warn_unsupported(support, unit::String = "dataset")
    bad = [s.dataset for s in support if s.n_locs > 0 && !s.supported]
    isempty(bad) && return
    @warn "driftcorrect: $(unit)(s) $(bad) do not co-localize with the other $(unit)s more " *
          "than chance after alignment (info.dataset_support, z <= $(_TIE_Z)); their inter " *
          "alignment rests on no evidence (too sparse, or not the same structure). Treat " *
          "their positions as unaligned."
    return
end

"""
Alignment support of a corrected result (`_dataset_support`) over the units the inter alignment
placed: the datasets, or in chunked continuous mode (`work` = the chunked data, emitters in the
same order as `sc`) the chunks. Logs one warning naming the unsupported ones.
"""
function _support_and_warn(sc::SMLD, work::SMLD, chunked::Bool)
    support = chunked ? _dataset_support(sc, Int[e.dataset for e in work.emitters], work.n_datasets) :
                        _dataset_support(sc)
    _warn_unsupported(support, chunked ? "chunk" : "dataset")
    return support
end

