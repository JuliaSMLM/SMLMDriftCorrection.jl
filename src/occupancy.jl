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
point). Node `id` holds the points `perm[lo[id]:hi[id]]`; `left[id] == 0` marks a leaf.
"""
struct _PopKDTree{N}
    pts::Vector{NTuple{N,Float64}}      # in the order given
    perm::Vector{Int}                   # tree order -> index into `pts`
    lo::Vector{Int}
    hi::Vector{Int}
    left::Vector{Int}
    right::Vector{Int}
    bmin::Vector{NTuple{N,Float64}}
    bmax::Vector{NTuple{N,Float64}}
end

"Most points in a `_PopKDTree` leaf."
const _KDTREE_LEAFSIZE = 32

_point(e, ::Val{2}) = (Float64(e.x), Float64(e.y))
_point(e, ::Val{3}) = (Float64(e.x), Float64(e.y), Float64(e.z))

function _PopKDTree(pts::Vector{NTuple{N,Float64}}) where {N}
    n = length(pts)
    tree = _PopKDTree{N}(pts, collect(1:n), Int[], Int[], Int[], Int[],
                         NTuple{N,Float64}[], NTuple{N,Float64}[])
    n > 0 && _build!(tree, 1, n)
    return tree
end
_PopKDTree(sd::SMLD) = _PopKDTree(NTuple{nDims(sd),Float64}[_point(e, Val(nDims(sd))) for e in sd.emitters])

# Build the node over tree positions `l:h` and return its id; children split at the median of
# the widest dimension (`partialsort!`; the bounds come from the points, so a split that is
# not a clean partition costs speed, never correctness).
function _build!(tree::_PopKDTree{N}, l::Int, h::Int) where {N}
    pts, perm = tree.pts, tree.perm
    bmin = bmax = pts[perm[l]]
    for k in (l + 1):h
        p = pts[perm[k]]
        bmin = map(min, bmin, p)
        bmax = map(max, bmax, p)
    end
    push!(tree.lo, l); push!(tree.hi, h); push!(tree.left, 0); push!(tree.right, 0)
    push!(tree.bmin, bmin); push!(tree.bmax, bmax)
    id = length(tree.lo)
    if h - l + 1 > _KDTREE_LEAFSIZE
        d = argmax(bmax .- bmin)
        m = (l + h) ÷ 2
        partialsort!(view(perm, l:h), m - l + 1; by = i -> pts[i][d])
        tree.left[id] = _build!(tree, l, m)
        tree.right[id] = _build!(tree, m + 1, h)
    end
    return id
end

"""
Exact count of the points of `tree` (the query point itself included) within `r` of each
point `tree.pts[q]`, `q` in `queries`.
"""
function _count_within(tree::_PopKDTree{N}, r::Float64, queries::AbstractVector{Int}) where {N}
    isempty(tree.pts) && return zeros(Int, length(queries))
    r2 = r * r
    spts = tree.pts[tree.perm]
    cnt = zeros(Int, length(queries))
    stack = Int[]
    for (iq, q) in enumerate(queries)
        qp = tree.pts[q]
        c = 0
        push!(stack, 1)
        while !isempty(stack)
            id = pop!(stack)
            dmin = 0.0; dmax = 0.0
            for d in 1:N
                a = tree.bmin[id][d] - qp[d]; b = qp[d] - tree.bmax[id][d]
                g = max(a, b, 0.0)
                dmin += g * g
                f = max(abs(tree.bmin[id][d] - qp[d]), abs(tree.bmax[id][d] - qp[d]))
                dmax += f * f
            end
            dmin > r2 && continue
            if dmax <= r2
                c += tree.hi[id] - tree.lo[id] + 1
            elseif tree.left[id] == 0
                for j in tree.lo[id]:tree.hi[id]
                    s = 0.0
                    for d in 1:N
                        s += (spts[j][d] - qp[d])^2
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

"Relative tolerance within which occupancy sums count as equal (`_occ_gt`)."
const _OCC_RTOL = 1e-9

"""
Whether occupancy `a` exceeds `b` by more than rounding: the sums of inverse density weights
are equal in exact arithmetic for equal site sets but differ in the last bits by summation
order, so every occupancy comparison goes through this.
"""
_occ_gt(a, b) = a > b + _OCC_RTOL * max(abs(a), abs(b), 1)

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
    return nb + nc > 0 && (nb - nc) > z * sqrt(nb + nc) + _OCC_RTOL * max(nb, nc, 1)
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

