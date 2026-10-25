# Codex review fixtures (4323bf9 rounding, 912f617 order, b2795da P1 and P2): hand-placed localization arrangements through driftcorrect.
using Test, SMLMDriftCorrection, SMLMSim, Random

const DC = SMLMDriftCorrection

include(joinpath(@__DIR__, "..", "points_smld.jl"))

# Codex review of 4323bf9: occupancy sums that are equal in exact arithmetic differ in the
# last bits (10.0 vs 9.999999999999982) and must not decide the guard. Ten A and ten B
# compact motifs; the target holds 6:15 copies of each A shifted by (3, 0) and a varying
# number of B copies, the reference four copies of each. Both candidates cover ten sites.
@testset "occupancy ties are decided with a tolerance (Codex 4323bf9)" begin
    A = [(0.600 + 0.110i, 0.600 + 0.023i^2) for i in 0:9]
    B = [(4.400 + 0.105i, 4.400 + 0.011i^2) for i in 0:9]
    ref = reduce(vcat, [fill(p, 4) for p in vcat(A, B)])
    nb = [3, 3, 3, 4, 4, 3, 2, 4, 4, 2]
    tgt = vcat(reduce(vcat, [fill((p[1] + 3.0, p[2]), 5 + i) for (i, p) in enumerate(A)]),
               reduce(vcat, [fill(p, nb[i]) for (i, p) in enumerate(B)]))
    smld = points_smld([ref, tgt])
    shifts = Vector{Float64}[]
    rng = Random.MersenneTwister(77)
    for k in 1:6
        sm = k == 1 ? smld : begin
            em = deepcopy(smld.emitters)
            i2 = findall(e -> e.dataset == 2, em)
            em[i2] = em[i2][Random.randperm(rng, length(i2))]
            BasicSMLD(em, smld.camera, 300, 2, Dict{String,Any}())
        end
        (_, info) = DC.driftcorrect(sm; quality = :fft, degree = 0)
        push!(shifts, info.model.inter[2].dm .- info.model.inter[1].dm)
    end
    @test all(sh -> maximum(abs.(sh .- [3.0, 0.0])) < 0.005, shifts)
end

# Codex review of 912f617: a center that matches neither candidate must not merge two
# inlier groups 36 nm apart when it comes first in the emitter order.
@testset "occupancy: unmatched centers do not merge groups (Codex 912f617)" begin
    for centers_first in (true, false)
        tgt = Tuple{Float64,Float64}[]
        for (cx, cy) in ((4.010, 1.0), (4.030, 2.0))
            grp = vcat([(cx - 0.018, cy) for _ in 1:10], [(cx + 0.018, cy) for _ in 1:10])
            centers_first ? append!(tgt, [(cx, cy)], grp) : append!(tgt, grp, [(cx, cy)])
        end
        for p in ((5.0, 4.0), (5.2, 5.0), (5.7, 4.3)); append!(tgt, fill(p, 4)); end
        ref = Tuple{Float64,Float64}[]
        for cy in (1.0, 2.0), sgn in (-1, 1); append!(ref, fill(((cy == 1.0 ? 4.010 : 4.030) + sgn * 0.040, cy), 4)); end
        for p in ((5.0, 4.0), (5.2, 5.0), (5.7, 4.3)); append!(ref, fill((p[1] - 3.0, p[2]), 4)); end
        smld = points_smld([ref, tgt])
        for q in (:fft, :singlepass)
            (_, info) = DC.driftcorrect(smld; quality = q, degree = 0)
            @test maximum(abs.(info.model.inter[2].dm .- info.model.inter[1].dm)) < 0.005
        end
    end
end

# -----------------------------------------------------------------------
# Codex review of b2795da. Target triples straddle a site boundary; a wrong CC (3 um)
# covers four target localizations on false reference sites, zero covers all seven.
# Greedy per-candidate thinning scored zero 3 against 4 and `:fft` applied 3 um.
# -----------------------------------------------------------------------
@testset "occupancy: a superset of inliers never loses (Codex P1)" begin
    tgt = Tuple{Float64,Float64}[]
    for δ in (0.0, -0.018, 0.018)
        push!(tgt, (4.010 + δ, 1.0)); push!(tgt, (4.030 + δ, 2.0))
    end
    push!(tgt, (5.0, 3.0))
    ref = [(4.010, 1.0), (4.030, 2.0), (5.0, 3.0)]
    for sgn in (-1, 1), _ in 1:4
        push!(ref, (1.010 + sgn * 0.034, 1.0)); push!(ref, (1.030 + sgn * 0.034, 2.0))
    end
    for order in (identity, reverse)
        smld = points_smld([ref, order(tgt)])
        (_, info) = DC.driftcorrect(smld; quality = :fft, degree = 0)
        @test maximum(abs.(info.model.inter[2].dm .- info.model.inter[1].dm)) < 0.005
    end
end

# Codex P2: the zero incumbent covers 49 sites, a refine ~16 nm away 40, and the real
# entropy cost prefers the biased reference; the 40-site shift must not win.
@testset "entropy cannot buy back lost coverage (Codex P2)" begin
    g = [(1.0 + 0.2 * i, 1.0 + 0.2 * j) for i in 0:6 for j in 0:6]
    ref = Tuple{Float64,Float64}[]
    for p in g[1:40]
        push!(ref, p)
        for _ in 1:10; push!(ref, (p[1] - 0.020, p[2])); end
    end
    for p in g[41:49]; push!(ref, (p[1] + 0.020, p[2])); end
    smld = points_smld([ref, g])
    (_, info) = DC.driftcorrect(smld; quality = :singlepass, degree = 0)
    @test maximum(abs.(info.model.inter[2].dm .- info.model.inter[1].dm)) < 0.005
end

# Codex review of b1dbd6a: with weights restricted to the disagreeing localizations the sums
# shrank and a relative tolerance shrank with them, so a 1e-8 difference decided. Each
# candidate has one exclusive inlier, in a compact group of 10,000 (incumbent) or 10,001 (CC)
# localizations: weights 1e-4 and 0.99990001e-4 on top of 100 sites covered under both shifts
# (the reference holds A and A + 3 um, the target A + 3 um). The tie goes to the CC.
@testset "occupancy tolerance is absolute (Codex b1dbd6a)" begin
    A = [(0.600 + 0.110i, 0.600 + 0.023i^2) for i in 0:99]
    qb = (14.0, 3.0)    # incumbent-only group: one localization at qb, the rest 20 nm away
    qa = (3.0 + 15.0, 5.0)    # CC-only group (at +3 um in x, as A)
    grp(q, n) = vcat([q], fill((q[1] + 0.020, q[2]), n - 1))
    ref = vcat(A, [(p[1] + 3.0, p[2]) for p in A], [(qb[1] - 0.0295, qb[2]), (qa[1] - 3.0 - 0.0295, qa[2])])
    tgt = vcat([(p[1] + 3.0, p[2]) for p in A], grp(qb, 10_000), grp(qa, 10_001))
    smld = points_smld([ref, tgt])
    (_, info) = DC.driftcorrect(smld; quality = :fft, degree = 0)
    @test abs((info.model.inter[2].dm .- info.model.inter[1].dm)[1] - 3.0) < 0.01
end
