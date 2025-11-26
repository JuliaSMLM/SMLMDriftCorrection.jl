# Legendre polynomial drift model implementation
# Uses orthogonal basis functions for better optimization conditioning

using LegendrePolynomials: Pl

"""
    LegendrePoly1D

Univariate Legendre polynomial drift model for a single spatial dimension.
Coefficients are for P_1(t), P_2(t), ..., P_degree(t) where t is normalized to [-1, 1].

Note: P_0 (constant) is NOT included because the inter-dataset shift already
handles global offsets. This matches the convention for standard Polynomial1D
which uses t^1, t^2, ... (no constant term).

The key advantage over standard polynomials: orthogonal basis functions mean
each coefficient captures independent variation, leading to better-conditioned
optimization.
"""
mutable struct LegendrePoly1D <: AbstractIntraDrift1D
    degree::Int
    coefficients::Vector{<:Real}
    n_frames::Int  # needed for time normalization
end

function LegendrePoly1D(degree::Int, n_frames::Int)
    # degree coefficients for P_1 through P_degree (no P_0 constant term)
    return LegendrePoly1D(degree, zeros(degree), n_frames)
end

"""
Normalize frame number to [-1, 1] for Legendre polynomial evaluation.
Frame 1 maps to -1, frame n_frames maps to +1.
"""
function normalize_frame(frame::Int, n_frames::Int)
    return 2 * (frame - 1) / (n_frames - 1) - 1
end

"""
Apply drift to coordinate using Legendre polynomial model.
Uses P_1 through P_degree (no constant P_0 term).
"""
function applydrift(x::AbstractFloat, framenum::Int, p::LegendrePoly1D)
    t = normalize_frame(framenum, p.n_frames)
    for nn = 1:p.degree
        x += p.coefficients[nn] * Pl(t, nn)
    end
    return x
end

"""
Correct drift from coordinate using Legendre polynomial model.
Uses P_1 through P_degree (no constant P_0 term).
"""
function correctdrift(x::AbstractFloat, framenum::Int, p::LegendrePoly1D)
    t = normalize_frame(framenum, p.n_frames)
    for nn = 1:p.degree
        x -= p.coefficients[nn] * Pl(t, nn)
    end
    return x
end

"""
    IntraLegendre

Intra-dataset Legendre drift model - wraps one LegendrePoly1D per spatial dimension.
"""
mutable struct IntraLegendre <: AbstractIntraDrift
    ndims::Int
    dm::Vector{LegendrePoly1D}
end

function IntraLegendre(ndims::Int, n_frames::Int; degree::Int=2)
    dm = Vector{LegendrePoly1D}(undef, ndims)
    for ii = 1:ndims
        dm[ii] = LegendrePoly1D(degree, n_frames)
    end
    return IntraLegendre(ndims, dm)
end

"""
Convert Legendre intra-dataset polynomials to coefficient vector (θ).
Layout: [c1_x, c2_x, ..., c1_y, c2_y, ...]  (coefficients for P_1, P_2, ...)
"""
function intra2theta(p::IntraLegendre)
    n_coeffs = p.dm[1].degree  # degree coefficients (P_1 through P_degree)
    l = p.ndims * n_coeffs
    θ = zeros(Real, l)
    for ii = 1:p.ndims, jj = 1:n_coeffs
        θ[jj + (ii - 1) * n_coeffs] = p.dm[ii].coefficients[jj]
    end
    return θ
end

"""
Convert coefficient vector (θ) to Legendre intra-dataset polynomials.
"""
function theta2intra!(p::IntraLegendre, θ::Vector{<:Real})
    n_coeffs = p.dm[1].degree
    for ii = 1:p.ndims, jj = 1:n_coeffs
        p.dm[ii].coefficients[jj] = θ[jj + (ii - 1) * n_coeffs]
    end
end

"""
Initialize Legendre coefficients with random values.
Called by findintra! before optimization.
"""
function initialize_random!(p::IntraLegendre, rscale::Real, nframes::Int)
    for jj = 1:p.ndims
        n_coeffs = p.dm[jj].degree
        # For Legendre, coefficients don't need frame-scaling
        # because basis functions are already normalized to [-1, 1]
        p.dm[jj].coefficients = rscale * randn(n_coeffs)
    end
end

"""
    LegendrePolynomial

Combined intra + inter Legendre drift model.
Contains one IntraLegendre per dataset plus inter-dataset shifts.
"""
mutable struct LegendrePolynomial <: AbstractIntraInter
    ndatasets::Int
    intra::Vector{IntraLegendre}
    inter::Vector{InterShift}
end

"""
Construct Legendre drift model from SMLD data.

# Arguments
- `smld`: SMLD data structure
- `degree`: polynomial degree (default 2)
- `initialize`: "zeros" (default) or "random"
- `rscale`: scale for random initialization (default 0.01)
"""
function LegendrePolynomial(ndims::Int, ndatasets::Int, nframes::Int;
            degree=2, initialize::String="zeros", rscale=0.01)

    intra = Vector{IntraLegendre}(undef, ndatasets)
    inter = Vector{InterShift}(undef, ndatasets)

    for ii = 1:ndatasets
        intra[ii] = IntraLegendre(ndims, nframes; degree=degree)
        inter[ii] = InterShift(ndims)
    end

    if initialize == "random"
        for ii = 1:ndatasets, jj = 1:ndims
            inter[ii].dm[jj] = rscale * randn()
            # For Legendre polynomials, coefficients don't need frame-scaling
            # because basis functions are already normalized to [-1, 1]
            # Use 'degree' coefficients for P_1 through P_degree (no P_0)
            intra[ii].dm[jj].coefficients = rscale * randn(degree)
        end
    end

    if initialize == "continuous"
        for ii = 1:ndatasets, jj = 1:ndims
            if ii == 1
                inter[ii].dm[jj] = rscale * randn()
            else
                # Chain inter-shifts for continuous drift
                inter[ii].dm[jj] = inter[ii-1].dm[jj] +
                    applydrift(0.0, nframes, intra[ii-1].dm[jj])
            end
            intra[ii].dm[jj].coefficients = rscale * randn(degree)
        end
    end

    return LegendrePolynomial(ndatasets, intra, inter)
end

# Constructor from 2D SMLD
function LegendrePolynomial(smld::SMLD; degree::Int=2, initialize::String="zeros", rscale=0.1)
    return LegendrePolynomial(2, smld.n_datasets, smld.n_frames;
                              degree=degree, initialize=initialize, rscale=rscale)
end

# Constructor from 3D SMLD
function LegendrePolynomial(smld::BasicSMLD{Float64, Emitter3DFit{Float64}};
                            degree::Int=2, initialize::String="zeros", rscale=0.1)
    return LegendrePolynomial(3, smld.n_datasets, smld.n_frames;
                              degree=degree, initialize=initialize, rscale=rscale)
end
