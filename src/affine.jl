"""
Affine (similarity) transform functions for align_smld.
Supports 2D (rotation + uniform scale + translation) and 3D (Euler + scale + translation).
"""

# ============================================================================
# 2D Affine: θ = [θ_rot, scale, tx, ty]
# Transform: [x'; y'] = s * R(θ) * [x; y] + [tx; ty]
# ============================================================================

"""
    apply_affine_2d(params, x, y) -> (x', y')

Apply 2D similarity transform. `params = [θ_rot, scale, tx, ty]`.
"""
function apply_affine_2d(params::AbstractVector, x::Real, y::Real)
    θ, s, tx, ty = params[1], params[2], params[3], params[4]
    cosθ = cos(θ)
    sinθ = sin(θ)
    x_new = s * (cosθ * x - sinθ * y) + tx
    y_new = s * (sinθ * x + cosθ * y) + ty
    return (x_new, y_new)
end

"""
    apply_affine_2d!(x_out, y_out, params, x_in, y_in)

In-place batch 2D affine transform.
"""
function apply_affine_2d!(x_out::Vector{T}, y_out::Vector{T},
                          params::AbstractVector,
                          x_in::Vector{T}, y_in::Vector{T}) where {T<:Real}
    θ, s, tx, ty = params[1], params[2], params[3], params[4]
    cosθ = cos(θ)
    sinθ = sin(θ)
    @inbounds for i in eachindex(x_in)
        x_out[i] = s * (cosθ * x_in[i] - sinθ * y_in[i]) + tx
        y_out[i] = s * (sinθ * x_in[i] + cosθ * y_in[i]) + ty
    end
end

# ============================================================================
# 3D Affine: θ = [θx, θy, θz, scale, tx, ty, tz]
# Uses Euler angles (small-angle regime for SMLM)
# R = Rz(θz) * Ry(θy) * Rx(θx)
# ============================================================================

"""
    apply_affine_3d(params, x, y, z) -> (x', y', z')

Apply 3D similarity transform. `params = [θx, θy, θz, scale, tx, ty, tz]`.
"""
function apply_affine_3d(params::AbstractVector, x::Real, y::Real, z::Real)
    θx, θy, θz, s, tx, ty, tz = params[1], params[2], params[3], params[4], params[5], params[6], params[7]

    # Rx
    cx, sx = cos(θx), sin(θx)
    # Ry
    cy, sy = cos(θy), sin(θy)
    # Rz
    cz, sz = cos(θz), sin(θz)

    # R = Rz * Ry * Rx (standard extrinsic Euler)
    # Row 1
    r11 = cy * cz
    r12 = cz * sx * sy - cx * sz
    r13 = sx * sz + cx * cz * sy
    # Row 2
    r21 = cy * sz
    r22 = cx * cz + sx * sy * sz
    r23 = cx * sy * sz - cz * sx
    # Row 3
    r31 = -sy
    r32 = cy * sx
    r33 = cx * cy

    x_new = s * (r11 * x + r12 * y + r13 * z) + tx
    y_new = s * (r21 * x + r22 * y + r23 * z) + ty
    z_new = s * (r31 * x + r32 * y + r33 * z) + tz
    return (x_new, y_new, z_new)
end

"""
    apply_affine_3d!(x_out, y_out, z_out, params, x_in, y_in, z_in)

In-place batch 3D affine transform.
"""
function apply_affine_3d!(x_out::Vector{T}, y_out::Vector{T}, z_out::Vector{T},
                          params::AbstractVector,
                          x_in::Vector{T}, y_in::Vector{T}, z_in::Vector{T}) where {T<:Real}
    θx, θy, θz, s, tx, ty, tz = params[1], params[2], params[3], params[4], params[5], params[6], params[7]
    cx, sx = cos(θx), sin(θx)
    cy, sy = cos(θy), sin(θy)
    cz, sz = cos(θz), sin(θz)
    r11 = cy * cz;  r12 = cz * sx * sy - cx * sz;  r13 = sx * sz + cx * cz * sy
    r21 = cy * sz;  r22 = cx * cz + sx * sy * sz;  r23 = cx * sy * sz - cz * sx
    r31 = -sy;      r32 = cy * sx;                   r33 = cx * cy
    @inbounds for i in eachindex(x_in)
        x_out[i] = s * (r11 * x_in[i] + r12 * y_in[i] + r13 * z_in[i]) + tx
        y_out[i] = s * (r21 * x_in[i] + r22 * y_in[i] + r23 * z_in[i]) + ty
        z_out[i] = s * (r31 * x_in[i] + r32 * y_in[i] + r33 * z_in[i]) + tz
    end
end

# ============================================================================
# Inverse (correction) transforms: undo a forward affine
# Forward: x' = s * R(θ) * x + t
# Inverse: x = (1/s) * R(-θ) * (x' - t)
# Used in cost functions and post-optimization correction.
# ============================================================================

"""
    correct_affine_2d!(x_out, y_out, params, x_in, y_in)

Apply INVERSE 2D similarity transform (correction direction).
Given forward params [θ, s, tx, ty], computes x = (1/s) * R(-θ) * (x' - t).
"""
function correct_affine_2d!(x_out::Vector{T}, y_out::Vector{T},
                            params::AbstractVector,
                            x_in::Vector{T}, y_in::Vector{T}) where {T<:Real}
    θ, s, tx, ty = params[1], params[2], params[3], params[4]
    inv_s = one(T) / s
    cosθ = cos(θ)
    sinθ = sin(θ)
    @inbounds for i in eachindex(x_in)
        dx = x_in[i] - tx
        dy = y_in[i] - ty
        x_out[i] = inv_s * (cosθ * dx + sinθ * dy)
        y_out[i] = inv_s * (-sinθ * dx + cosθ * dy)
    end
end

"""
    correct_affine_3d!(x_out, y_out, z_out, params, x_in, y_in, z_in)

Apply INVERSE 3D similarity transform (correction direction).
Given forward params [θx, θy, θz, s, tx, ty, tz], computes x = (1/s) * R^T * (x' - t).
"""
function correct_affine_3d!(x_out::Vector{T}, y_out::Vector{T}, z_out::Vector{T},
                            params::AbstractVector,
                            x_in::Vector{T}, y_in::Vector{T}, z_in::Vector{T}) where {T<:Real}
    θx, θy, θz, s, tx, ty, tz = params[1], params[2], params[3], params[4], params[5], params[6], params[7]
    inv_s = one(T) / s
    cx, sx = cos(θx), sin(θx)
    cy, sy = cos(θy), sin(θy)
    cz, sz = cos(θz), sin(θz)
    # Forward R = Rz * Ry * Rx; inverse R^T:
    # R^T row 1 = R col 1
    r11 = cy * cz;           r12 = cy * sz;           r13 = -sy
    r21 = cz * sx * sy - cx * sz;  r22 = cx * cz + sx * sy * sz;  r23 = cy * sx
    r31 = sx * sz + cx * cz * sy;  r32 = cx * sy * sz - cz * sx;  r33 = cx * cy
    @inbounds for i in eachindex(x_in)
        dx = x_in[i] - tx
        dy = y_in[i] - ty
        dz = z_in[i] - tz
        x_out[i] = inv_s * (r11 * dx + r12 * dy + r13 * dz)
        y_out[i] = inv_s * (r21 * dx + r22 * dy + r23 * dz)
        z_out[i] = inv_s * (r31 * dx + r32 * dy + r33 * dz)
    end
end

# ============================================================================
# Apply transform to SMLD (used by align_smld after optimization)
# Applies the INVERSE (correction) — params describe how target differs from reference.
# ============================================================================

"""
    apply_affine_transform!(smld, transform::AffineTransform2D)

Correct 2D affine transform on all emitters (in-place). Applies the inverse
of the stored forward transform to map target coordinates back to reference space.
"""
function apply_affine_transform!(smld, transform::AffineTransform2D)
    θ, s = transform.θ, transform.scale
    tx, ty = transform.tx, transform.ty
    inv_s = 1.0 / s
    cosθ = cos(θ)
    sinθ = sin(θ)
    for nn in eachindex(smld.emitters)
        dx = smld.emitters[nn].x - tx
        dy = smld.emitters[nn].y - ty
        smld.emitters[nn].x = inv_s * (cosθ * dx + sinθ * dy)
        smld.emitters[nn].y = inv_s * (-sinθ * dx + cosθ * dy)
        # Scale uncertainties by inverse scale factor
        smld.emitters[nn].σ_x *= inv_s
        smld.emitters[nn].σ_y *= inv_s
    end
end

"""
    apply_affine_transform!(smld, transform::AffineTransform3D)

Correct 3D affine transform on all emitters (in-place). Applies the inverse
of the stored forward transform to map target coordinates back to reference space.
"""
function apply_affine_transform!(smld, transform::AffineTransform3D)
    θx, θy, θz, s = transform.θx, transform.θy, transform.θz, transform.scale
    tx, ty, tz = transform.tx, transform.ty, transform.tz
    inv_s = 1.0 / s
    cx, sx = cos(θx), sin(θx)
    cy, sy = cos(θy), sin(θy)
    cz, sz = cos(θz), sin(θz)
    # R^T (transpose of forward rotation)
    r11 = cy * cz;           r12 = cy * sz;           r13 = -sy
    r21 = cz * sx * sy - cx * sz;  r22 = cx * cz + sx * sy * sz;  r23 = cy * sx
    r31 = sx * sz + cx * cz * sy;  r32 = cx * sy * sz - cz * sx;  r33 = cx * cy
    for nn in eachindex(smld.emitters)
        dx = smld.emitters[nn].x - tx
        dy = smld.emitters[nn].y - ty
        dz = smld.emitters[nn].z - tz
        smld.emitters[nn].x = inv_s * (r11 * dx + r12 * dy + r13 * dz)
        smld.emitters[nn].y = inv_s * (r21 * dx + r22 * dy + r23 * dz)
        smld.emitters[nn].z = inv_s * (r31 * dx + r32 * dy + r33 * dz)
        smld.emitters[nn].σ_x *= inv_s
        smld.emitters[nn].σ_y *= inv_s
        smld.emitters[nn].σ_z *= inv_s
    end
end

"""
    affine_to_shift(transform::AffineTransform2D) -> Vector{Float64}

Extract translation component from 2D affine transform.
"""
affine_to_shift(t::AffineTransform2D) = [t.tx, t.ty]

"""
    affine_to_shift(transform::AffineTransform3D) -> Vector{Float64}

Extract translation component from 3D affine transform.
"""
affine_to_shift(t::AffineTransform3D) = [t.tx, t.ty, t.tz]

"""
    affine_to_shift(transform::ShiftTransform) -> Vector{Float64}

Extract shift vector from ShiftTransform (identity operation).
"""
affine_to_shift(t::ShiftTransform) = t.shift
