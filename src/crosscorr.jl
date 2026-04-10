using FourierTools
using SMLMData
using LinearAlgebra

"""
    gaussian_subpixel_2d(cc, peak_idx; halfwidth=3)

Refine peak location using 2D Gaussian fit via quadratic surface fitting.
Fits a 2D quadratic to log(intensity) around the peak, which is equivalent
to fitting a 2D Gaussian. Returns subpixel (row, col) coordinates.

Falls back to integer peak if fit fails or window is too small.
"""
function gaussian_subpixel_2d(cc::AbstractMatrix, peak_idx::CartesianIndex; halfwidth::Int=3)
    pi, pj = peak_idx[1], peak_idx[2]
    nr, nc = size(cc)

    # Check bounds - need at least 1 pixel margin for fitting
    if pi <= halfwidth || pi > nr - halfwidth || pj <= halfwidth || pj > nc - halfwidth
        return float(pi), float(pj)
    end

    # Extract window around peak
    i_range = (pi - halfwidth):(pi + halfwidth)
    j_range = (pj - halfwidth):(pj + halfwidth)
    window = cc[i_range, j_range]

    # Shift to ensure positive values for log
    min_val = minimum(window)
    window_shifted = window .- min_val .+ 1.0

    # Take log for Gaussian -> quadratic transformation
    log_window = log.(window_shifted)

    # Build design matrix for 2D quadratic: z = a + b*x + c*y + d*x^2 + e*y^2 + f*x*y
    # where x, y are relative to center of window
    n = 2 * halfwidth + 1
    npts = n * n
    A = zeros(npts, 6)
    z = zeros(npts)

    idx = 1
    for di in -halfwidth:halfwidth
        for dj in -halfwidth:halfwidth
            A[idx, 1] = 1.0
            A[idx, 2] = di
            A[idx, 3] = dj
            A[idx, 4] = di^2
            A[idx, 5] = dj^2
            A[idx, 6] = di * dj
            z[idx] = log_window[di + halfwidth + 1, dj + halfwidth + 1]
            idx += 1
        end
    end

    # Solve least squares: A * coeffs = z
    coeffs = try
        A \ z
    catch
        return float(pi), float(pj)
    end

    # Check for NaN in coefficients
    if any(isnan, coeffs) || any(isinf, coeffs)
        return float(pi), float(pj)
    end

    # coeffs = [a, b, c, d, e, f]
    # For quadratic z = a + b*x + c*y + d*x^2 + e*y^2 + f*x*y
    # Maximum is at: dz/dx = b + 2*d*x + f*y = 0
    #                dz/dy = c + 2*e*y + f*x = 0
    # Solving: [2d  f ] [x]   [-b]
    #          [f   2e] [y] = [-c]

    d, e, f = coeffs[4], coeffs[5], coeffs[6]
    b, c = coeffs[2], coeffs[3]

    # Check that we have a maximum (d < 0 and e < 0 for concave down)
    if d >= 0 || e >= 0
        return float(pi), float(pj)
    end

    # Solve 2x2 system
    det = 4 * d * e - f^2
    if abs(det) < 1e-10
        return float(pi), float(pj)
    end

    x_offset = (f * c - 2 * e * b) / det
    y_offset = (f * b - 2 * d * c) / det

    # Sanity check: offset should be within halfwidth and not NaN
    if isnan(x_offset) || isnan(y_offset) || abs(x_offset) > halfwidth || abs(y_offset) > halfwidth
        return float(pi), float(pj)
    end

    return float(pi) + x_offset, float(pj) + y_offset
end

"""
    gaussian_subpixel_3d(cc, peak_idx; halfwidth=3)

Refine peak location using 3D Gaussian fit via quadratic surface fitting.
Returns subpixel (i, j, k) coordinates.
"""
function gaussian_subpixel_3d(cc::AbstractArray{T,3}, peak_idx::CartesianIndex; halfwidth::Int=2) where T
    pi, pj, pk = peak_idx[1], peak_idx[2], peak_idx[3]
    ni, nj, nk = size(cc)

    # Check bounds
    if pi <= halfwidth || pi > ni - halfwidth ||
       pj <= halfwidth || pj > nj - halfwidth ||
       pk <= halfwidth || pk > nk - halfwidth
        return float(pi), float(pj), float(pk)
    end

    # For 3D, use simpler center-of-mass approach with Gaussian weighting
    # (full 3D quadratic fit has 10 parameters and can be unstable)
    i_range = (pi - halfwidth):(pi + halfwidth)
    j_range = (pj - halfwidth):(pj + halfwidth)
    k_range = (pk - halfwidth):(pk + halfwidth)
    window = cc[i_range, j_range, k_range]

    # Subtract background and threshold
    bg = minimum(window)
    window_bg = max.(window .- bg, 0.0)
    total = sum(window_bg)

    if total < 1e-10
        return float(pi), float(pj), float(pk)
    end

    # Compute center of mass
    ci, cj, ck = 0.0, 0.0, 0.0
    for di in -halfwidth:halfwidth
        for dj in -halfwidth:halfwidth
            for dk in -halfwidth:halfwidth
                w = window_bg[di + halfwidth + 1, dj + halfwidth + 1, dk + halfwidth + 1]
                ci += di * w
                cj += dj * w
                ck += dk * w
            end
        end
    end

    ci /= total
    cj /= total
    ck /= total

    # Sanity check
    if abs(ci) > halfwidth || abs(cj) > halfwidth || abs(ck) > halfwidth
        return float(pi), float(pj), float(pk)
    end

    return float(pi) + ci, float(pj) + cj, float(pk) + ck
end

"""
Produce a histogram image from the localization coordinates x and y.
x and y are in arbitrary units.
ROI is [x_min, x_max, y_min, y_max] of the Region Of Interest in the
same units as x and y.  If not provided, these values are estimated
from the coordinate data.
histbinsize is the size of the bins of each coordinate in the same units.
"""
function histimage2D(x::AbstractVector{T}, y::AbstractVector{T};
    ROI::AbstractVector{T}=[-1.0],
    histbinsize::Union{AbstractVector{T}, T}=1.0
) where {T<:Real}
    if size(ROI, 1) == 4 && ROI != T[-1.0]
        x_min = ROI[1]
        x_max = ROI[2]
        y_min = ROI[3]
        y_max = ROI[4]
    else
        # Find the minimum and maximum values of x and y.
        x_min = floor(minimum(x))
        x_max =  ceil(maximum(x))
        y_min = floor(minimum(y))
        y_max =  ceil(maximum(y))
    end
    lhbs = length(histbinsize)
    if lhbs == 1
         histbinsize = [histbinsize[1], histbinsize[1]]
    elseif lhbs != 2
       error("histbinsize length invalid: $lhbs")
    end
    #println("histimage2D: xy = $x_min, $x_max, $y_min, $y_max") 
    # Compute the number of bins in x and y.
    imszX = round(Int, (x_max .- x_min) ./ histbinsize[1])
    imszY = round(Int, (y_max .- y_min) ./ histbinsize[2])
    #println("histimage2D: imsx = $imszX, imsy = $imszY")
    # Create a blank image.
    im = zeros(Int, imszX, imszY)
    # Convert (x, y) coordinates into bin size units.
    xx = round.(Int, (x .- x_min) ./ histbinsize[1])
    yy = round.(Int, (y .- y_min) ./ histbinsize[2])
    # Exclude points that are outside the image dimensions.
    mask = (xx .> 0) .& (xx .<= imszX) .& (yy .> 0) .& (yy .<= imszY)
    xx = xx[mask]
    yy = yy[mask]
    # Make the histogram image.
    for i in 1:size(xx, 1)
        im[xx[i], yy[i]] += 1
    end
    # Return image.
    return im
end

function histimage2D(x::AbstractMatrix{T}, y::AbstractMatrix{T};
    ROI::AbstractVector{T}=[-1.0],
    histbinsize::Union{AbstractVector{T}, T}=1.0
) where {T<:Real}
    histimage(x[:], y[:]; ROI=ROI, histbinsize=histbinsize)
end

"""
Produce a histogram image from the localization coordinates x, y, z.
x, y, z are in arbitrary units.
ROI is [x_min, x_max, y_min, y_max, z_min, z_max] of the Region Of
Interest in the same units as x, y, z.  If not provided, these values
are estimated from the coordinate data.
histbinsize is the size of the bins of each coordinate in the same units.
"""
function histimage3D(x::AbstractVector{T}, y::AbstractVector{T},
    z::AbstractVector{T};
    ROI::AbstractVector{T}=[-1.0],
    histbinsize::Union{AbstractVector{T}, T}=1.0
) where {T<:Real}
    if size(ROI, 1) == 6 && ROI[1] >= 0.0
        x_min = ROI[1]
        x_max = ROI[2]
        y_min = ROI[3]
        y_max = ROI[4]
        z_min = ROI[5]
        z_max = ROI[6]
    else
        # Find the minimum and maximum values of x, y and z.
        x_min = floor(minimum(x))
        x_max =  ceil(maximum(x))
        y_min = floor(minimum(y))
        y_max =  ceil(maximum(y))
        z_min = floor(minimum(z))
        z_max =  ceil(maximum(z))
    end
    lhbs = length(histbinsize)
    if lhbs == 1
         histbinsize = [histbinsize[1], histbinsize[1], histbinsize[1]]
    elseif lhbs != 3
       error("histbinsize length invalid: $lhbs")
    end
    #println("histimage3D: xyz = $x_min, $x_max, $y_min, $y_max, $z_min, $z_max")
    # Compute the number of bins in x, y and z.
    imszX = round(Int, (x_max .- x_min) ./ histbinsize[1])
    imszY = round(Int, (y_max .- y_min) ./ histbinsize[2])
    imszZ = round(Int, (z_max .- z_min) ./ histbinsize[3])
    #println("histimage3D: imsx = $imszX, imsy = $imszY, imsz = $imszZ")
    # Create a blank image.
    im = zeros(Int, imszX, imszY, imszZ)
    # Convert (x, y, z) coordinates into bin size units.
    xx = round.(Int, (x .- x_min) ./ histbinsize[1])
    yy = round.(Int, (y .- y_min) ./ histbinsize[2])
    zz = round.(Int, (z .- z_min) ./ histbinsize[3])
    # Exclude points that are outside the image dimensions.
    mask = (xx .> 0) .& (xx .<= imszX) .&
           (yy .> 0) .& (yy .<= imszY) .&
           (zz .> 0) .& (zz .<= imszZ)
    xx = xx[mask]
    yy = yy[mask]
    zz = zz[mask]
    # Make the histogram image.
    for i in 1:size(xx, 1)
        im[xx[i], yy[i], zz[i]] += 1
    end
    # Return image.
    return im
end

function histimage3D(x::AbstractMatrix{T}, y::AbstractMatrix{T},
    z::AbstractMatrix{T};
    ROI::AbstractVector{T}=[-1.0],
    histbinsize::Union{AbstractVector{T}, T}=1.0
) where {T<:Real}
    histimage3D(x[:], y[:], z[:]; ROI=ROI, histbinsize=histbinsize)
end

"""
Compute the cross-correlation between two 2D images with zero-padding.

Zero-padding to 2x size eliminates cyclic wrap-around artifacts that cause
false peaks at large shifts.
"""
function crosscorr2D(im1::AbstractMatrix{T}, im2::AbstractMatrix{T}
) where {T<:Real}
    # Pad to common size then 2x to eliminate cyclic artifacts
    sz1 = max(size(im1, 1), size(im2, 1))
    sz2 = max(size(im1, 2), size(im2, 2))
    im1_pad = zeros(T, 2*sz1, 2*sz2)
    im2_pad = zeros(T, 2*sz1, 2*sz2)
    im1_pad[1:size(im1,1), 1:size(im1,2)] .= im1
    im2_pad[1:size(im2,1), 1:size(im2,2)] .= im2

    # Compute the cross-correlation
    cc = FourierTools.ccorr(im1_pad, im2_pad; centered=true)
    return cc
end

"""
Compute the cross-correlation between two 3D images with zero-padding.

Zero-padding to 2x size eliminates cyclic wrap-around artifacts.
"""
function crosscorr3D(im1::AbstractArray{T}, im2::AbstractArray{T}
) where {T<:Real}
    # Zero-pad to 2x size to eliminate cyclic artifacts
    sz1, sz2, sz3 = size(im1)
    im1_pad = zeros(T, 2*sz1, 2*sz2, 2*sz3)
    im2_pad = zeros(T, 2*sz1, 2*sz2, 2*sz3)
    im1_pad[1:sz1, 1:sz2, 1:sz3] .= im1
    im2_pad[1:sz1, 1:sz2, 1:sz3] .= im2

    # Compute the cross-correlation
    cc = FourierTools.ccorr(im1_pad, im2_pad, [1, 2, 3]; centered=true)
    return cc
end

"""
Compute the cross-correlation between two 2D images, weighted by intensity.
"""
function crosscorr2Dweighted(im1::AbstractMatrix{T}, im2::AbstractMatrix{T}
) where {T<:Real}

    # Create a mask of the images (assumed the same size).
    mask = ones(size(im1))
    # Compute the area of the images (assumed the same).
    A = prod(size(mask))
    # Compute the total intensities of the images.    
    N1 = sum(im1)
    N2 = sum(im2)
    # Normalization.
    NP = real(fftshift2d(ifft2d(abs.(fft2d(mask)) .^ 2)))
    # Compute the Fourier transforms of the images.
    F1 = fft2d(im1 .* mask)
    F2 = fft2d(im2 .* mask)
    # Compute the cross-correlation.
    cc = A^2 / (N1 * N2) .* real(fftshift2d(ifft2d(F1 .* conj(F2)))) ./ NP
    # Return the cross-correlation.
    return cc
end

"""
Perform a cross-correlation between images representing localizations in two
SMLD structures and compute the shift between the two original images.
histbinsize is the size of the bins in the histogram image in the
same units as the localization coordinates.
"""
function findshift(smld1::T, smld2::T;
    histbinsize::Real=1.0
) where {T<:SMLD}

    # Check for empty datasets
    if isempty(smld1.emitters)
        error("findshift: smld1 has no emitters")
    end
    if isempty(smld2.emitters)
        error("findshift: smld2 has no emitters")
    end

    n_dims = nDims(smld1)

    # Convert histbinsize to match coordinate type
    coord_type = typeof(smld1.emitters[1].x)
    histbinsize = coord_type(histbinsize)

    # Compute the histogram images (assume the same size for both images).
    if smld1.camera.pixel_edges_x[1]   != smld2.camera.pixel_edges_x[1]   ||
       smld1.camera.pixel_edges_x[end] != smld2.camera.pixel_edges_x[end] ||
       smld1.camera.pixel_edges_y[1]   != smld2.camera.pixel_edges_y[1]   ||
       smld1.camera.pixel_edges_y[end] != smld2.camera.pixel_edges_y[end]
        error("Images must have the same size.")
    end
    if n_dims == 2
        ROI = float([smld1.camera.pixel_edges_x[1],
                     smld1.camera.pixel_edges_x[end],
                     smld1.camera.pixel_edges_y[1],
                     smld1.camera.pixel_edges_y[end]])
    elseif n_dims == 3
        smld1_z = [e.z for e in smld1.emitters]
        ROI = float([smld1.camera.pixel_edges_x[1],
                     smld1.camera.pixel_edges_x[end],
                     smld1.camera.pixel_edges_y[1],
                     smld1.camera.pixel_edges_y[end],
                     round(minimum(smld1_z)),
                     round(maximum(smld1_z))])
    end
    imsz_x = smld1.camera.pixel_edges_x[end] - smld1.camera.pixel_edges_x[1]
    imsz_y = smld1.camera.pixel_edges_y[end] - smld1.camera.pixel_edges_y[1]
    if n_dims == 3
        imsz_z = maximum(smld1_z) - minimum(smld1_z)
    end
    smld1_x = [e.x for e in smld1.emitters]
    smld1_y = [e.y for e in smld1.emitters]
    smld2_x = [e.x for e in smld2.emitters]
    smld2_y = [e.y for e in smld2.emitters]
    if n_dims == 2
        im1 = histimage2D(smld1_x, smld1_y; ROI=ROI, histbinsize=histbinsize)
        im2 = histimage2D(smld2_x, smld2_y; ROI=ROI, histbinsize=histbinsize)
    elseif n_dims == 3
        smld1_z = [e.z for e in smld1.emitters]
        smld2_z = [e.z for e in smld2.emitters]
        im1 = histimage3D(smld1_x, smld1_y, smld1_z;
                          ROI=ROI, histbinsize=histbinsize)
        im2 = histimage3D(smld2_x, smld2_y, smld2_z;
                          ROI=ROI, histbinsize=histbinsize)
    end
    # Compute the cross-correlation (with zero-padding)
    if n_dims == 2
        cc = crosscorr2D(im1, im2)
    elseif n_dims == 3
        cc = crosscorr3D(im1, im2)
    end

    # Calculate the center of the (padded) cross-correlation output
    # This is where shift=0 should appear
    if mod(size(cc, 1), 2) == 0
        mid1 = size(cc, 1) / 2 + 1
    else
        mid1 = (size(cc, 1) + 1) / 2
    end
    if mod(size(cc, 2), 2) == 0
        mid2 = size(cc, 2) / 2 + 1
    else
        mid2 = (size(cc, 2) + 1) / 2
    end
    if n_dims == 3
        if mod(size(cc, 3), 2) == 0
            mid3 = size(cc, 3) / 2 + 1
        else
            mid3 = (size(cc, 3) + 1) / 2
        end
    end
    # Find the maximum location in the cross-correlation, which will
    # correspond to the shift between the two images.
    peak_idx = argmax(cc)

    # Refine peak location with subpixel Gaussian fitting
    # This fits a 2D Gaussian (via quadratic in log-space) to improve accuracy
    # FourierTools.ccorr(A, B) peaks at -δ when B is shifted by +δ relative to A.
    # So to get the true shift of B relative to A, we compute: center - peak
    if n_dims == 2
        peak_i, peak_j = gaussian_subpixel_2d(cc, peak_idx; halfwidth=3)
        shift = float([mid1 - peak_i, mid2 - peak_j])
    elseif n_dims == 3
        peak_i, peak_j, peak_k = gaussian_subpixel_3d(cc, peak_idx; halfwidth=2)
        shift = float([mid1 - peak_i, mid2 - peak_j, mid3 - peak_k])
    end
    # Convert the shift to an (x, y {, z}) coordinate.
    shift = histbinsize .* shift
    # Return the shift.
    return shift
end

"""
    findshift_damped(smld1, smld2; histbinsize, prior_shift, prior_sigma)

Find shift with Gaussian damping centered at prior_shift.
Used to refine outlier shifts by searching near expected location.
"""
function findshift_damped(smld1::T, smld2::T;
    histbinsize::Real=1.0,
    prior_shift::Vector{<:Real}=[0.0, 0.0],
    prior_sigma::Real=1.0
) where {T<:SMLD}

    # Check for empty datasets
    if isempty(smld1.emitters)
        error("findshift_damped: smld1 has no emitters")
    end
    if isempty(smld2.emitters)
        error("findshift_damped: smld2 has no emitters")
    end

    n_dims = nDims(smld1)

    # Convert histbinsize to match coordinate type
    coord_type = typeof(smld1.emitters[1].x)
    histbinsize = coord_type(histbinsize)

    # Build histogram images (same as findshift)
    if smld1.camera.pixel_edges_x[1]   != smld2.camera.pixel_edges_x[1]   ||
       smld1.camera.pixel_edges_x[end] != smld2.camera.pixel_edges_x[end] ||
       smld1.camera.pixel_edges_y[1]   != smld2.camera.pixel_edges_y[1]   ||
       smld1.camera.pixel_edges_y[end] != smld2.camera.pixel_edges_y[end]
        error("Images must have the same size.")
    end

    if n_dims == 2
        ROI = float([smld1.camera.pixel_edges_x[1],
                     smld1.camera.pixel_edges_x[end],
                     smld1.camera.pixel_edges_y[1],
                     smld1.camera.pixel_edges_y[end]])
    elseif n_dims == 3
        smld1_z = [e.z for e in smld1.emitters]
        ROI = float([smld1.camera.pixel_edges_x[1],
                     smld1.camera.pixel_edges_x[end],
                     smld1.camera.pixel_edges_y[1],
                     smld1.camera.pixel_edges_y[end],
                     round(minimum(smld1_z)),
                     round(maximum(smld1_z))])
    end

    smld1_x = [e.x for e in smld1.emitters]
    smld1_y = [e.y for e in smld1.emitters]
    smld2_x = [e.x for e in smld2.emitters]
    smld2_y = [e.y for e in smld2.emitters]

    if n_dims == 2
        im1 = histimage2D(smld1_x, smld1_y; ROI=ROI, histbinsize=histbinsize)
        im2 = histimage2D(smld2_x, smld2_y; ROI=ROI, histbinsize=histbinsize)
        cc = crosscorr2D(im1, im2)
    elseif n_dims == 3
        smld1_z = [e.z for e in smld1.emitters]
        smld2_z = [e.z for e in smld2.emitters]
        im1 = histimage3D(smld1_x, smld1_y, smld1_z; ROI=ROI, histbinsize=histbinsize)
        im2 = histimage3D(smld2_x, smld2_y, smld2_z; ROI=ROI, histbinsize=histbinsize)
        cc = crosscorr3D(im1, im2)
    end

    # Calculate center of cross-correlation
    mid1 = size(cc, 1) ÷ 2 + 1
    mid2 = size(cc, 2) ÷ 2 + 1

    # Convert prior_shift to pixel coordinates
    prior_px = prior_shift ./ histbinsize

    # Apply Gaussian damping centered at prior_shift
    σ_px = prior_sigma / histbinsize
    if n_dims == 2
        for j in 1:size(cc, 2), i in 1:size(cc, 1)
            di = (i - mid1) - prior_px[1]
            dj = (j - mid2) - prior_px[2]
            weight = exp(-(di^2 + dj^2) / (2 * σ_px^2))
            cc[i, j] *= weight
        end
    else  # 3D
        mid3 = size(cc, 3) ÷ 2 + 1
        for k in 1:size(cc, 3), j in 1:size(cc, 2), i in 1:size(cc, 1)
            di = (i - mid1) - prior_px[1]
            dj = (j - mid2) - prior_px[2]
            dk = (k - mid3) - (length(prior_shift) > 2 ? prior_px[3] : 0.0)
            weight = exp(-(di^2 + dj^2 + dk^2) / (2 * σ_px^2))
            cc[i, j, k] *= weight
        end
    end

    # Find maximum of damped cross-correlation with subpixel refinement
    # FourierTools.ccorr peaks at -δ when B is shifted by +δ, so shift = center - peak
    peak_idx = argmax(cc)
    if n_dims == 2
        peak_i, peak_j = gaussian_subpixel_2d(cc, peak_idx; halfwidth=3)
        shift = float([mid1 - peak_i, mid2 - peak_j])
    else
        mid3 = size(cc, 3) ÷ 2 + 1
        peak_i, peak_j, peak_k = gaussian_subpixel_3d(cc, peak_idx; halfwidth=2)
        shift = float([mid1 - peak_i, mid2 - peak_j, mid3 - peak_k])
    end
    shift = histbinsize .* shift

    return shift
end

# ============================================================================
# Shift-field affine: sub-region CC shifts + least-squares affine fit
# ============================================================================

"""
    find_affine_shift_field(smld1, smld2; histbinsize=0.05, min_locs=200,
                            n_tiles_target=30, tile_range=(1.0, 5.0)) -> (a,b,c,d,e,f)

Recover a full 2D affine transform between two SMLDs using local
cross-correlation shifts measured on a grid of sub-regions.

Algorithm:
1. Global CC shift to coarsely align
2. Compute overlap density from histograms to find regions with signal in both channels
3. Select spatially distributed tiles with sufficient localizations
4. CC each tile to measure local shifts
5. Robust least-squares fit: Δx = a*x + b*y + c,  Δy = d*x + e*y + f

Returns a NamedTuple `(a, b, c, d, e, f, global_shift, n_tiles, rms_residual)`.
The full correction for a point (x,y) after global shift is:
  x_corrected = x - (a*x + b*y + c)
  y_corrected = y - (d*x + e*y + f)
"""
function find_affine_shift_field(smld1::S, smld2::S;
        histbinsize::Real=0.05,
        min_locs::Int=200,
        n_tiles_target::Int=30,
        tile_range::Tuple{Float64,Float64}=(1.0, 5.0)) where {S<:SMLD}

    nDims(smld1) == 2 || error("find_affine_shift_field: only 2D supported")

    # --- Step 1: global CC shift ---
    global_shift = findshift(smld1, smld2; histbinsize=histbinsize)
    smld2_shifted = deepcopy(smld2)
    correctdrift!(smld2_shifted, global_shift)

    # --- Step 2: adaptive tile size ---
    x1 = [e.x for e in smld1.emitters]; y1 = [e.y for e in smld1.emitters]
    x2 = [e.x for e in smld2_shifted.emitters]; y2 = [e.y for e in smld2_shifted.emitters]

    fov_xmin = max(minimum(x1), minimum(x2))
    fov_xmax = min(maximum(x1), maximum(x2))
    fov_ymin = max(minimum(y1), minimum(y2))
    fov_ymax = min(maximum(y1), maximum(y2))
    fov_area = (fov_xmax - fov_xmin) * (fov_ymax - fov_ymin)

    # Target ~min_locs per tile per channel
    density_1 = length(x1) / fov_area
    density_2 = length(x2) / fov_area
    min_density = min(density_1, density_2)
    tile_side = clamp(sqrt(min_locs / min_density), tile_range[1], tile_range[2])

    # --- Step 3: generate tile grid and score by overlap ---
    nx_tiles = max(1, floor(Int, (fov_xmax - fov_xmin) / tile_side))
    ny_tiles = max(1, floor(Int, (fov_ymax - fov_ymin) / tile_side))

    # Histogram for overlap scoring (coarser bins for speed)
    score_binsize = max(histbinsize, tile_side / 20)
    ROI = float([fov_xmin, fov_xmax, fov_ymin, fov_ymax])
    im1 = histimage2D(x1, y1; ROI=ROI, histbinsize=Float64(score_binsize))
    im2 = histimage2D(x2, y2; ROI=ROI, histbinsize=Float64(score_binsize))
    # Pad to common size
    sz = (max(size(im1, 1), size(im2, 1)), max(size(im1, 2), size(im2, 2)))
    im1_p = zeros(eltype(im1), sz); im1_p[1:size(im1,1), 1:size(im1,2)] .= im1
    im2_p = zeros(eltype(im2), sz); im2_p[1:size(im2,1), 1:size(im2,2)] .= im2
    overlap = sqrt.(Float64.(im1_p) .* Float64.(im2_p))

    # Score each tile
    tile_info = Vector{NamedTuple{(:cx,:cy,:xr,:yr,:score,:n1,:n2), Tuple{Float64,Float64,Tuple{Float64,Float64},Tuple{Float64,Float64},Float64,Int,Int}}}()
    for ix in 0:nx_tiles-1, iy in 0:ny_tiles-1
        xlo = fov_xmin + ix * tile_side
        xhi = xlo + tile_side
        ylo = fov_ymin + iy * tile_side
        yhi = ylo + tile_side

        # Count locs in each channel
        n1 = count(i -> x1[i] >= xlo && x1[i] < xhi && y1[i] >= ylo && y1[i] < yhi, eachindex(x1))
        n2 = count(i -> x2[i] >= xlo && x2[i] < xhi && y2[i] >= ylo && y2[i] < yhi, eachindex(x2))

        # Overlap score from histogram
        px_lo_x = max(1, round(Int, (xlo - fov_xmin) / score_binsize) + 1)
        px_hi_x = min(sz[1], round(Int, (xhi - fov_xmin) / score_binsize))
        px_lo_y = max(1, round(Int, (ylo - fov_ymin) / score_binsize) + 1)
        px_hi_y = min(sz[2], round(Int, (yhi - fov_ymin) / score_binsize))
        score = (px_hi_x >= px_lo_x && px_hi_y >= px_lo_y) ?
                sum(view(overlap, px_lo_x:px_hi_x, px_lo_y:px_hi_y)) : 0.0

        push!(tile_info, (cx=(xlo+xhi)/2, cy=(ylo+yhi)/2,
                          xr=(xlo, xhi), yr=(ylo, yhi),
                          score=score, n1=n1, n2=n2))
    end

    # Filter by minimum locs and sort by score
    valid = filter(t -> t.n1 >= min_locs && t.n2 >= min_locs, tile_info)
    sort!(valid, by=t -> -t.score)

    # --- Step 4: select spatially distributed tiles ---
    selected = _select_distributed_tiles(valid, n_tiles_target)

    length(selected) < 4 && error("find_affine_shift_field: only $(length(selected)) valid tiles (need ≥4)")

    # --- Step 5: CC each tile ---
    pixelsize = smld1.camera.pixel_edges_x[2] - smld1.camera.pixel_edges_x[1]
    tile_cx = Float64[]; tile_cy = Float64[]
    tile_dx = Float64[]; tile_dy = Float64[]

    for t in selected
        # Build sub-camera for this tile
        xr, yr = t.xr, t.yr
        sub_nx = round(Int, (xr[2] - xr[1]) / pixelsize) + 2
        sub_ny = round(Int, (yr[2] - yr[1]) / pixelsize) + 2
        sub_cam = IdealCamera(
            collect(range(xr[1], step=pixelsize, length=sub_nx + 1)),
            collect(range(yr[1], step=pixelsize, length=sub_ny + 1)))

        mask1 = [(e.x >= xr[1] && e.x < xr[2] && e.y >= yr[1] && e.y < yr[2]) for e in smld1.emitters]
        mask2 = [(e.x >= xr[1] && e.x < xr[2] && e.y >= yr[1] && e.y < yr[2]) for e in smld2_shifted.emitters]
        sub1 = BasicSMLD(smld1.emitters[mask1], sub_cam, smld1.n_frames, 1)
        sub2 = BasicSMLD(smld2_shifted.emitters[mask2], sub_cam, smld2_shifted.n_frames, 1)

        shift_local = try
            findshift(sub1, sub2; histbinsize=histbinsize)
        catch
            continue
        end

        push!(tile_cx, t.cx); push!(tile_cy, t.cy)
        push!(tile_dx, shift_local[1]); push!(tile_dy, shift_local[2])
    end

    n_measured = length(tile_dx)
    n_measured < 4 && error("find_affine_shift_field: only $n_measured successful tile shifts (need ≥4)")

    # --- Step 6: robust affine fit with outlier rejection ---
    a, b, c, d, e, f, rms = _robust_affine_fit(tile_cx, tile_cy, tile_dx, tile_dy)

    return (a=a, b=b, c=c, d=d, e=e, f=f,
            global_shift=global_shift, n_tiles=n_measured, rms_residual=rms)
end

"""
Select spatially distributed tiles: greedy farthest-point sampling.
"""
function _select_distributed_tiles(candidates, n_target)
    isempty(candidates) && return typeof(candidates)()
    n_target = min(n_target, length(candidates))

    selected_idx = [1]  # start with highest-scored tile

    for _ in 2:n_target
        best_dist = -1.0
        best_idx = -1
        for (j, t) in enumerate(candidates)
            j in selected_idx && continue
            # Min distance to any already-selected tile
            min_d = minimum(sqrt((t.cx - candidates[k].cx)^2 + (t.cy - candidates[k].cy)^2)
                           for k in selected_idx)
            # Weight by score to prefer dense tiles
            weighted = min_d * sqrt(t.score + 1)
            if weighted > best_dist
                best_dist = weighted
                best_idx = j
            end
        end
        best_idx == -1 && break
        push!(selected_idx, best_idx)
    end

    return [candidates[i] for i in selected_idx]
end

"""
Robust affine fit with iterative outlier rejection.
Fits Δx = a*x + b*y + c, Δy = d*x + e*y + f.
"""
function _robust_affine_fit(cx, cy, dx, dy; max_iter=3, outlier_sigma=3.0)
    N = length(cx)
    mask = trues(N)

    a, b, c, d, e, f = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    rms = Inf

    for iter in 1:max_iter
        idx = findall(mask)
        n = length(idx)
        n < 4 && break

        A = zeros(2n, 6)
        bv = zeros(2n)
        for (ki, i) in enumerate(idx)
            A[ki, 1] = cx[i]; A[ki, 2] = cy[i]; A[ki, 3] = 1.0
            A[n+ki, 4] = cx[i]; A[n+ki, 5] = cy[i]; A[n+ki, 6] = 1.0
            bv[ki] = dx[i]; bv[n+ki] = dy[i]
        end

        params = A \ bv
        a, b, c, d, e, f = params

        # Compute residuals
        res_x = [dx[i] - (a*cx[i] + b*cy[i] + c) for i in 1:N]
        res_y = [dy[i] - (d*cx[i] + e*cy[i] + f) for i in 1:N]
        res_mag = sqrt.(res_x.^2 .+ res_y.^2)

        rms = sqrt(mean(res_mag[mask].^2))

        # Reject outliers
        if iter < max_iter
            threshold = outlier_sigma * rms
            for i in 1:N
                if mask[i] && res_mag[i] > threshold
                    mask[i] = false
                end
            end
        end
    end

    return a, b, c, d, e, f, rms
end

# ============================================================================
# Fourier-Mellin: recover rotation + scale + shift via FFT
# ============================================================================

"""
    _logpolar_transform(mag::Matrix, n_angles, n_radii; r_min_frac=0.02, r_max_frac=0.45)

Reproject a centered magnitude spectrum to log-polar coordinates.
Returns a (n_angles × n_radii) matrix where:
- Row index = angle (0 to 2π)
- Col index = log(radius) from r_min to r_max

Uses bilinear interpolation for subpixel sampling.
"""
function _logpolar_transform(mag::Matrix{T}, n_angles::Int, n_radii::Int;
                              r_min_frac::Float64=0.02, r_max_frac::Float64=0.45) where {T<:Real}
    nr, nc = size(mag)
    cy, cx = nr ÷ 2 + 1, nc ÷ 2 + 1
    r_max = r_max_frac * min(nr, nc)
    r_min = max(r_min_frac * min(nr, nc), 2.0)
    log_r_min = log(r_min)
    log_r_max = log(r_max)

    lp = zeros(T, n_angles, n_radii)

    @inbounds for ai in 1:n_angles
        θ = 2π * (ai - 1) / n_angles
        cosθ = cos(θ)
        sinθ = sin(θ)
        for ri in 1:n_radii
            r = exp(log_r_min + (ri - 1) * (log_r_max - log_r_min) / (n_radii - 1))
            # Sample point in centered spectrum
            fy = cy + r * cosθ
            fx = cx + r * sinθ

            # Bilinear interpolation
            iy = floor(Int, fy)
            ix = floor(Int, fx)
            if iy >= 1 && iy < nr && ix >= 1 && ix < nc
                dy = fy - iy
                dx = fx - ix
                lp[ai, ri] = (1 - dy) * (1 - dx) * mag[iy, ix] +
                             dy * (1 - dx) * mag[iy+1, ix] +
                             (1 - dy) * dx * mag[iy, ix+1] +
                             dy * dx * mag[iy+1, ix+1]
            end
        end
    end
    return lp
end

"""
    _highpass_filter!(mag, sigma_frac=0.05)

Apply a Gaussian high-pass filter to suppress the DC/low-frequency peak
in the magnitude spectrum that would dominate the log-polar CC.
"""
function _highpass_filter!(mag::Matrix{T}; sigma_frac::Float64=0.05) where {T<:Real}
    nr, nc = size(mag)
    cy, cx = nr ÷ 2 + 1, nc ÷ 2 + 1
    σ = sigma_frac * min(nr, nc)
    σ2 = 2 * σ^2
    @inbounds for j in 1:nc, i in 1:nr
        r2 = (i - cy)^2 + (j - cx)^2
        mag[i, j] *= (1 - exp(-r2 / σ2))
    end
end

"""
    find_affine_fft(smld1, smld2; histbinsize=0.05, max_rotation=10.0) -> AffineTransform2D

Recover 2D similarity transform (rotation + scale + translation) between two
SMLDs using the Fourier-Mellin approach:

1. Histogram both datasets → 2D images
2. FFT → magnitude spectra (translation-invariant)
3. Log-polar reprojection of magnitude spectra
4. Cross-correlate log-polar images → rotation and scale
5. De-rotate/de-scale dataset2, cross-correlate → translation

Returns an `AffineTransform2D(θ, scale, tx, ty)`.

This is a one-pass FFT method — fast (< 1s for any dataset size) but less
accurate than entropy-based refinement. Use as initial guess for entropy
optimization or as standalone fast alignment.
"""
function find_affine_fft(smld1::S, smld2::S;
                          histbinsize::Real=0.05,
                          max_rotation::Real=10.0) where {S<:SMLD}

    isempty(smld1.emitters) && error("find_affine_fft: smld1 has no emitters")
    isempty(smld2.emitters) && error("find_affine_fft: smld2 has no emitters")
    nDims(smld1) == 2 || error("find_affine_fft: only 2D supported")

    coord_type = typeof(smld1.emitters[1].x)
    hbs = coord_type(histbinsize)

    # Step 1: Build histogram images on shared ROI
    ROI = float([smld1.camera.pixel_edges_x[1],
                 smld1.camera.pixel_edges_x[end],
                 smld1.camera.pixel_edges_y[1],
                 smld1.camera.pixel_edges_y[end]])

    x1 = [e.x for e in smld1.emitters]
    y1 = [e.y for e in smld1.emitters]
    x2 = [e.x for e in smld2.emitters]
    y2 = [e.y for e in smld2.emitters]

    im1 = Float64.(histimage2D(x1, y1; ROI=ROI, histbinsize=hbs))
    im2 = Float64.(histimage2D(x2, y2; ROI=ROI, histbinsize=hbs))

    # Step 2: FFT → magnitude spectra (translation-invariant)
    F1 = fftshift(fft(im1))
    F2 = fftshift(fft(im2))
    mag1 = abs.(F1)
    mag2 = abs.(F2)

    # High-pass filter to suppress DC peak
    _highpass_filter!(mag1)
    _highpass_filter!(mag2)

    # Step 3: Log-polar reprojection
    n_angles = 360
    n_radii = 128
    lp1 = _logpolar_transform(mag1, n_angles, n_radii)
    lp2 = _logpolar_transform(mag2, n_angles, n_radii)

    # Step 4: Cross-correlate log-polar images → (Δangle, Δlog_radius)
    cc_lp = crosscorr2D(lp1, lp2)

    # Mask to max_rotation range: only consider angles within ±max_rotation°
    max_angle_px = round(Int, max_rotation / (360.0 / n_angles))
    cc_mid1 = size(cc_lp, 1) ÷ 2 + 1
    cc_mid2 = size(cc_lp, 2) ÷ 2 + 1
    # Zero out rows outside ±max_rotation (angle is row axis)
    for i in 1:size(cc_lp, 1)
        if abs(i - cc_mid1) > max_angle_px
            cc_lp[i, :] .= 0.0
        end
    end

    peak_idx = argmax(cc_lp)
    peak_i, peak_j = gaussian_subpixel_2d(cc_lp, peak_idx; halfwidth=3)

    # Convert peak to rotation and scale
    Δangle_px = cc_mid1 - peak_i
    Δlogr_px = cc_mid2 - peak_j

    # CC convention: peak offset = negative of the transform angle (same as shift CC)
    θ_recovered = -Δangle_px * (2π / n_angles)

    # Scale from log-radius shift
    nr, nc = size(mag1)
    r_max = 0.45 * min(nr, nc)
    r_min = max(0.02 * min(nr, nc), 2.0)
    log_r_step = (log(r_max) - log(r_min)) / (n_radii - 1)
    # FM recovers Fourier-space scale; spatial scale is the reciprocal
    scale_recovered = exp(-Δlogr_px * log_r_step)

    # Step 5: Compute translation from centroids
    # Forward model: x2 = s * R(θ) * x1 + t
    # Centroids satisfy: c2 = s * R(θ) * c1 + t
    # Therefore: t = c2 - s * R(θ) * c1
    c1x, c1y = mean(x1), mean(y1)
    c2x, c2y = mean(x2), mean(y2)

    s = scale_recovered
    cosθr = cos(θ_recovered)
    sinθr = sin(θ_recovered)
    tx = c2x - s * (cosθr * c1x - sinθr * c1y)
    ty = c2y - s * (sinθr * c1x + cosθr * c1y)

    return AffineTransform2D(θ_recovered, scale_recovered, tx, ty)
end
