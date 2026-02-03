using FourierTools
using SMLMData

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
    if size(ROI, 1) == 4 && ROI[1] >= 0.0
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
    # Zero-pad to 2x size to eliminate cyclic artifacts
    sz1, sz2 = size(im1)
    im1_pad = zeros(T, 2*sz1, 2*sz2)
    im2_pad = zeros(T, 2*sz1, 2*sz2)
    im1_pad[1:sz1, 1:sz2] .= im1
    im2_pad[1:sz1, 1:sz2] .= im2

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
    if smld1.camera.pixel_edges_x[1]   != smld2.camera.pixel_edges_x[1]   &&
       smld1.camera.pixel_edges_x[end] != smld2.camera.pixel_edges_x[end] &&
       smld1.camera.pixel_edges_y[1]   != smld2.camera.pixel_edges_y[1]   &&
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
    peak = argmax(cc)
    # Since the FFT has been centered, the shift is relative to the
    # center of the transformed histogram images.
    # NOTE: Cross-correlation ccorr(A, B) peaks at -δ when B is shifted by +δ
    # relative to A. We want the shift of B relative to A, so we negate.
    if n_dims == 2
        shift = float([mid1 - peak[1], mid2 - peak[2]])  # negated: center - peak
    elseif n_dims == 3
        shift = float([mid1 - peak[1], mid2 - peak[2], mid3 - peak[3]])
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
    if smld1.camera.pixel_edges_x[1]   != smld2.camera.pixel_edges_x[1]   &&
       smld1.camera.pixel_edges_x[end] != smld2.camera.pixel_edges_x[end] &&
       smld1.camera.pixel_edges_y[1]   != smld2.camera.pixel_edges_y[1]   &&
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

    # Find maximum of damped cross-correlation
    # NOTE: Cross-correlation peaks at -δ when B is shifted by +δ, so we negate
    peak = argmax(cc)
    if n_dims == 2
        shift = float([mid1 - peak[1], mid2 - peak[2]])  # negated: center - peak
    else
        mid3 = size(cc, 3) ÷ 2 + 1
        shift = float([mid1 - peak[1], mid2 - peak[2], mid3 - peak[3]])
    end
    shift = histbinsize .* shift

    return shift
end
