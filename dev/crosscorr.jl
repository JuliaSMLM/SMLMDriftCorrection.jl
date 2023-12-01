using Revise
using SMLMData
using FourierTools

"""
Produce a histogram image from the localization coordinates x and y,
x and y are in arbitrary units.
ROI is [x_min, x_max, y_min, y_max] of the Region Of Interest in the
same units as x and y.  If not provided, these values are estimated
from the coordinate data.
histbinsize is the size of the bins in the same units.
"""
function histimage2D(x::AbstractVector{T}, y::AbstractVector{T};
    ROI::AbstractVector{T}=[-1.0], histbinsize::T=1.0,
) where {T<:Real}
    if size(ROI, 1) == 4 && ROI[1] >= 0.0
        x_min = ROI[1]
        x_max = ROI[2]
        y_min = ROI[3]
        y_max = ROI[4]
    else
        # Find the minimum and maximum values of x and y.
        x_min = floor(minimum(x))
        x_max = ceil(maximum(x))
        y_min = floor(minimum(y))
        y_max = ceil(maximum(y))
    end
    #println("histimage2D: $x_min, $x_max, $y_min, $y_max") 
    # Compute the number of pixels in x and y.
    imszX = round(Int, (x_max .- x_min) ./ histbinsize)
    imszY = round(Int, (y_max .- y_min) ./ histbinsize)
    # Make sure the number of pixels in each coordinate is odd.
    if mod(imszX, 2) == 0
        imszX += 1
    end
    if mod(imszY, 2) == 0
        imszY += 1
    end
    println("histimage2D: imsx = $imszX, imsy = $imszY")
    # Create a blank image.
    im = zeros(Int, imszX, imszY)
    # Convert (x, y) coordinates into bin size units.
    xx = round.(Int, (x .- x_min) ./ histbinsize)
    yy = round.(Int, (y .- y_min) ./ histbinsize)
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
    ROI::AbstractVector{T}=[-1.0], histbinsize::T=1.0,
) where {T<:Real}
    histimage(x[:], y[:]; ROI=ROI, histbinsize=histbinsize)
end

"""
Produce a histogram image from the localization coordinates x, y, z.
x, y, z are in arbitrary units.
ROI is [x_min, x_max, y_min, y_max, z_min, z_max] of the Region Of
Interest in the same units as x, y, z.  If not provided, these values
are estimated from the coordinate data.
histbinsize is the size of the bins in the same units.
"""
function histimage3D(x::AbstractVector{T}, y::AbstractVector{T},
    z::AbstractVector{T};
    ROI::AbstractVector{T}=[-1.0], histbinsize::T=1.0,
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
        x_max = ceil(maximum(x))
        y_min = floor(minimum(y))
        y_max = ceil(maximum(y))
        z_min = floor(minimum(z))
        z_max = ceil(maximum(z))
    end
    # Compute the number of pixels in x, y and z.
    imszX = round(Int, (x_max .- x_min) ./ histbinsize)
    imszY = round(Int, (y_max .- y_min) ./ histbinsize)
    imszZ = round(Int, (z_max .- z_min) ./ histbinsize)
    # Make sure the number of pixels in each coordinate is odd.
    if mod(imszX, 2) == 0
        imszX += 1
    end
    if mod(imszY, 2) == 0
        imszY += 1
    end
    if mod(imszZ, 2) == 0
        imszZ += 1
    end
    println("histimage3D: imsx = $imszX, imsy = $imszY, imsz = $imszZ")
    # Create a blank image.
    im = zeros(Int, imszX, imszY, imszZ)
    # Convert (x, y, z) coordinates into bin size units.
    xx = round.(Int, (x .- x_min) ./ histbinsize)
    yy = round.(Int, (y .- y_min) ./ histbinsize)
    zz = round.(Int, (z .- z_min) ./ histbinsize)
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
    ROI::AbstractVector{T}=[-1.0], histbinsize::T=1.0,
) where {T<:Real}
    histimage3D(x[:], y[:], z[:]; ROI=ROI, histbinsize=histbinsize)
end

"""
Compute the cross-correlation between two 2D images.
"""
function crosscorr2D(im1::AbstractMatrix{T}, im2::AbstractMatrix{T}
) where {T<:Real}
    # Compute the cross-correlation.
    cc = FourierTools.ccorr(im1, im2; centered=true)
    # Return the cross-correlation.
    return cc
end

"""
Compute the cross-correlation between two 3D images.
"""
function crosscorr3D(im1::AbstractArray{T}, im2::AbstractArray{T}
) where {T<:Real}
    # Compute the cross-correlation.
    cc = FourierTools.ccorr(im1, im2, [1, 2, 3]; centered=true)
    # Return the cross-correlation.
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
    #NP = real(ifft2d(abs.(fft2d(mask)).^2))
    # Compute the Fourier transforms of the images.
    F1 = fft2d(im1 .* mask)
    F2 = fft2d(im2 .* mask)
    # Compute the cross-correlation.
    cc = A^2 / (N1 * N2) .* real(fftshift2d(ifft2d(F1 .* conj(F2)))) ./ NP
    #cc = A^2/(N1*N2) .* real(ifft2d(F1 .* conj(F2))) ./ NP
    # Return the cross-correlation.
    return cc
end

"""
Perform a cross-correlation between images representing
localizations in two SMLD structures and compute the shift
between the two original images.
histbinsize is the size of the bins in the histogram image in the
same units as the localization coordinates.
"""
function findshift2D(smld1::T, smld2::T; histbinsize::AbstractFloat=1.0
) where {T<:SMLMData.SMLD}
    # Compute the histogram images assume the same size for both images).
    if smld1.datasize[1] != smld2.datasize[1] &&
       smld1.datasize[2] != smld2.datasize[2]
        error("Images must have the same size.")
    end
    if smld1.datasize[1] == 0.0 || smld1.datasize[2] == 0.0
        println("findshift2D: smld.datasize(s) are zero.")
        ROI = [-1.0]
    else
        ROI = float([0, smld1.datasize[1], 0, smld1.datasize[2]])
    end
    im1 = histimage2D(smld1.x, smld1.y; ROI=ROI, histbinsize=histbinsize)
    im2 = histimage2D(smld2.x, smld2.y; ROI=ROI, histbinsize=histbinsize)
    # Determine the midpoints of the histogram images.
    # calculate the FFT center zero frequency index of im
    if mod(size(im1, 1), 2) == 0
        mid1 = size(im1, 1) / 2 + 1
    else
        mid1 = (size(im1, 1) + 1) / 2
    end

    if mod(size(im1, 2), 2) == 0
        mid2 = size(im1, 2) / 2 + 1
    else
        mid2 = (size(im1, 2) + 1) / 2
    end

    # mid1 = round(Int, size(im1, 1) / 2)
    # mid2 = round(Int, size(im1, 2) / 2)
    println("findshift2D: mid1 = $mid1, mid2 = $mid2")
    # Compute the cross-correlation.
    cc = crosscorr2D(im1, im2)
    ccw = crosscorr2Dweighted(im1, im2)
    println("findshift2D: shift  = $(argmax(cc))")
    println("findshift2D: shiftw = $(argmax(ccw))")
    # Find the maximum location in the cross-correlation, which will
    # correspond to the shift between the two images.
    shift = argmax(cc)
    # The -1 accounts for the fact that the first element of an array is
    # at index 1.
    #shift = float([shift[1], shift[2]]) .- 1
    # Since the FFT has been centered, the shift is relative to the
    # center of the transformed histogram images.
    shift = float([shift[1] - mid1, shift[2] - mid2])
    # Convert the shift to an (x, y) coordinate.
    shift = histbinsize .* shift
    # Return the shift.
    return shift
end

"""
Perform a cross-correlation between images representing
localizations in two SMLD structures and compute the shift
between the two original images.
histbinsize is the size of the bins in the histogram image in the
same units as the localization coordinates.
pixelsizeZunit is the conversion factor from (x, y) coordinates (typically,
pixels) to the units of z coordinates (typically, um).  This is needed in
3D to convert z into the same units as x and y so that the shifts in all
directions are calculated in the same units.
"""
function findshift3D(smld1::T, smld2::T; histbinsize::AbstractFloat=1.0,
    pixelsizeZunit::AbstractFloat=0.100
) where {T<:SMLMData.SMLD3D}
    # Compute the histogram images assume the same size for both images).
    if smld1.datasize[1] != smld2.datasize[1] &&
       smld1.datasize[2] != smld2.datasize[2] &&
       smld1.datasize[3] != smld2.datasize[3]
        error("Images must have the same size.")
    end
    if smld1.datasize[1] == 0.0 || smld1.datasize[2] == 0.0 ||
       smld1.datasize[3] == 0.0
        println("findshift3D: smld.datasize(s) are zero.")
        ROI = [-1.0]
    else
        ROI = float([0, smld1.datasize[1], 0, smld1.datasize[2],
            0, smld1.datasize[3] / pixelsizeZunit])
    end
    im1 = histimage3D(smld1.x, smld1.y, smld1.z ./ pixelsizeZunit;
        ROI=ROI, histbinsize=histbinsize)
    im2 = histimage3D(smld2.x, smld2.y, smld2.z ./ pixelsizeZunit;
        ROI=ROI, histbinsize=histbinsize)
    # Determine the midpoints of the histogram images.

    if mod(size(im1, 1), 2) == 0
        mid1 = size(im1, 1) / 2 + 1
    else
        mid1 = (size(im1, 1) + 1) / 2
    end

    if mod(size(im1, 2), 2) == 0
        mid2 = size(im1, 2) / 2 + 1
    else
        mid2 = (size(im1, 2) + 1) / 2
    end

    if mod(size(im1, 3), 2) == 0
        mid3 = size(im1, 3) / 2 + 1
    else
        mid3 = (size(im1, 3) + 1) / 2
    end

    # Compute the cross-correlation.
    cc = crosscorr3D(im1, im2)
    # Find the maximum location in the cross-correlation, which will
    # correspond to the shift between the two images.
    shift = argmax(cc)
    # The -1 accounts for the fact that the first element of an array is
    # at index 1.
    #shift = float([shift[1], shift[2], shift[3]]) .- 1
    # Since the FFT has been centered, the shift is relative to the
    # center of the transformed histogram images.
    shift = float([shift[1] - mid1, shift[2] - mid2, shift[3] - mid3])
    # Convert the shift to an (x, y, z) coordinate.
    shift = histbinsize .* shift
    # Convert the z-shift back to original units.
    shift[3] = shift[3] .* pixelsizeZunit
    # Return the shift.
    return shift
end