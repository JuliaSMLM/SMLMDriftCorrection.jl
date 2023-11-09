using Revise
#using FourierTools

"""
Produce a histogram image from the localization coordinates x and y,
x and y are assumed to be in nm.
histbinsize is the size of the bins in nm.
"""
function histimage(x::Vector{T}, y::Vector{T}; histbinsize::T=5.0
    ) where {T<:Real}
    x_min = minimum(x)
    x_max = maximum(x)
    y_min = minimum(y)
    y_max = maximum(y)
    # Compute the number of pixels in x and y.
    imszX = round(Int, (x_max .- x_min) ./ histbinsize)
    imszY = round(Int, (y_max .- y_min) ./ histbinsize)
    # Create a blank image.
    im = zeros(imszX, imszY)
    # Convert (x, y) coordinates into pixel units.
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

    # Return image
    return im
end