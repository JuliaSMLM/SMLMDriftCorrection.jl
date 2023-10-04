using Revise
using FourierTools

"""
Produce a histogram image from the localization coordinates x and y,
"""
function histimage(x, y; histbinsize=5)
    # Compute histogram
    h = fit(Histogram, x, nbins)
    # Compute image
    img = zeros(nbins, nbins)
    for i in 1:nbins
        for j in 1:nbins
            img[i, j] = h.weights[i, j]
        end
    end
    # Compute x and y bins
    xbins = range(minimum(x), stop=maximum(x), length=nbins+1)
    ybins = range(minimum(y), stop=maximum(y), length=nbins+1)
    # Return image and bins
    return img, xbins, ybins
end