"""
Main interfact for drift correction (DC).  This algorithm consists of an
    intra-dataset portion and an inter-dataset portion.

# Fields
- smld:       structure containing (X, Y) coordinates (pixel)
- intramodel: model for intra-dataset DC = "Polynomial"
- degree:     degree for polynomial intra-dataset DC = 2
- d_cutoff:   distance cutoff (pixel) = 0.1
- knn_intra:  not used = 4
- knn_inter:  not used = 4
- verbose:    flag for more output = 0

"""
function driftcorrect(smld::SMLMData.SMLD;
    intramodel::String = "Polynomial",
    degree::Int = 2,
    d_cutoff = 0.1,
    knn_intra::Int = 4,
    knn_inter::Int = 4,
    verbose::Int=0)

    if intramodel == "Polynomial"
        driftmodel = Polynomial(smld; degree = degree)
    end

    # Intra-dataset 
    if verbose>0
        @info("SMLMDriftCorrection: starting intra")
    end
    Threads.@threads for nn = 1:smld.ndatasets
        findintra!(driftmodel.intra[nn], smld, nn, d_cutoff)
    end

    # Inter-dataset: Correct them all to datatset 1
    if verbose>0
        @info("SMLMDriftCorrection: starting inter to dataset 1")
    end
    Threads.@threads  for nn = 2:smld.ndatasets
        refdatasets = [1]
        findinter!(driftmodel, smld, nn, refdatasets, d_cutoff)
    end

    # if verbose>0
    #     @info("SMLMDriftCorrection: starting inter to all others")
    # end
    # # Correct each to all others
    # for ii = 1:2, nn = 1:smld.ndatasets
    #     if verbose>1
    #         println("SMLMDriftCorrection: round $ii dataset $nn")
    #     end        
    #     findinter!(driftmodel, smld, nn, d_cutoff)
    # end
    
    if verbose>0
        @info("SMLMDriftCorrection: starting inter to earlier")
    end
    for ii = 2:smld.ndatasets
        if verbose>1
            println("SMLMDriftCorrection: dataset $ii")
        end        
        findinter!(driftmodel, smld, ii, collect((1:(ii-1))),d_cutoff)
    end
    
    smd_found = correctdrift(smld, driftmodel)

    return smd_found
end
