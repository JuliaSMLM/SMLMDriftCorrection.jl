"""
Main interface for drift correction (DC).  This algorithm consists of an
    intra-dataset portion and an inter-dataset portion.

# Fields
- smld:       structure containing (X, Y) coordinates (pixel)
- intramodel: model for intra-dataset DC:
              {"Polynomial", "LegendrePoly"} = "Polynomial"
- cost_fun:   intra/inter cost function: {"Kdtree", "Entropy"} = "Kdtree"
- cost_fun_intra: intra cost function override: ""
- cost_fun_inter: inter cost function override: ""
- degree:     degree for polynomial intra-dataset DC = 2
- d_cutoff:   distance cutoff (pixel) = 0.1
- maxn:       maximum number of neighbors considered = 200
- crosscorr:  flag for inter-dataset cross-correlation correction = false
- histbinsize: histogram bin size for inter-datset cross-correlation
               correction (pixel) = -1.0 [< 0 means no correction]
- verbose:    flag for more output = 0
# Output
- smd_found:  structure containing drift corrected (X, Y) coordinates (pixel)
"""
function driftcorrect(smld::SMLMData.SMLD;
    intramodel::String = "Polynomial",
    cost_fun::String = "Kdtree",
    cost_fun_intra::String = "",
    cost_fun_inter::String = "",
    degree::Int = 2,
    d_cutoff::AbstractFloat = 0.1,
    maxn::Int = 200,
    histbinsize::AbstractFloat = -1.0,
    verbose::Int = 0)

    # Overrides for cost function specifications
    if isempty(cost_fun_intra)
        cost_fun_intra = cost_fun
    end
    if isempty(cost_fun_inter)
        cost_fun_inter = cost_fun
    end

    if intramodel == "Polynomial"
        driftmodel = Polynomial(smld; degree = degree)
    elseif intramodel == "LegendrePoly"
        driftmodel = LegendrePolynomial(smld; degree = degree)
    end

    # Intra-dataset 
    if verbose>0
        @info("SMLMDriftCorrection: starting intra")
    end
    Threads.@threads for nn = 1:smld.ndatasets
        findintra!(driftmodel.intra[nn], cost_fun_intra, smld, nn, d_cutoff, maxn)
    end

    # Inter-dataset: Correct them all to datatset 1
    if verbose>0
        @info("SMLMDriftCorrection: starting inter to dataset 1")
    end
    #Threads.@threads
    for nn = 2:smld.ndatasets
        refdatasets = [1]
        findinter!(driftmodel, cost_fun_inter, smld, nn, refdatasets, d_cutoff, maxn,
                   histbinsize)
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
        findinter!(driftmodel, cost_fun_inter, smld, ii, collect((1:(ii-1))), 
                   d_cutoff, maxn, histbinsize)
    end
    
    smd_found = correctdrift(smld, driftmodel)

    return smd_found
end