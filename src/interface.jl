
function driftcorrect(smld::SMLMData.SMLD;
    intramodel::String = "Polynomial",
    degree::Int = 2,
    d_cutoff = 0.1,
    verbose::Int=0)

    if intramodel == "Polynomial"
        driftmodel = Polynomial(smld; degree = degree)
    end

    # Intra dataset 
    if verbose>0
        @info("SMLMDriftCorrection: starting intra")
    end
    Threads.@threads for nn = 1:smld.ndatasets
        findintra!(driftmodel.intra[nn], smld, nn, d_cutoff)
    end

    # Correct them all to datatset 1
    if verbose>0
        @info("SMLMDriftCorrection: starting inter to dataset 1")
    end
    Threads.@threads  for nn = 2:smld.ndatasets
        refdataset = 1
        findinter!(driftmodel, smld, nn, refdataset, d_cutoff)
    end

    if verbose>0
        @info("SMLMDriftCorrection: starting inter to all others")
    end
    # Correct each to all others
    for ii = 1:2, nn = 1:smld.ndatasets
        if verbose>1
            println("SMLMDriftCorrection: round $ii dataset $nn")
        end        
        findinter!(driftmodel, smld, nn, d_cutoff)
    end

    smd_found = correctdrift(smld, driftmodel)

    return smd_found
end
