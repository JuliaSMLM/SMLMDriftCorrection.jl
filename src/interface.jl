"""
    driftcorrect(smld; kwargs...) -> (smld, model)

Main interface for drift correction (DC). This algorithm consists of an
intra-dataset portion and an inter-dataset portion. All distance units are in μm.

# Arguments
- `smld`: SMLD structure containing (X, Y) or (X, Y, Z) localization coordinates (μm)

# Keyword Arguments
- `intramodel="Polynomial"`: model for intra-dataset DC: {"Polynomial", "LegendrePoly"}
- `cost_fun="Entropy"`: cost function: {"Kdtree", "Entropy", "SymEntropy"}
- `cost_fun_intra=""`: intra cost function override (defaults to cost_fun)
- `cost_fun_inter=""`: inter cost function override (defaults to cost_fun)
- `degree=2`: degree for polynomial intra-dataset DC
- `d_cutoff=0.01`: distance cutoff in μm (Kdtree cost function)
- `maxn=200`: maximum number of neighbors (Entropy/SymEntropy cost functions)
- `histbinsize=-1.0`: histogram bin size for inter-dataset cross-correlation (< 0 disables)
- `verbose=0`: verbosity level

# Returns
NamedTuple with fields:
- `smld`: drift-corrected SMLD
- `model`: fitted drift model (Polynomial or LegendrePolynomial)

# Example
```julia
result = driftcorrect(smld; cost_fun="Kdtree", degree=2)
corrected_smld = result.smld
drift_model = result.model

# Or destructure directly:
(; smld, model) = driftcorrect(smld; cost_fun="Kdtree")
smld_corr, drift_model = driftcorrect(smld)  # tuple-style also works
```
"""
function driftcorrect(smld::SMLD;
    intramodel::String = "Polynomial",
    cost_fun::String = "Entropy",
    cost_fun_intra::String = "",
    cost_fun_inter::String = "",
    degree::Int = 2,
    d_cutoff::AbstractFloat = 0.01,
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
    Threads.@threads for nn = 1:smld.n_datasets
        findintra!(driftmodel.intra[nn], cost_fun_intra, smld, nn, d_cutoff,
	           maxn)
    end

    # Inter-dataset: Correct them all to datatset 1
    if verbose>0
        @info("SMLMDriftCorrection: starting inter to dataset 1")
    end
    #Threads.@threads
    for nn = 2:smld.n_datasets
        refdatasets = [1]
        findinter!(driftmodel, cost_fun_inter, smld, nn, refdatasets,
	           d_cutoff, maxn, histbinsize)
    end

    # if verbose>0
    #     @info("SMLMDriftCorrection: starting inter to all others")
    # end
    # # Correct each to all others
    # for ii = 1:2, nn = 1:smld.n_datasets
    #     if verbose>1
    #         println("SMLMDriftCorrection: round $ii dataset $nn")
    #     end        
    #     findinter!(driftmodel, smld, nn, d_cutoff)
    # end
    
    if verbose>0
        @info("SMLMDriftCorrection: starting inter to earlier")
    end
    for ii = 2:smld.n_datasets
        if verbose>1
            println("SMLMDriftCorrection: dataset $ii")
        end        
        findinter!(driftmodel, cost_fun_inter, smld, ii, collect((1:(ii-1))), 
                   d_cutoff, maxn, histbinsize)
    end
    
    smld_corrected = correctdrift(smld, driftmodel)

    return (smld=smld_corrected, model=driftmodel)
end
