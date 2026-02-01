"""
Main interface for drift correction (DC).  This algorithm consists of an
    intra-dataset portion and an inter-dataset portion.  The drift corrected
    coordinates are returned as output.  All distance units are in μm.

# Arguments
- `smld`:           structure containing (X, Y) or (X, Y, Z) localization
                    coordinates (μm)
- `intramodel`:     model for intra-dataset DC:
                    {"Polynomial", "LegendrePoly"} = "Polynomial"
- `cost_fun`:       intra/inter cost function: {"Kdtree", "Entropy"} = "Kdtree"
- `cost_fun_intra`: intra cost function override: ""
- `cost_fun_inter`: inter cost function override: ""
- `degree`:         degree for polynomial intra-dataset DC = 2
- `d_cutoff`:       distance cutoff (μm) = 0.01 [Kdtree cost function]
- `maxn`:           maximum number of neighbors considered = 200
                    [Entropy cost function]
- `histbinsize`:    histogram bin size for inter-datset cross-correlation
                    correction (μm) = -1.0 [< 0 means no correction]
- `verbose`:        flag for more output = 0

# Returns
- `(smld_corrected, info)`: Tuple of corrected SMLD and `DriftInfo` metadata

# Example
```julia
(smld_corrected, info) = driftcorrect(smld; degree=2)
# Access model for warm starts or trajectory extraction
model = info.model
```
"""
function driftcorrect(smld::SMLD;
    intramodel::String = "Polynomial",
    cost_fun::String = "Kdtree",
    cost_fun_intra::String = "",
    cost_fun_inter::String = "",
    degree::Int = 2,
    d_cutoff::AbstractFloat = 0.01,
    maxn::Int = 200,
    histbinsize::AbstractFloat = -1.0,
    verbose::Int = 0)

    t_start = time_ns()

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

    # Track optimization results
    intra_results = Vector{Any}(undef, smld.n_datasets)
    inter_costs = Float64[]

    # Intra-dataset
    if verbose>0
        @info("SMLMDriftCorrection: starting intra")
    end
    Threads.@threads for nn = 1:smld.n_datasets
        intra_results[nn] = findintra!(driftmodel.intra[nn], cost_fun_intra, smld, nn, d_cutoff,
	           maxn)
    end

    # Inter-dataset: Correct them all to datatset 1
    if verbose>0
        @info("SMLMDriftCorrection: starting inter to dataset 1")
    end
    #Threads.@threads
    for nn = 2:smld.n_datasets
        refdatasets = [1]
        cost = findinter!(driftmodel, cost_fun_inter, smld, nn, refdatasets,
	           d_cutoff, maxn, histbinsize)
        push!(inter_costs, cost)
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
        cost = findinter!(driftmodel, cost_fun_inter, smld, ii, collect((1:(ii-1))),
                   d_cutoff, maxn, histbinsize)
        push!(inter_costs, cost)
    end

    smld_found = correctdrift(smld, driftmodel)

    elapsed_ns = time_ns() - t_start

    # Aggregate optimization statistics
    total_iterations = sum(Optim.iterations(r) for r in intra_results)
    all_converged = all(Optim.converged(r) for r in intra_results)
    intra_costs = [Optim.minimum(r) for r in intra_results]
    total_cost = sum(intra_costs) + sum(inter_costs)
    history = vcat(intra_costs, inter_costs)

    info = DriftInfo(
        driftmodel,
        elapsed_ns,
        :cpu,
        total_iterations,
        all_converged,
        total_cost,
        history
    )

    return (smld_found, info)
end
