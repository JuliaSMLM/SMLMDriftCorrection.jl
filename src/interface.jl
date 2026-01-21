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
- `dataset_mode=:registered`: acquisition mode for multi-dataset handling:
    - `:registered`: datasets are independent (stage registered between acquisitions)
    - `:continuous`: drift accumulates across datasets (one long acquisition split into files)
- `chunk_frames=0`: for continuous mode, split each dataset into chunks of this many frames (0 = no chunking)
- `n_chunks=0`: alternative to chunk_frames - specify number of chunks per dataset (0 = no chunking)
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

# For continuous acquisition (drift accumulates across datasets):
result = driftcorrect(smld; dataset_mode=:continuous)

# For finer-grained drift correction, chunk into smaller pieces:
result = driftcorrect(smld; dataset_mode=:continuous, n_chunks=10)
result = driftcorrect(smld; dataset_mode=:continuous, chunk_frames=1000)
```
"""
function driftcorrect(smld::SMLD;
    intramodel::String = "Polynomial",
    cost_fun::String = "Entropy",
    cost_fun_intra::String = "",
    cost_fun_inter::String = "",
    dataset_mode::Symbol = :registered,
    chunk_frames::Int = 0,
    n_chunks::Int = 0,
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

    # Handle chunking for continuous mode
    chunk_info = nothing
    smld_work = smld  # Working SMLD (may be chunked)

    if dataset_mode == :continuous && (chunk_frames > 0 || n_chunks > 0)
        chunk_info = chunk_smld(smld; chunk_frames=chunk_frames, n_chunks=n_chunks)
        smld_work = chunk_info.smld

        if verbose > 0
            @info("SMLMDriftCorrection: chunking into $(chunk_info.n_chunks) chunks per dataset " *
                  "($(chunk_info.frames_per_chunk) frames each, $(smld_work.n_datasets) total chunks)")
        end
    end

    if intramodel == "Polynomial"
        driftmodel = Polynomial(smld_work; degree = degree)
    elseif intramodel == "LegendrePoly"
        driftmodel = LegendrePolynomial(smld_work; degree = degree)
    end

    # Intra-dataset
    if verbose>0
        @info("SMLMDriftCorrection: starting intra")
    end
    Threads.@threads for nn = 1:smld_work.n_datasets
        findintra!(driftmodel.intra[nn], cost_fun_intra, smld_work, nn, d_cutoff,
	           maxn)
    end

    # Inter-dataset correction
    if dataset_mode == :continuous
        # Continuous mode: initialize inter-shifts from endpoints, then optimize
        # This provides a warm start for optimization based on drift continuity
        if verbose > 0
            @info("SMLMDriftCorrection: continuous mode - initializing inter from endpoints")
        end

        # First dataset/chunk: zero offset (reference)
        ndims = driftmodel.intra[1].ndims
        driftmodel.inter[1].dm .= 0.0

        # Chain subsequent datasets/chunks: warm start from endpoint of previous
        for nn = 2:smld_work.n_datasets
            endpoint_prev = endpoint_drift(driftmodel.intra[nn-1], smld_work.n_frames)
            startpoint_curr = startpoint_drift(driftmodel.intra[nn])

            for dim in 1:ndims
                driftmodel.inter[nn].dm[dim] = driftmodel.inter[nn-1].dm[dim] +
                                                endpoint_prev[dim] -
                                                startpoint_curr[dim]
            end

            if verbose > 1
                println("  Chunk $nn initial inter: $(driftmodel.inter[nn].dm)")
            end

            # Now optimize from this starting point
            findinter!(driftmodel, cost_fun_inter, smld_work, nn, collect(1:(nn-1)),
                       d_cutoff, maxn, histbinsize)
        end

    elseif dataset_mode == :registered
        # Registered mode: datasets are independent, optimize freely
        if verbose > 0
            @info("SMLMDriftCorrection: registered mode - starting inter to dataset 1")
        end
        for nn = 2:smld_work.n_datasets
            refdatasets = [1]
            findinter!(driftmodel, cost_fun_inter, smld_work, nn, refdatasets,
                       d_cutoff, maxn, histbinsize)
        end

        if verbose > 0
            @info("SMLMDriftCorrection: starting inter to earlier datasets")
        end
        for ii = 2:smld_work.n_datasets
            if verbose > 1
                println("SMLMDriftCorrection: dataset $ii")
            end
            findinter!(driftmodel, cost_fun_inter, smld_work, ii, collect(1:(ii-1)),
                       d_cutoff, maxn, histbinsize)
        end

    else
        error("Unknown dataset_mode: $dataset_mode. Use :registered or :continuous")
    end

    # Apply corrections
    if chunk_info !== nothing && chunk_info.n_chunks > 1
        # Chunked mode: correct the chunked SMLD, then copy coordinates back to original structure
        smld_work_corrected = correctdrift(smld_work, driftmodel)

        # Copy corrected coordinates back to original SMLD structure
        smld_corrected = deepcopy(smld)
        is_3d = nDims(smld) == 3
        for i in eachindex(smld.emitters)
            smld_corrected.emitters[i].x = smld_work_corrected.emitters[i].x
            smld_corrected.emitters[i].y = smld_work_corrected.emitters[i].y
            if is_3d
                smld_corrected.emitters[i].z = smld_work_corrected.emitters[i].z
            end
        end
    else
        smld_corrected = correctdrift(smld, driftmodel)
    end

    return (smld=smld_corrected, model=driftmodel)
end
