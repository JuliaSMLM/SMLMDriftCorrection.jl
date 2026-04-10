"""
Diagnostics for evaluating drift correction quality.

These functions are read-only and do not modify the drift model or SMLD.
They are intended to be run before/after `driftcorrect` to assess whether
residual drift remains in a corrected dataset.
"""

"""
    position_frame_correlation(smld::SMLD; K::Int=20, mode::Symbol=:intra)

Diagnose residual drift by correlating local position residuals with frame
number (intra mode) or dataset index (inter mode).

For each localization, the `K` nearest spatial neighbors are found and a
residual is computed as `pos_i - mean(neighbor_positions)`. If drift
correction is effective, these residuals should be uncorrelated with frame
number (intra mode) or dataset index (inter mode). A nonzero correlation
indicates remaining systematic drift.

# Arguments
- `smld::SMLD`: localization dataset (2D or 3D, single or multi-dataset)
- `K::Int=20`: number of spatial neighbors used to estimate the local mean
- `mode::Symbol=:intra`: `:intra` runs per-dataset (residual vs frame);
  `:inter` runs across all datasets combined (residual vs dataset index)

# Returns
For `mode = :intra`, a NamedTuple:
```
(per_dataset = [(dataset, n_locs, corr_x, corr_y, corr_z,
                 residuals_x, residuals_y, residuals_z, frames), ...],
 summary = (mean_abs_corr_x, mean_abs_corr_y, mean_abs_corr_z),
 mode = :intra, K = K)
```

For `mode = :inter`, a NamedTuple:
```
(corr_x, corr_y, corr_z,
 residuals_x, residuals_y, residuals_z,
 dataset_indices,
 mode = :inter, K = K)
```

The `corr_z` and `residuals_z` fields are `nothing` for 2D data.

# Example
```julia
(smld_corrected, info) = driftcorrect(smld)
diag = position_frame_correlation(smld_corrected; K=20, mode=:intra)
diag.summary.mean_abs_corr_x  # small values indicate good correction
```
"""
function position_frame_correlation(smld::SMLD; K::Int=20, mode::Symbol=:intra)
    if mode !== :intra && mode !== :inter
        throw(ArgumentError("mode must be :intra or :inter, got $mode"))
    end
    is_3d = nDims(smld) == 3

    if mode === :intra
        return _position_frame_correlation_intra(smld, K, is_3d)
    else
        return _position_frame_correlation_inter(smld, K, is_3d)
    end
end

"""
    _compute_local_residuals(x, y, z, K, is_3d)

Build a KDTree over the provided coordinates and compute, for each point,
the residual between its position and the mean of its `K` nearest neighbors
(self excluded). Returns `(residuals_x, residuals_y, residuals_z)` where
`residuals_z` is `nothing` for 2D data.
"""
function _compute_local_residuals(x::Vector{Float64}, y::Vector{Float64},
                                  z::Union{Vector{Float64},Nothing},
                                  K::Int, is_3d::Bool)
    N = length(x)
    k = min(K, N - 1)

    residuals_x = Vector{Float64}(undef, N)
    residuals_y = Vector{Float64}(undef, N)
    residuals_z = is_3d ? Vector{Float64}(undef, N) : nothing

    if k < 1
        fill!(residuals_x, 0.0)
        fill!(residuals_y, 0.0)
        if is_3d
            fill!(residuals_z, 0.0)
        end
        return (residuals_x, residuals_y, residuals_z)
    end

    if is_3d
        data = Matrix{Float64}(undef, 3, N)
        @inbounds for i in 1:N
            data[1, i] = x[i]
            data[2, i] = y[i]
            data[3, i] = z[i]
        end
    else
        data = Matrix{Float64}(undef, 2, N)
        @inbounds for i in 1:N
            data[1, i] = x[i]
            data[2, i] = y[i]
        end
    end

    kdtree = KDTree(data; leafsize=10)
    idxs, _ = knn(kdtree, data, k + 1, true)  # k+1 because first is self

    inv_k = 1.0 / k

    @inbounds for i in 1:N
        idx_i = idxs[i]
        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0
        for j in 1:k
            nb = idx_i[j + 1]  # skip self at index 1
            sum_x += x[nb]
            sum_y += y[nb]
            if is_3d
                sum_z += z[nb]
            end
        end
        residuals_x[i] = x[i] - sum_x * inv_k
        residuals_y[i] = y[i] - sum_y * inv_k
        if is_3d
            residuals_z[i] = z[i] - sum_z * inv_k
        end
    end

    return (residuals_x, residuals_y, residuals_z)
end

"""
Safe correlation that returns 0.0 when one of the inputs has zero variance
(e.g. a single-frame dataset). Both vectors must be nonempty and the same
length.
"""
function _safe_cor(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    if length(a) < 2
        return 0.0
    end
    if std(a) == 0.0 || std(b) == 0.0
        return 0.0
    end
    return cor(a, b)
end

function _position_frame_correlation_intra(smld::SMLD, K::Int, is_3d::Bool)
    n_datasets = smld.n_datasets

    per_dataset = Vector{NamedTuple}(undef, 0)
    abs_corr_x = Float64[]
    abs_corr_y = Float64[]
    abs_corr_z = Float64[]

    for ds in 1:n_datasets
        smld_ds = filter_by_dataset(smld, ds)
        emitters = smld_ds.emitters
        N = length(emitters)

        if N < K + 1
            @warn "Dataset $ds has $N localizations (< K+1 = $(K+1)); skipping"
            push!(per_dataset, (
                dataset = ds,
                n_locs = N,
                corr_x = 0.0,
                corr_y = 0.0,
                corr_z = is_3d ? 0.0 : nothing,
                residuals_x = Float64[],
                residuals_y = Float64[],
                residuals_z = is_3d ? Float64[] : nothing,
                frames = Int[],
            ))
            continue
        end

        x = Float64[e.x for e in emitters]
        y = Float64[e.y for e in emitters]
        z = is_3d ? Float64[e.z for e in emitters] : nothing
        frames = Int[e.frame for e in emitters]

        residuals_x, residuals_y, residuals_z = _compute_local_residuals(x, y, z, K, is_3d)

        frames_f = Float64.(frames)
        corr_x = _safe_cor(residuals_x, frames_f)
        corr_y = _safe_cor(residuals_y, frames_f)
        corr_z = is_3d ? _safe_cor(residuals_z, frames_f) : nothing

        push!(abs_corr_x, abs(corr_x))
        push!(abs_corr_y, abs(corr_y))
        if is_3d
            push!(abs_corr_z, abs(corr_z))
        end

        push!(per_dataset, (
            dataset = ds,
            n_locs = N,
            corr_x = corr_x,
            corr_y = corr_y,
            corr_z = corr_z,
            residuals_x = residuals_x,
            residuals_y = residuals_y,
            residuals_z = residuals_z,
            frames = frames,
        ))
    end

    mean_abs_corr_x = isempty(abs_corr_x) ? 0.0 : mean(abs_corr_x)
    mean_abs_corr_y = isempty(abs_corr_y) ? 0.0 : mean(abs_corr_y)
    mean_abs_corr_z = if !is_3d
        nothing
    elseif isempty(abs_corr_z)
        0.0
    else
        mean(abs_corr_z)
    end

    summary = (
        mean_abs_corr_x = mean_abs_corr_x,
        mean_abs_corr_y = mean_abs_corr_y,
        mean_abs_corr_z = mean_abs_corr_z,
    )

    return (
        per_dataset = per_dataset,
        summary = summary,
        mode = :intra,
        K = K,
    )
end

function _position_frame_correlation_inter(smld::SMLD, K::Int, is_3d::Bool)
    emitters = smld.emitters
    N = length(emitters)

    if N < K + 1
        @warn "SMLD has $N localizations (< K+1 = $(K+1)); returning zero correlations"
        return (
            corr_x = 0.0,
            corr_y = 0.0,
            corr_z = is_3d ? 0.0 : nothing,
            residuals_x = Float64[],
            residuals_y = Float64[],
            residuals_z = is_3d ? Float64[] : nothing,
            dataset_indices = Int[],
            mode = :inter,
            K = K,
        )
    end

    x = Float64[e.x for e in emitters]
    y = Float64[e.y for e in emitters]
    z = is_3d ? Float64[e.z for e in emitters] : nothing
    dataset_indices = Int[e.dataset for e in emitters]

    residuals_x, residuals_y, residuals_z = _compute_local_residuals(x, y, z, K, is_3d)

    ds_f = Float64.(dataset_indices)
    corr_x = _safe_cor(residuals_x, ds_f)
    corr_y = _safe_cor(residuals_y, ds_f)
    corr_z = is_3d ? _safe_cor(residuals_z, ds_f) : nothing

    return (
        corr_x = corr_x,
        corr_y = corr_y,
        corr_z = corr_z,
        residuals_x = residuals_x,
        residuals_y = residuals_y,
        residuals_z = residuals_z,
        dataset_indices = dataset_indices,
        mode = :inter,
        K = K,
    )
end
