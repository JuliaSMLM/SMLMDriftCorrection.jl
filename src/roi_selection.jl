# Automatic ROI subsampling for faster drift correction
# Stage drift is spatially uniform, so models estimated from a dense subset apply everywhere

"""
    calculate_n_locs_required(n_frames; kwargs...) -> Int

Calculate minimum number of localizations needed for accurate drift estimation.

The formula accounts for:
- Polynomial degree (more coefficients = more data needed)
- Localization precision vs target drift precision
- Temporal density of pairs (depends on blink rate and frame count)

# Keyword Arguments
- `degree=2`: Polynomial degree for drift model
- `σ_loc=0.010`: Typical localization precision (μm), default 10 nm
- `σ_target=0.001`: Target drift precision (μm), default 1 nm
- `safety_factor=2.0`: Safety multiplier for required pairs
- `n_frames_target=1000`: Target frame window for temporal pairing

# Returns
Integer number of localizations required.

# Details
Based on error propagation: to achieve σ_target from σ_loc measurements,
need n_pairs = 2*(degree+1)*(σ_loc/σ_target)² pairs. The number of pairs
available depends on the temporal density of emitter blinks.
"""
function calculate_n_locs_required(n_frames::Int;
    degree::Int = 2,
    σ_loc::Float64 = 0.010,
    σ_target::Float64 = 0.001,
    safety_factor::Float64 = 2.0,
    n_frames_target::Int = 1000)

    # Required pairs for target precision
    n_pairs = 2 * (degree + 1) * (σ_loc / σ_target)^2 * safety_factor

    # Temporal scaling: ~3 blinks/emitter spread over n_frames
    # Pairs form within a ~1000-frame window
    λ_window = 3.0 * (n_frames_target / n_frames)

    # Pairs per emitter (Poisson pairing)
    pairs_per_emitter = max(λ_window^2 / 2, 0.01)  # guard against very small

    # Convert to emitters, then to localizations (~10 locs/emitter)
    n_emitters = n_pairs / pairs_per_emitter
    return ceil(Int, n_emitters * 10)
end

"""
    find_dense_roi(smld, n_locs_target; bin_size=1.0) -> indices

Find indices of localizations within a dense contiguous rectangular region.

Selects a rectangular ROI containing at least `n_locs_target` localizations,
chosen from the densest part of the field of view. Returns ALL localizations
within the selected region (not scattered points).

This ensures that multiple blinks from the same emitter are included together,
which is essential for entropy-based drift correction.

# Arguments
- `smld`: SMLD structure with localization data
- `n_locs_target`: Minimum number of localizations needed

# Keyword Arguments
- `bin_size=1.0`: Bin size in μm for density estimation grid

# Returns
Vector of integer indices into smld.emitters for all localizations in the ROI.
"""
function find_dense_roi(smld::SMLD, n_locs_target::Int; bin_size::Float64 = 1.0)
    n_locs = length(smld.emitters)

    # Edge case: if target >= total, return all
    if n_locs_target >= n_locs
        return collect(1:n_locs)
    end

    # Extract coordinates
    x = [e.x for e in smld.emitters]
    y = [e.y for e in smld.emitters]

    x_min, x_max = extrema(x)
    y_min, y_max = extrema(y)

    # Build density grid
    nx = max(1, ceil(Int, (x_max - x_min) / bin_size))
    ny = max(1, ceil(Int, (y_max - y_min) / bin_size))

    # Count localizations per bin
    counts = zeros(Int, nx, ny)
    for i in 1:n_locs
        ix = clamp(floor(Int, (x[i] - x_min) / bin_size) + 1, 1, nx)
        iy = clamp(floor(Int, (y[i] - y_min) / bin_size) + 1, 1, ny)
        counts[ix, iy] += 1
    end

    # Find smallest rectangular window containing >= n_locs_target
    # Use sliding window approach, starting small and growing
    best_roi = nothing
    best_count = 0

    # Precompute cumulative sum for fast window queries
    cumsum_counts = zeros(Int, nx + 1, ny + 1)
    for i in 1:nx, j in 1:ny
        cumsum_counts[i+1, j+1] = counts[i, j] + cumsum_counts[i, j+1] +
                                   cumsum_counts[i+1, j] - cumsum_counts[i, j]
    end

    # Helper to get count in rectangle [i1:i2, j1:j2]
    function window_count(i1, i2, j1, j2)
        return cumsum_counts[i2+1, j2+1] - cumsum_counts[i1, j2+1] -
               cumsum_counts[i2+1, j1] + cumsum_counts[i1, j1]
    end

    # Search for smallest window with enough localizations
    # Start with small windows, grow until we find one with enough
    found = false
    for window_size in 1:max(nx, ny)
        if found
            break
        end
        for wx in 1:min(window_size, nx)
            wy = window_size - wx + 1
            if wy < 1 || wy > ny
                continue
            end
            for i1 in 1:(nx - wx + 1)
                for j1 in 1:(ny - wy + 1)
                    i2 = i1 + wx - 1
                    j2 = j1 + wy - 1
                    cnt = window_count(i1, i2, j1, j2)
                    if cnt >= n_locs_target
                        if best_roi === nothing || cnt < best_count
                            best_roi = (i1, i2, j1, j2)
                            best_count = cnt
                            found = true
                        end
                    end
                end
            end
        end
    end

    # If no window found (shouldn't happen), return all
    if best_roi === nothing
        return collect(1:n_locs)
    end

    # Convert bin indices to spatial coordinates
    i1, i2, j1, j2 = best_roi
    roi_x_min = x_min + (i1 - 1) * bin_size
    roi_x_max = x_min + i2 * bin_size
    roi_y_min = y_min + (j1 - 1) * bin_size
    roi_y_max = y_min + j2 * bin_size

    # Return indices of ALL localizations within the ROI
    indices = Int[]
    for i in 1:n_locs
        if roi_x_min <= x[i] <= roi_x_max && roi_y_min <= y[i] <= roi_y_max
            push!(indices, i)
        end
    end

    return indices
end
