```@meta
CurrentModule = SMLMDriftCorrection
```

# API Reference

```@index
Pages = ["api.md", "configuration.md"]
```

## Main Interface

```@docs
driftcorrect
```

See [Configuration](@ref) for full documentation of [`DriftConfig`](@ref) and [`DriftInfo`](@ref).

## Utility Functions

```@docs
filter_emitters
drift_trajectory
```

## Drift Models

```@docs
LegendrePolynomial
IntraLegendre
LegendrePoly1D
```

## Drift Application

```@docs
applydrift
correctdrift
```

## Entropy Functions

```@docs
entropy_HD
ub_entropy
```

## Cross-Correlation

```@docs
findshift
histimage2D
crosscorr2D
```
