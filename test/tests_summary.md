*** Tests Summary ***

- tests/runtests.jl*
  Nmer dataset driftcorrect K-d tree/PairCorr
- examples/applycorrect.jl
  demonstrate applying and correcting drift to simulated data (plotlyJS)
- examples/finddrift.jl
  as applycorrect,jl, but more elaborate (GLMakie)
- examples/test_entropy.jl
  Keith's original SEAdjust-like plots
- dev/test_cc.jl*
  compare Kdtree, Entropy and findshift2D on simulated data
- dev/test_crosscorr.jl#
  test cross-correlation (findshift2D and findshift3D) on simulated data
- dev/test3D.jl#
  test 3D drift correction on real and simulated data (GLMakie)
- dev/loc_entropy/test_entropy.jl*
  SEAdjust-like plots (CairoMakie)
- dev/loc_entropy/time_entropy.jl*
  timing test for entropy calculated on simulated data