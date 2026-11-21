# cudaGlmnet 0.0.0.9015

* Development version bumped from 0.0.0.9014.

# cudaGlmnet 0.0.0.9014

* Development version bumped from 0.0.0.9013.

# cudaGlmnet 0.0.0.9013

* Add a dual / Woodbury route to `cuda_resident_subsets()` via a new `method`
  argument (`"auto"` / `"primal"` / `"dual"`). For `p > m` (the production
  `p >> n` shape) the auto/dual route solves the smaller `m x m` system
  `beta = Xs' (Xs Xs' + m*lambda*I)^{-1} ys` using the C-stage `rcupp_d_gemm`
  part plus `potrf` / `potrs`, instead of materializing a `p x p` Gram. Both
  routes agree with the exact closed form to machine precision; `sys_dim` and
  the resolved `method` are returned. Strategy (`resident` / `cold`) stays
  orthogonal to method. `rcupp` itself remains algorithm-agnostic.

* Add a dual / Woodbury wall-time comparison to the `subset-residency`
  vignette: measured primal / dual ratios of 3.7x (`p = 1000, m = 120`) and
  80.6x (`p = 3000, m = 150`); the gap grows with `p / m`, and `p < m`
  correctly falls back to primal via `method = "auto"`. A snapshot, not a
  guarantee.

* Add optional P2 framework adapters: a SuperLearner `SL.cudaGlmnet`
  wrapper (plus a `predict` method), and lazily-registered `mlr3` /
  parsnip engines. The frameworks are `Suggests` only, guarded by
  `requireNamespace` and `tryCatch` at load time. Gaussian-only and
  `obsWeights` are documented limits. The `mlr3` / parsnip registrations are
  not verified end-to-end on this machine (frameworks not installed); that is
  left as a CI follow-up. Only the SuperLearner fit/predict path is exercised
  by the local test suite.

* Document the overlap with the separate `SLrcupp` package: `SL.rcupp` is a
  standalone CV-based `SL.*` learner that selects `lambda.min` / `lambda.1se`
  via `cv.cuda_glmnet()`, while `SL.cudaGlmnet` is the thin in-package wrapper
  predicting at a single penalty. Both are kept intentionally; see the
  `engines` vignette for the in-package adapters.

* Development version bumped from 0.0.0.9012.

# cudaGlmnet 0.0.0.9012

* Development version bumped from 0.0.0.9011.

# cudaGlmnet 0.0.0.9011

* Add `cuda_resident_subsets()`: a downstream composition that keeps one full
  panel resident in a single `rcupp` context and refits many row subsets via a
  device-side gather, with a `resident` vs `cold` strategy for ROI. It composes
  rcupp's generic B-stage parts (`gather_rows` / `syrk` / `potrf` / `potrs`);
  rcupp itself stays algorithm-agnostic. A `subset-residency` vignette measures
  the data-movement payoff (the flat cross-subset ratio from the previous
  benchmark bends upward here) with an always-on correctness gate.
* Add a `bio-flagship-benchmark` vignette: a p>>n reference-panel case study
  measuring whole lambda-path fits and repeated CV/bootstrap-style refits over
  candidate panels. It reports where device residency pays off (within a panel)
  and honestly marks the flat cross-subset ROI that needs a device-side gather.
* Development version bumped from 0.0.0.9010.

# cudaGlmnet 0.0.0.9010

* Development version bumped from 0.0.0.9009.

# cudaGlmnet 0.0.0.9009

* Development version bumped from 0.0.0.9008.

# cudaGlmnet 0.0.0.9008

* Add a `ridge-benchmark` vignette measuring whole lambda-path fits against
  cold fits, a pure-R oracle, `glmnet` and a direct closed-form solve.
* Development version bumped from 0.0.0.9007.

# cudaGlmnet 0.0.0.9007

* Add glmnet-style `alpha` elastic-net mixing to `cuda_glmnet()` and
  `cv.cuda_glmnet()`. For `alpha = 1` the lasso path is unchanged and
  bit-identical; for `alpha < 1` the solver loop is composed downstream
  (Rcpp glue plus rcupp substrate) from rcupp's resident device operators,
  so rcupp itself gains no algorithm-specific code. Adds `enet_fista_cpu()`
  as the pure-R numerical oracle.
* Development version bumped from 0.0.0.9006.

# cudaGlmnet 0.0.0.9006

* Reconciled NEWS.md with DESCRIPTION version 0.0.0.9006.

# cudaGlmnet 0.0.0.9005

* Initial development version.
