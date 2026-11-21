# cudaGlmnet 0.0.0.9015

* **Part B / §9 half (2), qualified**: add an mlr3 learner
  `regr.cuda_subset_ridge` (`R/engine_mlr3_subsets.R`) plus internal
  `ctx_resident_subsets_cpp()` (`src/ctx_resident.cpp`). The learner holds ONE
  persistent `rcupp::ctx_create()` device context across B subsets and across
  CV folds, amortising device setup on the `p >> m` repeated subset-ridge
  workload. On that workload (real CUDA, fair primary comparator = exact CPU
  Cholesky, machine precision) the GPU dual route is 22x–333x faster; vs the
  framework `glmnet` engine ~9x–44x but flagged `glmnet_comparable = FALSE`
  (unequal optimum). This meets §9 half (2) **for repeated closed-form subset
  ridge only** and does NOT reverse the lasso-path result (GPU still loses to
  glmnet Fortran CD). See the `framework-subset-ridge` vignette and
  `benchmark/results/framework_subset_ridge.csv`. No new `rcupp` C ABI symbol;
  `rcupp_ctx_abi()` stays 1. Needs the Part A rcupp (`ctx_create()`); runtime
  guarded so an older rcupp degrades gracefully.

* **Breaking numerical contract change (alpha < 1):** `cuda_glmnet()` and
  `cv.cuda_glmnet()` now take `lambda` / `alpha` in `glmnet`'s Gaussian
  convention. A given `(lambda, alpha)` now reproduces
  `glmnet::glmnet(lambda, alpha)` (verified to ~1e-6 with default solver
  tolerances, ~1e-8 at tight convergence), whereas previously the two differed
  because glmnet scales the response by its RMS `s = sqrt(mean(y^2))`.
  Internally the solver keeps its clean, data-independent standard elastic net
  `lambda_c[(1-alpha_c)/2 ||beta||^2 + alpha_c ||beta||_1]` and maps
  `alpha_c = s*alpha/(1-alpha+s*alpha)`,
  `lambda_c = lambda*(1-alpha+s*alpha)/s`. A scalar rescaling of `lambda`
  alone is impossible for `0 < alpha < 1` because the ridge and lasso penalties
  need different scalings; `alpha` must be remapped too. At `alpha = 1` the map
  is the identity, so the lasso path stays bit-identical to earlier versions.
  The returned `lambda` remains in glmnet units (for `coef(s=)` lookup) and
  fits now also carry `alpha`; CV applies the remap per fold, matching
  `cv.glmnet()`'s per-fold re-standardisation.

* Added the exact derivation and validation as reproducible diagnostics in
  `benchmark/`: `diagnose_glmnet_lambda.R` (KKT shape fit),
  `diagnose_glmnet_scale.R` (identifies `s = sqrt(mean(y^2))`, not `sd(y)`),
  `validate_lambda_remap.R` (end-to-end remap, ~1e-8 at all alpha).
  `tests/testthat/test-glmnet-parity.R` now asserts glmnet agreement directly;
  oracle-based tests call the pure-R oracle at the remapped parameters.

# cudaGlmnet 0.0.0.9014

* Fix the optional P2 framework engines for real `mlr3` 1.8.0 / parsnip 1.6.1
  and verify them end-to-end (register -> fit -> predict), superseding the
  structural-only note in 0.0.0.9013. The `mlr3` learner's `.train` /
  `.predict` are now **private** R6 methods (mlr3 dispatches them via
  `get_private()`; as public methods registration succeeds but `train()`
  fails with "attempt to apply non-function"). The parsnip engine is ported
  to the >= 1.x `value=` / `options=` API of `set_fit` / `set_pred` /
  `set_encoding` (the old `func=` / `defaults=` / `args=` form failed
  silently); registration is now idempotent across repeated loads. A
  `cuda_enet()` model-spec constructor is exported, and `engines.Rmd` plus
  `tests/testthat/test-engines.R` now exercise all three frameworks live.

* Record the honest section 9 P2 result in the `engines` vignette: the
  functional half (register -> CV -> predict) is met for mlr3, but the speed
  half is **not** — single-fit GPU wall times (2.75-11.82 s) are ~100x slower
  than the `glmnet` engine (0.02-0.09 s) on the measured scales.
  `cudaGlmnet`'s advantage is on repeated-fit workloads, not one-off fits.

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
