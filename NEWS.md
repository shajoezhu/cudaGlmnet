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
