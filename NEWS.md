# cudaGlmnet 0.0.0.9008

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
