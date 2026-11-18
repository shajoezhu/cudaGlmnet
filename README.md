# cudaGlmnet

<img src="man/figures/logo.png" align="right" height="160" alt="cudaGlmnet hexsticker" />

[![Check](https://github.com/shajoezhu/cudaGlmnet/actions/workflows/check.yaml/badge.svg)](https://github.com/shajoezhu/cudaGlmnet/actions/workflows/check.yaml)

A minimal **glmnet-like front end for the Gaussian lasso**, solved by GPU FISTA
through the low-level [rcupp](https://github.com/shajoezhu/rcupp) package. It
adds a lambda path, cross-validation, standardization and an intercept, with
glmnet-style `coef(s=)` / `predict(s=)` methods — while keeping the compiled
FISTA solver as the single numerical engine.

If you want the raw operators/sessions instead of a glmnet-style API, use
`rcupp` directly.

## What it provides

- `cuda_glmnet()` — a full **lambda path** (`lambda_max = max|X^T y|/n` down to
  `lambda.min.ratio`), standardization (population scale `1/n`), an intercept
  (via centering), coefficients mapped back to the original scale, and glmnet
  style `s` selection by numeric `lambda` or log-lambda interpolation. `warm_start`
  is opt-in (default off, keeping cold-start fits bit-identical to one-shot).
- `cv.cuda_glmnet()` — K-fold CV with `lambda.min` / `lambda.1se`, custom
  `foldid` and user-supplied `lambda` grids.
- glmnet-style S3 methods — `coef()` / `predict()` / `print()` for
  `cuda_glmnet`, `cv.cuda_glmnet` and `cuda_lasso` objects.
- `cuda_lasso()` — the single fixed-lambda entry point, kept deliberately
  unchanged as the **numerical reference** (no intercept / standardization).
- `lasso_fista_cpu()` — a pure-R FISTA oracle with the identical numerical
  contract, used to verify the compiled path.

A whole lambda path (and each CV fold) runs through **one `rcupp` session**:
`X`/`y` are uploaded to the device once and the cached Lipschitz constant is
reused across every lambda, which is what makes repeated-fit workloads fast.

## Installation

`cudaGlmnet` depends on `rcupp` at run time (declared in `Imports` and
`Remotes: shajoezhu/rcupp`). Install `rcupp` first (or let `remotes` pull it
from GitHub):

```r
# install.packages("remotes")
remotes::install_github("shajoezhu/rcupp")
remotes::install_github("shajoezhu/cudaGlmnet")
```

CUDA is **optional** — it is inherited from how `rcupp` was built. With a GPU
`rcupp` you get `backend = "cuda"`; with the CPU fallback you get
`"cpu-shim"`. All fits run and give identical numerics either way.

## Quick start

```r
library(cudaGlmnet)
set.seed(1)
X <- matrix(rnorm(60 * 5), 60, 5)
y <- drop(X %*% c(2, -1, 0, 0, 0)) + rnorm(60, sd = 0.5)

# Full lambda path
fit <- cuda_glmnet(X, y, nlambda = 10)
coef(fit, s = fit$lambda[5])                # coefficients at one lambda
predict(fit, X[1:3, ], s = fit$lambda[5])   # predictions at one lambda

# Cross-validation
cv <- cv.cuda_glmnet(X, y, nlambda = 8, nfolds = 3)
cv$lambda.min
coef(cv, s = "lambda.1se")

# Single fixed-lambda numerical reference + pure-R oracle
ref <- cuda_lasso(X, y, lambda = 0.05)
max(abs(coef(ref) - lasso_fista_cpu(X, y, 0.05)$coefficients))  # ~0
```

## Scope and numerical notes

- Gaussian family, dense `x`, GPU FISTA. Not yet: elastic net (`alpha`),
  binomial, sparse `x`.
- FISTA and glmnet's coordinate descent solve the **same objective** but use
  different algorithms and convergence criteria: the drop-in promises agreement
  within tolerance, not bit-for-bit reproduction of glmnet's solution path.
- Benchmarks compare against the **pure-R** FISTA oracle, not glmnet's compiled
  Fortran; read speedups accordingly. GPU acceleration is real but conditional
  on problem size (large single fits and `p >> n` workloads win; many tiny
  rolling-window fits can lose).
- A comparison vignette (`gpu-lasso-comparison`) contrasts with
  `cuda.ml::cuda_ml_lasso()` and `gputools`; `cuda.ml` is an optional `Suggests`
  and the vignette skips gracefully when it (or its RAPIDS runtime) is absent.
  A `getting-started` vignette covers the basic API.

## Part of the cudaGlmnet family

| Package | Role | Link |
|---|---|---|
| rcupp | low-level Rcpp/CUDA bridge + reusable operator/session framework | [github](https://github.com/shajoezhu/rcupp) |
| **cudaGlmnet** | glmnet-style GPU lasso front end | this repo |
| cudaBio | PCA / GWAS power iteration (`p >> n`) downstream example | [github](https://github.com/shajoezhu/cudaBio) |
| cudaFin | rolling-window lasso forecasting downstream example | [github](https://github.com/shajoezhu/cudaFin) |

## License

Apache License 2.0.
