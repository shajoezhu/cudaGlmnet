#' @importFrom rcupp lasso_fista
#' @importFrom stats coef predict
#' @noRd
NULL

#' Fit a Gaussian lasso with GPU FISTA
#'
#' Minimal glmnet-like entry point. Calls the compiled FISTA solver in
#' \pkg{rcupp} exactly once; all iterations happen in that one call (on the GPU
#' when a CUDA backend was linked in, otherwise on the CPU reference shim).
#'
#' @section v1 limitations:
#' A single fixed \code{lambda}, dense \code{x}, Gaussian family only. **No
#' intercept and no standardization** are performed: pass already-centered /
#' scaled \code{x} and \code{y}. \code{standardize} is accepted for API
#' compatibility but is a no-op in v1. Lambda paths, elastic net, cross
#' validation, binomial and sparse \code{x} are future work.
#'
#' @param x Numeric \code{n x p} design matrix (no missing values).
#' @param y Numeric response of length \code{n}.
#' @param lambda A single non-negative lasso penalty.
#' @param max_iter Maximum number of FISTA iterations.
#' @param tol Relative convergence tolerance.
#' @param standardize Accepted but ignored (no-op in v1; see limitations).
#' @param ... Ignored (reserved for future glmnet-compatible arguments).
#'
#' @return An object of class \code{"cuda_lasso"}: a list with
#'   \code{coefficients}, \code{lambda}, \code{iterations}, \code{converged},
#'   \code{objective}, \code{backend} (\code{"cuda"} or \code{"cpu-shim"}) and
#'   \code{call}. See [coef.cuda_lasso()] and [predict.cuda_lasso()].
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(100 * 5), 100, 5)
#' y <- drop(X %*% c(2, -1, 0, 0, 0))
#' fit <- cuda_lasso(X, y, lambda = 0.05)
#' print(fit)
#' coef(fit)
#'
#' @export
cuda_lasso <- function(x, y, lambda, max_iter = 1000L, tol = 1e-6,
                       standardize = FALSE, ...) {
  if (!is.matrix(x) || !is.numeric(x) || anyNA(x)) {
    stop("'x' must be a numeric matrix without missing values", call. = FALSE)
  }
  y <- as.numeric(y)
  if (length(y) != nrow(x) || anyNA(y)) {
    stop("'y' must be numeric, of length nrow(x), without NA", call. = FALSE)
  }
  if (!is.numeric(lambda) || length(lambda) != 1L || is.na(lambda) ||
        lambda < 0) {
    stop("'lambda' must be a single non-negative number", call. = FALSE)
  }
  if (!is.numeric(max_iter) || length(max_iter) != 1L ||
        !is.finite(max_iter) || max_iter < 1) {
    stop("'max_iter' must be a single positive integer", call. = FALSE)
  }
  if (!is.numeric(tol) || length(tol) != 1L ||
        !is.finite(tol) || tol <= 0) {
    stop("'tol' must be a single positive number", call. = FALSE)
  }
  # nocov start -- rcupp is in Imports; this branch cannot fire.
  if (!requireNamespace("rcupp", quietly = TRUE)) {
    stop("package 'rcupp' is required; install it first", call. = FALSE)
  }
  # nocov end

  res <- lasso_fista(x, y, as.double(lambda), max_iter = max_iter, tol = tol)

  structure(
    list(
      coefficients = res$beta,
      lambda       = as.double(lambda),
      iterations   = res$iterations,
      converged    = res$converged,
      objective    = res$objective,
      backend      = res$backend,
      call         = match.call()
    ),
    class = "cuda_lasso"
  )
}
