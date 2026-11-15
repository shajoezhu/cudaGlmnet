#' Pure-R FISTA lasso (independent numerical oracle)
#'
#' A dependency-free reference implementation of the same Gaussian lasso
#' solved by \pkg{rcupp}. It is deliberately kept separate (base R only, no
#' compiled code) so it can validate the numbers produced by the compiled path.
#'
#' The numerical contract is IDENTICAL, line for line, to
#' \code{rcupp/src/cpu_stub.cpp} and \code{rcupp/src/lasso_fista_cuda.cu}:
#' objective \code{1/(2n)||y - Xb||^2 + lambda||b||_1}, Lipschitz
#' \code{L = lambda_max(X^T X)/n} via 100 deterministic power iterations,
#' FISTA momentum and the same relative convergence test.
#'
#' @param x Numeric \code{n x p} design matrix.
#' @param y Numeric response of length \code{n}.
#' @param lambda A single non-negative lasso penalty.
#' @param max_iter Maximum number of FISTA iterations.
#' @param tol Relative convergence tolerance.
#'
#' @return A list with \code{coefficients} (named when \code{x} has column
#'   names), \code{iterations}, \code{objective} and \code{converged}.
#' @export
lasso_fista_cpu <- function(x, y, lambda, max_iter = 1000L, tol = 1e-6) {
  x <- as.matrix(x)
  if (!is.numeric(x) || anyNA(x)) {
    stop("'x' must be a numeric matrix without missing values", call. = FALSE)
  }
  if (!is.numeric(y) || anyNA(y) || length(y) != nrow(x)) {
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
  storage.mode(x) <- "double"
  y <- as.numeric(y)
  n <- nrow(x)
  p <- ncol(x)

  # L = lambda_max(X'X)/n by deterministic power iteration.
  # Start vector v[j] = sin(j+1), j = 0..p-1 -- identical in all three impls.
  v <- sin(seq_len(p))
  nv <- sqrt(sum(v^2))
  if (nv > 0) v <- v / nv
  Lf <- 0
  for (k in seq_len(100L)) {
    u <- as.vector(crossprod(x, as.vector(x %*% v)))   # X'X v
    rq <- sum(v * u)                                    # Rayleigh quotient
    nu <- sqrt(sum(u^2))
    Lf <- rq
    if (nu > 0) {
      v <- u / nu
    } else {
      break
    }
  }

  L <- max(Lf / n, 1e-12)
  step <- 1 / (L * n)            # = 1/Lf
  thresh <- lambda / L

  beta <- numeric(p)
  z <- numeric(p)
  tk <- 1
  iters <- 0L
  converged <- 0L

  for (it in seq_len(max_iter)) {
    r <- y - as.vector(x %*% z)
    u <- z + step * as.vector(crossprod(x, r))
    a <- abs(u) - thresh
    beta_new <- sign(u) * pmax(a, 0)                   # soft-threshold
    tnew <- (1 + sqrt(1 + 4 * tk^2)) / 2
    mom <- (tk - 1) / tnew
    z_new <- beta_new + mom * (beta_new - beta)

    diff <- sqrt(sum((beta_new - beta)^2))
    denom <- max(1, sqrt(sum(beta_new^2)))

    beta <- beta_new
    z <- z_new
    tk <- tnew
    iters <- it
    if (diff / denom < tol) {
      converged <- 1L
      break
    }
  }

  names(beta) <- colnames(x)
  resid <- y - as.vector(x %*% beta)
  objective <- sum(resid^2) / (2 * n) + lambda * sum(abs(beta))

  list(coefficients = beta,
       iterations   = iters,
       objective    = objective,
       converged    = converged)
}
