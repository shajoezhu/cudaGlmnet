#' Pure-R elastic-net / ridge FISTA (independent numerical oracle)
#'
#' A dependency-free reference implementation (base R only, no compiled code)
#' of the Gaussian elastic net solved by the downstream \code{Rcpp} +
#' \pkg{rcupp} path in \code{cuda_glmnet(alpha = )}. It mirrors, line for line,
#' the numerical contract of \code{src/enet_path.cpp} and shares the same
#' Lipschitz constant via \code{.cg_lipschitz()}, so it can validate the
#' compiled path to tight tolerance.
#'
#' The objective is
#' \code{1/(2n)||y - Xb||^2 + lambda[(1-alpha)/2 ||b||_2^2 + alpha ||b||_1]}.
#' With \code{alpha = 1} it reduces to the lasso of [lasso_fista_cpu()]; with
#' \code{alpha = 0} it is pure ridge solved by iterated proximal gradient
#' (a closed-form / SVD path is a later phase). The prox is
#' \code{soft_threshold(u, thresh*alpha) / (1 + thresh*(1-alpha))} where
#' \code{thresh = lambda/L}.
#'
#' @param x Numeric \code{n x p} design matrix.
#' @param y Numeric response of length \code{n}.
#' @param lambda A single non-negative penalty.
#' @param alpha Elastic-net mixing in \code{[0, 1]}; \code{1} = lasso,
#'   \code{0} = ridge.
#' @param max_iter Maximum number of FISTA iterations.
#' @param tol Relative convergence tolerance.
#'
#' @return A list with \code{coefficients} (named when \code{x} has column
#'   names), \code{iterations}, \code{objective} and \code{converged}.
#' @export
enet_fista_cpu <- function(x, y, lambda, alpha = 1, max_iter = 1000L,
                           tol = 1e-6) {
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
  if (!is.numeric(alpha) || length(alpha) != 1L || !is.finite(alpha) ||
        alpha < 0 || alpha > 1) {
    stop("'alpha' must be a single number in [0, 1]", call. = FALSE)
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

  L <- .cg_lipschitz(x)
  step <- 1 / (L * n)
  thresh <- lambda / L
  t1 <- thresh * alpha
  inv <- 1 / (1 + thresh * (1 - alpha))

  beta <- numeric(p)
  z <- numeric(p)
  tk <- 1
  iters <- 0L
  converged <- 0L

  for (it in seq_len(max_iter)) {
    r <- y - as.vector(x %*% z)
    u <- z + step * as.vector(crossprod(x, r))
    beta_new <- sign(u) * pmax(abs(u) - t1, 0) * inv
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
  objective <- sum(resid^2) / (2 * n) +
    lambda * ((1 - alpha) / 2 * sum(beta^2) + alpha * sum(abs(beta)))

  list(coefficients = beta,
       iterations   = iters,
       objective    = objective,
       converged    = converged)
}
