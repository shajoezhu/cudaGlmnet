#' @useDynLib cudaGlmnet, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @noRd
NULL

# lintr's object_usage_linter checks each source file in isolation, so
# internal helpers defined in another R file (and the generated Rcpp
# wrapper) look like undefined globals. Declare them once at package level.
utils::globalVariables(c(".cg_check_alpha", ".cg_lipschitz",
                         ".cg_enet_fits", "enet_path_cpp"))

.cg_check_alpha <- function(alpha) {
  if (!is.numeric(alpha) || length(alpha) != 1L || !is.finite(alpha) ||
        alpha < 0 || alpha > 1) {
    stop("'alpha' must be a single number in [0, 1]", call. = FALSE)
  }
  invisible(TRUE)
}

# L = lambda_max(X'X)/n by 100 deterministic power iterations. Same start
# vector (sin(1..p)) and iteration count as lasso_fista_cpu() and
# cudaBio/src/bio.cpp, so the pure-R oracle and the compiled path share an
# identical Lipschitz constant (no flaky numeric gap between the two).
.cg_lipschitz <- function(Xs) {
  n <- nrow(Xs)
  p <- ncol(Xs)
  v <- sin(seq_len(p))
  nv <- sqrt(sum(v^2))
  if (nv > 0) v <- v / nv
  Lf <- 0
  for (k in seq_len(100L)) {
    u <- as.vector(crossprod(Xs, as.vector(Xs %*% v)))
    rq <- sum(v * u)
    nu <- sqrt(sum(u^2))
    Lf <- rq
    if (nu > 0) {
      v <- u / nu
    } else {
      break
    }
  }
  max(Lf / n, 1e-12)
}

# Turn one enet_path_cpp() call (whole lambda path, data resident in a single
# rcupp context) into the per-lambda list shape produced by .cg_fit_one(), so
# cuda_glmnet() / cv.cuda_glmnet() can reuse their existing assembly code.
.cg_enet_fits <- function(prep, lambda, alpha, max_iter, tol, warm_start) {
  L <- .cg_lipschitz(prep$Xs)
  res <- enet_path_cpp(prep$Xs, prep$ys, as.double(lambda), alpha, L,
                       warm_start, as.integer(max_iter), tol)
  backend <- if (isTRUE(res$cuda)) "cuda" else "cpu-shim"
  lapply(seq_along(lambda), function(j) {
    beta_s <- as.numeric(res$beta[, j])
    beta <- beta_s / prep$xs
    a0 <- prep$ym - sum(prep$xm * beta)
    resid <- prep$ys - as.vector(prep$Xs %*% beta_s)
    objective <- sum(resid^2) / (2 * prep$n) +
      lambda[j] * ((1 - alpha) / 2 * sum(beta_s^2) +
                     alpha * sum(abs(beta_s)))
    list(beta = beta, a0 = a0,
         iterations = res$iterations[j], converged = res$converged[j],
         objective = objective, backend = backend)
  })
}
