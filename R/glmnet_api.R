#' @importFrom rcupp lasso_fista
#' @noRd
NULL

# ---------------------------------------------------------------------------
# glmnet-compatible front end. All new API; cuda_lasso() is deliberately left
# untouched so the demo, the three benchmarks and lasso_fista_cpu() keep their
# exact numerical reference contract.
#
# Numerical convention matches glmnet's Gaussian objective:
#   minimise 1/(2n) ||y - a0 - X beta||^2 + lambda ||beta||_1
# Standardisation uses the population (1/n) scale, as glmnet does. Fitting is
# done on centered/standardised (Xs, ys) with the low-level FISTA solver, then
# coefficients are mapped back to the original scale.
# ---------------------------------------------------------------------------

.cg_check_xy <- function(x, y) {
  if (!is.matrix(x) || !is.numeric(x) || anyNA(x)) {
    stop("'x' must be a numeric matrix without missing values", call. = FALSE)
  }
  y <- as.numeric(y)
  if (length(y) != nrow(x) || anyNA(y)) {
    stop("'y' must be numeric, of length nrow(x), without NA", call. = FALSE)
  }
  invisible(y)
}

.cg_check_scalars <- function(max_iter, tol, nlambda, lambda.min.ratio) {
  if (!is.numeric(max_iter) || length(max_iter) != 1L ||
        !is.finite(max_iter) || max_iter < 1) {
    stop("'max_iter' must be a single positive integer", call. = FALSE)
  }
  if (!is.numeric(tol) || length(tol) != 1L ||
        !is.finite(tol) || tol <= 0) {
    stop("'tol' must be a single positive number", call. = FALSE)
  }
  if (!is.numeric(nlambda) || length(nlambda) != 1L ||
        !is.finite(nlambda) || nlambda < 1) {
    stop("'nlambda' must be a single positive integer", call. = FALSE)
  }
  if (!is.null(lambda.min.ratio) &&
        (!is.numeric(lambda.min.ratio) || length(lambda.min.ratio) != 1L ||
           !is.finite(lambda.min.ratio) || lambda.min.ratio <= 0 ||
           lambda.min.ratio >= 1)) {
    stop("'lambda.min.ratio' must be a single number in (0, 1)", call. = FALSE)
  }
  invisible(TRUE)
}

.cg_check_flags <- function(standardize, intercept) {
  for (nm in c("standardize", "intercept")) {
    v <- get(nm)
    if (!is.logical(v) || length(v) != 1L || is.na(v)) {
      stop(sprintf("'%s' must be TRUE or FALSE", nm), call. = FALSE)
    }
  }
  invisible(TRUE)
}

.cg_prep <- function(x, y, standardize, intercept) {
  n <- nrow(x)
  p <- ncol(x)
  xm <- if (intercept) colMeans(x) else rep(0, p)
  ym <- if (intercept) mean(y) else 0
  Xc <- x - matrix(xm, n, p, byrow = TRUE)
  yc <- y - ym
  xs <- if (standardize) sqrt(colMeans(Xc^2)) else rep(1, p)
  xs[!is.finite(xs) | xs < .Machine$double.eps] <- 1  # constant columns
  Xs <- Xc / matrix(xs, n, p, byrow = TRUE)
  storage.mode(Xs) <- "double"
  list(Xs = Xs, ys = yc, xm = xm, xs = xs, ym = ym, n = n, p = p)
}

.cg_lambda_seq <- function(prep, nlambda, lambda.min.ratio) {
  lmax <- max(abs(as.vector(crossprod(prep$Xs, prep$ys)))) / prep$n
  if (!is.finite(lmax) || lmax <= 0) {
    stop("cannot build a lambda path: all cross-product X'y are zero",
         call. = FALSE)
  }
  exp(seq(log(lmax), log(lmax * lambda.min.ratio), length.out = nlambda))
}

.cg_fit_one <- function(prep, lambda, max_iter, tol,
                        session = NULL, warm_start = FALSE) {
  # session = NULL is the original one-shot path (kept bit-identical for the
  # numerical reference). With a session, the design stays resident across the
  # whole lambda path; cold-start fits are bit-identical to the one-shot solve.
  res <- if (is.null(session)) {
    lasso_fista(prep$Xs, prep$ys, lambda,
                max_iter = max_iter, tol = tol)
  } else {
    rcupp::session_fit(session, lambda, warm_start = warm_start,
                       max_iter = max_iter, tol = tol)
  }
  beta_s <- as.numeric(res$beta)
  beta <- beta_s / prep$xs
  a0 <- prep$ym - sum(prep$xm * beta)
  list(beta = beta, a0 = a0,
       iterations = res$iterations, converged = res$converged,
       objective = res$objective, backend = res$backend)
}

.cg_resolve_lambda <- function(prep, lambda, nlambda, lambda.min.ratio) {
  if (is.null(lambda)) {
    ratio <- if (is.null(lambda.min.ratio)) {
      if (prep$n < prep$p) 0.01 else 1e-4
    } else {
      lambda.min.ratio
    }
    return(.cg_lambda_seq(prep, nlambda, ratio))
  }
  if (!is.numeric(lambda) || length(lambda) < 1L || anyNA(lambda) ||
        any(lambda < 0)) {
    stop("'lambda' must be NULL or a numeric vector of non-negative values",
         call. = FALSE)
  }
  sort(as.numeric(lambda), decreasing = TRUE)
}

#' Fit a glmnet-style Gaussian lasso path
#'
#' A glmnet-compatible front end over the GPU FISTA solver in \pkg{rcupp}.
#' Fits a decreasing sequence of \code{lambda} values, with optional
#' standardisation and an intercept, and returns an object supporting
#' [coef.cuda_glmnet()] / [predict.cuda_glmnet()] with glmnet-style \code{s}
#' selection. Use [cv.cuda_glmnet()] to choose \code{lambda} by
#' cross-validation.
#'
#' Unlike [cuda_lasso()] (the single-fixed-lambda numerical reference entry
#' point), this function standardises/centers by default and maps coefficients
#' back to the original scale, so results are directly comparable to
#' \code{glmnet::glmnet()} for the Gaussian family.
#'
#' @section Numerical notes:
#' The objective is \code{1/(2n)||y - a0 - X beta||^2 + lambda||beta||_1}.
#' Standardisation uses the population (\code{1/n}) scale. The path is fitted
#' through a single \pkg{rcupp} session so the (standardised) design matrix
#' stays resident across every lambda. By default (\code{warm_start = FALSE})
#' each lambda is \bold{cold-started} and is bit-identical to a one-shot
#' \code{rcupp::lasso_fista()} solve; set \code{warm_start = TRUE} to seed each
#' fit from the previous solution (fewer iterations, same optimum, not
#' bit-identical). FISTA and the coordinate descent of glmnet solve the same
#' objective but need not agree bit-for-bit.
#'
#' @param x Numeric \code{n x p} design matrix (no missing values).
#' @param y Numeric response of length \code{n}.
#' @param lambda Optional numeric vector of penalties (sorted decreasing
#'   internally). If \code{NULL} (default), a log-spaced path of length
#'   \code{nlambda} is generated from \code{max(abs(crossprod(X, y)))/n} down to
#'   \code{lambda.max * lambda.min.ratio}.
#' @param nlambda Number of lambda values when \code{lambda = NULL}.
#' @param lambda.min.ratio Smallest lambda as a fraction of the largest.
#'   Defaults to \code{0.01} when \code{n < p}, else \code{1e-4}
#'   (the glmnet rule).
#' @param standardize Scale columns to unit population variance (default
#'   \code{TRUE}).
#' @param intercept Fit an unpenalised intercept by centering (default
#'   \code{TRUE}).
#' @param max_iter Maximum FISTA iterations per lambda.
#' @param tol Relative convergence tolerance.
#' @param warm_start Logical; seed each lambda fit from the previous solution
#'   on the same session. Default \code{FALSE} (cold start, bit-identical to a
#'   one-shot solve). Warm starts are opt-in because they are not bit-identical.
#' @param ... Ignored (reserved for future glmnet-compatible arguments).
#'
#' @return An object of class \code{"cuda_glmnet"}: a list with \code{beta}
#'   (\code{p x L} coefficient matrix in original scale),
#'   \code{a0} (intercepts), \code{lambda}, \code{nzero}, \code{dev.ratio},
#'   \code{nulldev},
#'   \code{iterations}, \code{converged}, \code{backend}, \code{nobs},
#'   \code{nvars}, \code{standardize}, \code{intercept} and \code{call}.
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(60 * 5), 60, 5)
#' y <- drop(X %*% c(2, -1, 0, 0, 0)) + rnorm(60, sd = 0.5)
#' fit <- cuda_glmnet(X, y, nlambda = 10)
#' coef(fit, s = fit$lambda[5])
#' predict(fit, X[1:3, ], s = fit$lambda[5])
#'
#' @export
cuda_glmnet <- function(x, y, lambda = NULL, nlambda = 100L,
                        lambda.min.ratio = NULL, standardize = TRUE,
                        intercept = TRUE, max_iter = 1000L, tol = 1e-6,
                        warm_start = FALSE, ...) {
  y <- .cg_check_xy(x, y)
  .cg_check_scalars(max_iter, tol, nlambda, lambda.min.ratio)
  .cg_check_flags(standardize, intercept)
  if (!is.logical(warm_start) || length(warm_start) != 1L ||
        is.na(warm_start)) {
    stop("'warm_start' must be TRUE or FALSE", call. = FALSE)
  }
  if (storage.mode(x) != "double") storage.mode(x) <- "double"

  prep <- .cg_prep(x, y, standardize, intercept)
  lambda <- .cg_resolve_lambda(prep, lambda, nlambda, lambda.min.ratio)
  L <- length(lambda)

  # One session for the whole path: prep$Xs / prep$ys are fixed across all
  # lambda values, so they are uploaded to the device exactly once. Cold-start
  # fits (warm_start = FALSE) stay bit-identical to the one-shot reference.
  s <- rcupp::lasso_session()
  rcupp::session_set_data(s, prep$Xs, prep$ys)
  on.exit(rcupp::session_destroy(s), add = TRUE)

  fits <- lapply(lambda, function(l)
    .cg_fit_one(prep, l, max_iter, tol, session = s, warm_start = warm_start))
  # matrix() keeps beta a p x L matrix even when p == 1 (vapply would collapse).
  beta <- matrix(unlist(lapply(fits, `[[`, "beta"), use.names = FALSE),
                 nrow = prep$p, ncol = L)
  a0 <- vapply(fits, function(f) f$a0, numeric(1))

  rn <- colnames(x)
  rownames(beta) <- rn
  colnames(beta) <- paste0("s", seq_len(L) - 1L)

  dev <- vapply(seq_len(L), function(j) {
    r <- y - (a0[j] + drop(x %*% beta[, j, drop = FALSE]))
    sum(r^2)
  }, numeric(1))
  nulldev <- if (intercept) sum((y - mean(y))^2) else sum(y^2)
  dev.ratio <- if (nulldev > 0) 1 - dev / nulldev else rep(0, L)

  structure(
    list(
      beta       = beta,
      a0         = a0,
      lambda     = lambda,
      nzero      = colSums(abs(beta) > 1e-8),
      dev.ratio  = dev.ratio,
      nulldev    = nulldev,
      iterations = vapply(fits, function(f) as.numeric(f$iterations),
                          numeric(1)),
      converged  = vapply(fits, function(f) as.integer(f$converged),
                          integer(1)),
      backend    = fits[[1L]]$backend,
      nobs       = prep$n,
      nvars      = prep$p,
      standardize = standardize,
      intercept  = intercept,
      call       = match.call()
    ),
    class = "cuda_glmnet"
  )
}

#' Cross-validation for a cuda_glmnet lasso path
#'
#' K-fold cross-validation over a \bold{shared} lambda grid (generated on the
#' full data), with each fold re-standardised on its training rows only, to
#' avoid leakage. Mirrors \code{glmnet::cv.glmnet()} for the Gaussian family.
#'
#' @param x Numeric \code{n x p} design matrix.
#' @param y Numeric response of length \code{n}.
#' @param lambda Optional numeric vector of penalties; shared across folds. If
#'   \code{NULL}, generated on the full data (see [cuda_glmnet()]).
#' @param nfolds Number of folds (default 10). Ignored when \code{foldid} is
#'   supplied.
#' @param foldid Optional integer vector of length \code{n} assigning each
#'   observation to a fold. If \code{NULL}, a balanced random assignment is
#'   drawn (use \code{set.seed()} for reproducibility).
#' @param nlambda,lambda.min.ratio Path controls when \code{lambda = NULL}.
#' @param standardize,intercept Passed to [cuda_glmnet()].
#' @param max_iter,tol Solver controls.
#' @param warm_start Logical; passed to [cuda_glmnet()] and to each fold fit.
#'   Default \code{FALSE} (cold start).
#' @param ... Ignored (reserved for future arguments).
#'
#' @return An object of class \code{"cv.cuda_glmnet"}: a list with
#'   \code{lambda}, \code{cvm} (mean CV error), \code{cvsd}, \code{cvup},
#'   \code{cvlo}, \code{nzero}, \code{lambda.min}, \code{lambda.1se},
#'   \code{index}, \code{fit} (the full-data \code{cuda_glmnet} object),
#'   \code{nfolds} and \code{call}. \code{lambda.1se} is the largest lambda
#'   with \code{cvm <= min(cvm) + cvsd[min]}, as in glmnet.
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(60 * 5), 60, 5)
#' y <- drop(X %*% c(2, -1, 0, 0, 0)) + rnorm(60, sd = 0.5)
#' cv <- cv.cuda_glmnet(X, y, nlambda = 8, nfolds = 3)
#' cv$lambda.min
#' coef(cv, s = "lambda.1se")
#'
#' @export
cv.cuda_glmnet <- function(x, y, lambda = NULL, nfolds = 10L, foldid = NULL,
                           nlambda = 100L, lambda.min.ratio = NULL,
                           standardize = TRUE, intercept = TRUE,
                           max_iter = 1000L, tol = 1e-6,
                           warm_start = FALSE, ...) {
  y <- .cg_check_xy(x, y)
  .cg_check_scalars(max_iter, tol, nlambda, lambda.min.ratio)
  .cg_check_flags(standardize, intercept)
  if (!is.logical(warm_start) || length(warm_start) != 1L ||
        is.na(warm_start)) {
    stop("'warm_start' must be TRUE or FALSE", call. = FALSE)
  }
  if (storage.mode(x) != "double") storage.mode(x) <- "double"
  n <- nrow(x)

  if (!is.numeric(nfolds) || length(nfolds) != 1L ||
        !is.finite(nfolds) || nfolds < 2 || nfolds > n) {
    stop("'nfolds' must be a single integer between 2 and nrow(x)",
         call. = FALSE)
  }
  nfolds <- as.integer(nfolds)
  if (is.null(foldid)) {
    foldid <- sample(rep_len(seq_len(nfolds), n))
  } else {
    foldid <- as.integer(foldid)
    if (length(foldid) != n || anyNA(foldid) || any(foldid < 1L)) {
      stop("'foldid' must have length nrow(x), with no NA and positive values",
           call. = FALSE)
    }
    if (length(unique(foldid)) < 2L) {
      stop("'foldid' must contain at least two distinct folds", call. = FALSE)
    }
    nfolds <- max(foldid)
  }

  prep_full <- .cg_prep(x, y, standardize, intercept)
  lambda <- .cg_resolve_lambda(prep_full, lambda, nlambda, lambda.min.ratio)
  L <- length(lambda)

  fit <- cuda_glmnet(x, y, lambda = lambda, nlambda = nlambda,
                     standardize = standardize, intercept = intercept,
                     max_iter = max_iter, tol = tol, warm_start = warm_start)

  pred <- matrix(NA_real_, n, L)
  for (k in seq_len(nfolds)) {
    te <- which(foldid == k)
    tr <- setdiff(seq_len(n), te)
    pk <- .cg_prep(x[tr, , drop = FALSE], y[tr], standardize, intercept)
    # One session per fold covers that fold's whole lambda path; each fold's
    # training design is fixed across its L fits, so upload once and reuse.
    sk <- rcupp::lasso_session()
    rcupp::session_set_data(sk, pk$Xs, pk$ys)
    for (j in seq_len(L)) {
      f <- .cg_fit_one(pk, lambda[j], max_iter, tol,
                       session = sk, warm_start = warm_start)
      pred[te, j] <- f$a0 + drop(x[te, , drop = FALSE] %*% f$beta)
    }
    rcupp::session_destroy(sk)
  }

  err <- (pred - y)^2
  cvm <- colMeans(err)
  cvsd <- apply(err, 2, stats::sd) / sqrt(n)
  i_min <- which.min(cvm)
  i_1se <- min(which(cvm <= cvm[i_min] + cvsd[i_min]))

  structure(
    list(
      lambda      = lambda,
      cvm         = cvm,
      cvsd        = cvsd,
      cvup        = cvm + cvsd,
      cvlo        = cvm - cvsd,
      nzero       = fit$nzero,
      lambda.min  = lambda[i_min],
      lambda.1se  = lambda[i_1se],
      index       = c(min = i_min, `1se` = i_1se),
      name        = "Mean-Squared Error",
      fit         = fit,
      nfolds      = nfolds,
      call        = match.call()
    ),
    class = "cv.cuda_glmnet"
  )
}
