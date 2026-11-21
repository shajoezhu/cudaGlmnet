#' @importFrom rcupp lasso_fista
#' @noRd
NULL

# ---------------------------------------------------------------------------
# glmnet-compatible front end. All new API; cuda_lasso() is deliberately left
# untouched so the demo, the three benchmarks and lasso_fista_cpu() keep their
# exact numerical reference contract.
#
# lambda / alpha follow glmnet's Gaussian convention. glmnet fits on an
# RMS-standardised response (y/s, s = sqrt(mean(y^2))) and reports lambda and
# coefficients in raw-y units, which in raw-y units is the penalty
#   lambda * [ (1/s)(1-alpha)/2 ||beta||_2^2 + alpha ||beta||_1 ].
# cuda_glmnet() keeps the clean, data-independent standard elastic net
#   lambda_c * [ (1-alpha_c)/2 ||beta||_2^2 + alpha_c ||beta||_1 ]
# internally and maps the glmnet-convention (lambda, alpha) to
# (lambda_c, alpha_c) via .cg_glmnet_to_cuda(), so a given (lambda, alpha)
# reproduces glmnet::glmnet(lambda, alpha). At alpha = 1 the map is the
# identity, leaving the lasso path bit-identical to earlier versions.
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

.cg_lambda_seq <- function(prep, nlambda, lambda.min.ratio, alpha = 1) {
  # glmnet-style KKT at beta = 0: |X'y/n| <= lambda * alpha, so the largest
  # useful penalty is base / alpha. Floor tiny alpha so pure ridge (alpha = 0)
  # does not blow the path up; alpha = 1 leaves the original grid unchanged.
  a <- max(alpha, 1e-3)
  lmax <- max(abs(as.vector(crossprod(prep$Xs, prep$ys)))) / prep$n / a
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

.cg_resolve_lambda <- function(prep, lambda, nlambda, lambda.min.ratio,
                               alpha = 1) {
  if (is.null(lambda)) {
    ratio <- if (is.null(lambda.min.ratio)) {
      if (prep$n < prep$p) 0.01 else 1e-4
    } else {
      lambda.min.ratio
    }
    return(.cg_lambda_seq(prep, nlambda, ratio, alpha))
  }
  if (!is.numeric(lambda) || length(lambda) < 1L || anyNA(lambda) ||
        any(lambda < 0)) {
    stop("'lambda' must be NULL or a numeric vector of non-negative values",
         call. = FALSE)
  }
  sort(as.numeric(lambda), decreasing = TRUE)
}

# Map glmnet-convention (lambda, alpha) to cudaGlmnet's standard elastic net.
#
# glmnet solves the Gaussian elastic net on an RMS-standardised response
# (y/s, s = sqrt(mean(y^2))) and reports lambda / coefficients in raw-y units.
# In raw-y units that is
#   lambda_g * [ (1/s)(1-a_g)/2 ||b||^2 + a_g ||b||_1 ],
# whereas cudaGlmnet's (clean, data-independent) standard objective is
#   lambda_c * [ (1-a_c)/2 ||b||^2 + a_c ||b||_1 ].
# Equating the ridge and lasso coefficients gives the exact conversion
#   a_c       = s*a_g / (1 - a_g + s*a_g)
#   lambda_c  = lambda_g * (1 - a_g + s*a_g) / s
# so cuda_glmnet(lambda_g, a_g) reproduces glmnet::glmnet(lambda_g, a_g).
# Endpoints: a_g = 1 -> (a_c, lambda_c) = (1, lambda_g) (identity, lasso);
#            a_g = 0 -> (a_c, lambda_c) = (0, lambda_g / s) (pure ridge).
# A single scalar on lambda cannot work for 0 < a_g < 1, because the ridge and
# lasso components need different scalings; alpha must be remapped too.
.cg_glmnet_to_cuda <- function(prep, alpha) {
  s <- sqrt(mean(prep$ys^2))
  if (!is.finite(s) || s <= 0) s <- 1  # degenerate response: any scale gives b = 0
  denom <- 1 - alpha + s * alpha
  list(alpha_c = s * alpha / denom,
       lambda_scale = denom / s,
       s = s)
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
#' \code{lambda} and \code{alpha} follow \code{glmnet}'s Gaussian convention.
#' glmnet fits on an RMS-standardised response (\code{y/s},
#' \code{s = sqrt(mean(y^2))}) and reports lambda / coefficients in raw-y
#' units, so its penalty in raw-y units is
#' \code{lambda[(1/s)(1-alpha)/2 ||beta||_2^2 + alpha ||beta||_1]}. To be
#' drop-in compatible without baking that data-dependent factor into the
#' solver, \code{cuda_glmnet()} keeps the standard elastic net
#' \code{lambda_c[(1-alpha_c)/2 ||beta||_2^2 + alpha_c ||beta||_1]} internally
#' and maps \code{(lambda, alpha)} to \code{(lambda_c, alpha_c)} via
#' \code{alpha_c = s*alpha/(1-alpha+s*alpha)} and
#' \code{lambda_c = lambda*(1-alpha+s*alpha)/s}. A given \code{(lambda, alpha)}
#' therefore reproduces \code{glmnet::glmnet(lambda, alpha)}. At
#' \code{alpha = 1} the map is the identity (lasso unchanged); at
#' \code{alpha = 0} it is \code{lambda_c = lambda/s}. A scalar rescaling of
#' \code{lambda} alone cannot work for \code{0 < alpha < 1}, because the ridge
#' and lasso penalties need different scalings.
#'
#' Fitting is done on centered/standardised \code{(Xs, ys)} with the population
#' (\code{1/n}) scale, then coefficients are mapped back to the original scale.
#' [enet_fista_cpu()] is the pure-R oracle for the \emph{standard} objective;
#' it must be called with the remapped \code{(lambda_c, alpha_c)} to reproduce a
#' \code{cuda_glmnet()} fit. For \code{alpha = 1} the path is fitted through a
#' single \pkg{rcupp} lasso session (cold starts, \code{warm_start = FALSE},
#' stay bit-identical to a one-shot \code{rcupp::lasso_fista()} solve). For
#' \code{alpha < 1} it is fitted by a downstream \code{Rcpp} + \pkg{rcupp}
#' loop over resident device operators, with the design uploaded once. In
#' both cases \code{warm_start = TRUE} seeds each fit from the previous
#' solution (fewer iterations, same optimum, not bit-identical).
#'
#' @param x Numeric \code{n x p} design matrix (no missing values).
#' @param y Numeric response of length \code{n}.
#' @param lambda Optional numeric vector of penalties in glmnet's convention
#'   (sorted decreasing internally). If \code{NULL} (default), a log-spaced
#'   path of length \code{nlambda} is generated from
#'   \code{max(abs(crossprod(X, y)))/n} down to
#'   \code{lambda.max * lambda.min.ratio}. For \code{alpha < 1} the largest
#'   penalty is divided by \code{alpha} (glmnet-style KKT at \code{beta = 0}).
#'   The path is returned unchanged (glmnet convention); the internal solve
#'   uses the remapped values (see \code{Numerical notes}).
#' @param nlambda Number of lambda values when \code{lambda = NULL}.
#' @param lambda.min.ratio Smallest lambda as a fraction of the largest.
#'   Defaults to \code{0.01} when \code{n < p}, else \code{1e-4}
#'   (the glmnet rule).
#' @param standardize Scale columns to unit population variance (default
#'   \code{TRUE}).
#' @param intercept Fit an unpenalised intercept by centering (default
#'   \code{TRUE}).
#' @param alpha Elastic-net mixing in \code{[0, 1]} in glmnet's convention;
#'   \code{1} (default) is the lasso, \code{0} is pure ridge. The default keeps
#'   the lasso path bit-identical to earlier versions. See
#'   \code{Numerical notes} for the (data-dependent) remap applied for
#'   \code{alpha < 1}.
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
#'   \code{nvars}, \code{standardize}, \code{intercept}, \code{alpha} (the
#'   glmnet-convention mixing parameter) and \code{call}.
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
                        intercept = TRUE, alpha = 1, max_iter = 1000L,
                        tol = 1e-6, warm_start = FALSE, ...) {
  y <- .cg_check_xy(x, y)
  .cg_check_scalars(max_iter, tol, nlambda, lambda.min.ratio)
  .cg_check_flags(standardize, intercept)
  .cg_check_alpha(alpha)
  if (!is.logical(warm_start) || length(warm_start) != 1L ||
        is.na(warm_start)) {
    stop("'warm_start' must be TRUE or FALSE", call. = FALSE)
  }
  if (storage.mode(x) != "double") storage.mode(x) <- "double"

  prep <- .cg_prep(x, y, standardize, intercept)
  lambda <- .cg_resolve_lambda(prep, lambda, nlambda, lambda.min.ratio,
                               alpha)
  L <- length(lambda)

  # glmnet-convention (lambda, alpha) -> cuda's standard-objective parameters.
  # `lambda` stays in glmnet units for the returned object / coef(s=) lookup;
  # only the internal solve sees the remapped values. At alpha = 1 this is the
  # identity, so the lasso branch below remains bit-identical to old versions.
  conv <- .cg_glmnet_to_cuda(prep, alpha)
  lambda_c <- lambda * conv$lambda_scale

  if (alpha == 1) {
    # One lasso session for the whole path: prep$Xs / prep$ys are fixed across
    # all lambda values, so they are uploaded to the device exactly once.
    # Cold-start fits (warm_start = FALSE) stay bit-identical to the one-shot
    # reference. This branch is the original lasso path, kept unchanged.
    s <- rcupp::lasso_session()
    rcupp::session_set_data(s, prep$Xs, prep$ys)
    on.exit(rcupp::session_destroy(s), add = TRUE)
    fits <- lapply(lambda, function(l) {
      .cg_fit_one(prep, l, max_iter, tol,
                  session = s, warm_start = warm_start)
    })
  } else {
    # Elastic net / ridge: a downstream Rcpp + rcupp loop over resident
    # device operators. The rcupp lasso session is lasso-only, so the
    # algorithm lives here, composed from algorithm-agnostic parts. Solve the
    # standard objective at the remapped (lambda_c, alpha_c) so the fit matches
    # glmnet at the user-supplied (lambda, alpha).
    fits <- .cg_enet_fits(prep, lambda_c, conv$alpha_c, max_iter, tol,
                          warm_start)
  }
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
      alpha      = alpha,
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
#' @param alpha Elastic-net mixing in \code{[0, 1]}; see [cuda_glmnet()].
#'   Default \code{1} (lasso).
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
                           standardize = TRUE, intercept = TRUE, alpha = 1,
                           max_iter = 1000L, tol = 1e-6,
                           warm_start = FALSE, ...) {
  y <- .cg_check_xy(x, y)
  .cg_check_scalars(max_iter, tol, nlambda, lambda.min.ratio)
  .cg_check_flags(standardize, intercept)
  .cg_check_alpha(alpha)
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
  lambda <- .cg_resolve_lambda(prep_full, lambda, nlambda,
                               lambda.min.ratio, alpha)
  L <- length(lambda)

  fit <- cuda_glmnet(x, y, lambda = lambda, nlambda = nlambda,
                     standardize = standardize, intercept = intercept,
                     alpha = alpha, max_iter = max_iter, tol = tol,
                     warm_start = warm_start)

  pred <- matrix(NA_real_, n, L)
  for (k in seq_len(nfolds)) {
    te <- which(foldid == k)
    tr <- setdiff(seq_len(n), te)
    pk <- .cg_prep(x[tr, , drop = FALSE], y[tr], standardize, intercept)
    if (alpha == 1) {
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
    } else {
      # Each fold re-standardises on its training rows only, so its response
      # RMS (and hence the glmnet->cuda remap) is fold-specific: compute it from
      # pk$ys, exactly as glmnet::cv.glmnet() re-standardises per fold.
      conv_k <- .cg_glmnet_to_cuda(pk, alpha)
      fk <- .cg_enet_fits(pk, lambda * conv_k$lambda_scale, conv_k$alpha_c,
                          max_iter, tol, warm_start)
      for (j in seq_len(L)) {
        f <- fk[[j]]
        pred[te, j] <- f$a0 + drop(x[te, , drop = FALSE] %*% f$beta)
      }
    }
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
      alpha       = alpha,
      call        = match.call()
    ),
    class = "cv.cuda_glmnet"
  )
}
