#' Extract coefficients from a cuda_lasso fit
#'
#' @param object A \code{cuda_lasso} object.
#' @param ... Ignored.
#' @return A numeric vector of lasso coefficients (named when \code{x} had
#'   column names).
#' @export
coef.cuda_lasso <- function(object, ...) {
  object$coefficients
}

#' Predict from a cuda_lasso fit
#'
#' @param object A \code{cuda_lasso} object.
#' @param newx Numeric matrix with \code{p} columns. Required (no training-data
#'   default in v1).
#' @param ... Ignored.
#' @return A numeric vector of length \code{nrow(newx)}. No intercept is added.
#' @export
predict.cuda_lasso <- function(object, newx, ...) {
  if (missing(newx)) {
    stop("'newx' is required", call. = FALSE)
  }
  newx <- as.matrix(newx)
  if (!is.numeric(newx) || ncol(newx) != length(object$coefficients)) {
    stop("'newx' must be numeric with ", length(object$coefficients),
         " columns", call. = FALSE)
  }
  drop(newx %*% object$coefficients)
}

#' Print a cuda_lasso fit
#'
#' @param x A \code{cuda_lasso} object.
#' @param ... Ignored.
#' @return \code{x}, invisibly.
#' @export
print.cuda_lasso <- function(x, ...) {
  cat("\nCall: ", paste(deparse(x$call), collapse = " "), "\n", sep = "")
  cat("Backend:", x$backend, " | lambda:", format(x$lambda, digits = 6),
      " | iterations:", x$iterations, " | converged:", x$converged, "\n")
  cat("Non-zero coefficients:", sum(abs(x$coefficients) > 1e-8),
      "/", length(x$coefficients), "\n")
  invisible(x)
}

# ---------------------------------------------------------------------------
# glmnet-compatible front end: coef/predict helpers and S3 methods.
# ---------------------------------------------------------------------------

.cg_coef_names <- function(object) {
  rn <- rownames(object$beta)
  if (is.null(rn)) rn <- paste0("V", seq_len(nrow(object$beta)))
  c("(Intercept)", rn)
}

.cg_coef_matrix <- function(object, s) {
  lam <- object$lambda
  vals <- rbind(object$a0, object$beta)
  rownames(vals) <- .cg_coef_names(object)
  if (is.null(s)) {
    colnames(vals) <- paste0("s", seq_along(lam) - 1L)
    return(vals)
  }
  ss <- as.numeric(s)
  if (any(!is.finite(ss)) || any(ss <= 0)) {
    stop("'s' must contain positive lambda values", call. = FALSE)
  }
  if (length(lam) == 1L) {
    out <- matrix(rep(vals[, 1L], length(ss)),
                  nrow = nrow(vals), ncol = length(ss))
  } else {
    xl <- rev(log(pmax(lam, .Machine$double.xmin)))
    out <- vapply(ss, function(sj) {
      vapply(seq_len(nrow(vals)), function(i) {
        stats::approx(x = xl, y = rev(vals[i, ]),
                      xout = log(sj), rule = 2)$y
      }, numeric(1))
    }, numeric(nrow(vals)))
  }
  rownames(out) <- rownames(vals)
  colnames(out) <- paste0("s", seq_along(ss) - 1L)
  out
}

.cg_resolve_s <- function(object, s) {
  if (is.null(s)) return(object$lambda)
  if (is.character(s)) {
    map <- c(lambda.min = object$lambda.min, lambda.1se = object$lambda.1se)
    if (length(s) != 1L || !(s %in% names(map))) {
      stop("'s' must be NULL, a numeric lambda, \"lambda.min\" or ",
           "\"lambda.1se\"", call. = FALSE)
    }
    return(unname(map[[s]]))
  }
  as.numeric(s)
}

#' Coefficients from a cuda_glmnet fit
#'
#' @param object A \code{cuda_glmnet} object.
#' @param s Penalty value(s). \code{NULL} (default) returns the whole path;
#'   numeric values are interpolated on the log-lambda scale (glmnet-style).
#'   Strings such as \code{"lambda.min"} are only valid on a
#'   [cv.cuda_glmnet()] object.
#' @param ... Ignored.
#' @return A \code{(p + 1) x length(s)} numeric matrix; the first row is
#'   \code{"(Intercept)"}.
#' @export
coef.cuda_glmnet <- function(object, s = NULL, ...) {
  if (is.character(s)) {
    stop("string 's' (e.g. \"lambda.min\") is only valid on a ",
         "cv.cuda_glmnet object", call. = FALSE)
  }
  .cg_coef_matrix(object, s)
}

#' Predictions from a cuda_glmnet fit
#'
#' @param object A \code{cuda_glmnet} object.
#' @param newx Numeric matrix with \code{p} columns.
#' @param s Penalty value(s); see [coef.cuda_glmnet()].
#' @param ... Ignored.
#' @return An \code{nrow(newx) x length(s)} numeric matrix (use \code{drop()}
#'   for a single \code{s}).
#' @export
predict.cuda_glmnet <- function(object, newx, s = NULL, ...) {
  if (missing(newx)) {
    stop("'newx' is required", call. = FALSE)
  }
  newx <- as.matrix(newx)
  if (!is.numeric(newx) || ncol(newx) != nrow(object$beta)) {
    stop("'newx' must be numeric with ", nrow(object$beta),
         " columns", call. = FALSE)
  }
  C <- .cg_coef_matrix(object, s)
  pred <- sweep(newx %*% C[-1L, , drop = FALSE], 2, C[1L, ], "+")
  rownames(pred) <- rownames(newx)
  colnames(pred) <- colnames(C)
  pred
}

#' Print a cuda_glmnet fit
#'
#' @param x A \code{cuda_glmnet} object.
#' @param ... Ignored.
#' @return \code{x}, invisibly.
#' @export
print.cuda_glmnet <- function(x, ...) {
  cat("\nCall: ", paste(deparse(x$call), collapse = " "), "\n", sep = "")
  cat("cuda_glmnet: ", x$nobs, " obs, ", x$nvars, " vars, ",
      length(x$lambda), " lambdas | backend: ", x$backend, "\n", sep = "")
  cat("dev.ratio: ", format(min(x$dev.ratio), digits = 3), " .. ",
      format(max(x$dev.ratio), digits = 3), "\n", sep = "")
  invisible(x)
}

#' Coefficients from a cross-validated cuda_glmnet fit
#'
#' @param object A \code{cv.cuda_glmnet} object.
#' @param s \code{"lambda.min"}, \code{"lambda.1se"}, a numeric lambda, or
#'   \code{NULL} for the whole path. Defaults to \code{"lambda.1se"}.
#' @param ... Ignored.
#' @return A \code{(p + 1) x length(s)} numeric matrix.
#' @export
coef.cv.cuda_glmnet <- function(object, s = "lambda.1se", ...) {
  coef(object$fit, s = .cg_resolve_s(object, s), ...)
}

#' Predictions from a cross-validated cuda_glmnet fit
#'
#' @param object A \code{cv.cuda_glmnet} object.
#' @param newx Numeric matrix with \code{p} columns.
#' @param s See [coef.cv.cuda_glmnet()]. Defaults to \code{"lambda.1se"}.
#' @param ... Ignored.
#' @return An \code{nrow(newx) x length(s)} numeric matrix.
#' @export
predict.cv.cuda_glmnet <- function(object, newx, s = "lambda.1se", ...) {
  predict(object$fit, newx, s = .cg_resolve_s(object, s), ...)
}

#' Print a cross-validated cuda_glmnet fit
#'
#' @param x A \code{cv.cuda_glmnet} object.
#' @param ... Ignored.
#' @return \code{x}, invisibly.
#' @export
print.cv.cuda_glmnet <- function(x, ...) {
  cat("\nCall: ", paste(deparse(x$call), collapse = " "), "\n", sep = "")
  cat(x$nfolds, "-fold CV (", x$name, ")\n", sep = "")
  cat("lambda.min = ", format(x$lambda.min, digits = 6),
      "  (cvm = ", format(min(x$cvm), digits = 4), ")\n", sep = "")
  cat("lambda.1se = ", format(x$lambda.1se, digits = 6), "\n", sep = "")
  invisible(x)
}
