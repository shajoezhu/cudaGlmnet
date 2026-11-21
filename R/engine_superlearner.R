# ---------------------------------------------------------------------------
# SuperLearner "SL.*" adapter. SL.cudaGlmnet is a plain R function following
# the SuperLearner naming convention: SuperLearner discovers it by name at
# runtime, and it needs no SuperLearner install to be called directly. This is
# the only adapter exercised end-to-end by the local unit tests.
# ---------------------------------------------------------------------------

.sl_check_family <- function(family) {
  fam <- if (is.list(family)) family$family else as.character(family)[1L]
  if (!is.character(fam) || length(fam) < 1L ||
        !identical(tolower(fam[[1L]]), "gaussian")) {
    stop("SL.cudaGlmnet: only Gaussian family is supported", call. = FALSE)
  }
  invisible(TRUE)
}

#' GPU Gaussian lasso / elastic-net wrapper for SuperLearner
#'
#' A SuperLearner \code{SL.*} wrapper around [cuda_glmnet()]. It fits a
#' Gaussian lasso / elastic-net path on \code{X, Y} and returns predictions
#' for \code{newX}. SuperLearner finds the function by its \code{SL.} name;
#' no SuperLearner installation is needed to call it directly.
#'
#' Only the \strong{Gaussian} family is supported: \code{family = binomial()}
#' and other families are rejected with an error, because binomial and sparse
#' \code{x} are future work in \pkg{cudaGlmnet}. The \code{obsWeights}
#' argument is currently \strong{ignored, not yet supported}, and \code{id}
#' (SuperLearner observation ids) is likewise unused. Predictions are taken
#' at a single penalty; by default the smallest fitted lambda (the
#' least-regularised end of the path) is used.
#'
#' @param Y Numeric response of length \code{nrow(X)}.
#' @param X Numeric \code{n x p} training design (matrix or data frame).
#' @param newX Numeric design to predict on, with \code{p} columns.
#' @param family A Gaussian \code{stats} family (e.g. \code{gaussian()}) or
#'   the string \code{"gaussian"}. Non-Gaussian families are rejected.
#' @param obsWeights Ignored; observation weights are not yet supported.
#' @param id Ignored; SuperLearner observation ids.
#' @param ... Passed to [cuda_glmnet()] (e.g. \code{alpha}, \code{nlambda}).
#' @param s Penalty value at which predictions are made. \code{NULL}
#'   (default) selects the smallest fitted lambda; a numeric lambda is passed
#'   to [predict.cuda_glmnet()].
#'
#' @return A list with \code{pred} (numeric vector of length
#'   \code{nrow(newX)}) and \code{fit} (an object of class
#'   \code{"SL.cudaGlmnet"}, usable with \code{predict.SL.cudaGlmnet()}).
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(60 * 5), 60, 5)
#' Y <- drop(X %*% c(2, -1, 0, 0, 0)) + rnorm(60, sd = 0.5)
#' sl <- SL.cudaGlmnet(Y, X, X, family = gaussian(), nlambda = 10)
#' length(sl$pred)
#'
#' @export
SL.cudaGlmnet <- function(Y, X, newX, family = "gaussian", obsWeights = NULL,
                          id = NULL, ..., s = NULL) {
  .sl_check_family(family)
  X <- as.matrix(X)
  newX <- as.matrix(newX)
  object <- cuda_glmnet(X, Y, ...)
  if (is.null(s)) s <- min(object$lambda)
  pred <- as.numeric(predict(object, newx = newX, s = s))
  fit <- list(object = object, s = s, family = family)
  class(fit) <- "SL.cudaGlmnet"
  list(pred = pred, fit = fit)
}

#' Predict from a SuperLearner cuda_glmnet fit
#'
#' @param object The \code{fit} element returned by \code{SL.cudaGlmnet()}, of
#'   class \code{"SL.cudaGlmnet"}.
#' @param newdata Numeric design to predict on, with \code{p} columns.
#' @param ... Ignored (present for SuperLearner predict conventions).
#' @return A numeric vector of length \code{nrow(newdata)}.
#' @export
predict.SL.cudaGlmnet <- function(object, newdata, ...) {
  as.numeric(predict(object$object, newx = as.matrix(newdata),
                     s = object$s))
}
