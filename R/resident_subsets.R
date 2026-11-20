#' Refit many row subsets of a device-resident panel
#'
#' Composes \pkg{rcupp}'s generic device parts (\code{gather_rows}, \code{syrk},
#' \code{potrf}, \code{potrs}) into a downstream ridge solver that keeps ONE
#' full panel resident on the device and refits many row subsets through a
#' device-side gather, without re-uploading the design matrix per subset.
#' This is the "Rcpp glue plus rcupp substrate" pattern: the solver loop lives
#' in this package's \code{src/}, \pkg{rcupp} only supplies parts.
#'
#' For each subset \code{idx} (of common size \code{m}) the function solves the
#' closed-form ridge system
#' \deqn{(X_s^\top X_s + m \lambda I)\beta = X_s^\top y_s,}
#' where \code{X_s = x[idx, , drop = FALSE]} and \code{y_s = y[idx]}. The
#' \code{m*lambda} diagonal is constant across the equal-size subsets, so it is
#' uploaded once. \code{syrk} writes only the lower triangle and \code{potrf} /
#' \code{potrs} use the same \code{uplo}.
#'
#' Two strategies are provided for measuring the value of device residency:
#' \itemize{
#'   \item \code{"resident"} uploads the whole \code{n x p} panel once and
#'     selects each subset with \code{rcupp_d_gather_rows} (device to device).
#'   \item \code{"cold"} assembles each \code{m x p} subset on the host and
#'     re-uploads it on every refit (the old \code{set_data} cost). It is an
#'     ROI baseline, not a different algorithm: both produce identical betas.
#' }
#' Both strategies share a single \pkg{rcupp} context, so the difference is
#' the host-to-device transfer, not context/handle setup.
#'
#' @section Scope and honesty:
#' This is a \strong{parts demo of data-movement amortization}, not a
#' performance claim about closed-form ridge. The \code{p x p} Gram matrix and
#' Cholesky are demo-scale; a production \code{p >> n} path would use the
#' dual/Woodbury form. All subsets must currently have the SAME size \code{m},
#' \code{x} must be a dense numeric matrix, and \code{lambda} must be positive
#' enough to keep the system positive definite (an \code{info > 0} is returned
#' otherwise). Requires an \pkg{rcupp} built with the B-stage device parts.
#'
#' @param x Numeric \code{n x p} design matrix (no missing values).
#' @param y Numeric response of length \code{n}.
#' @param idx_list A non-empty list of integer index vectors. Each vector gives
#'   \strong{1-based} rows of \code{x} (R convention); all vectors must have
#'   the same positive length \code{m}, values in \code{[1, nrow(x)]}.
#' @param lambda A single non-negative ridge penalty. The diagonal added to the
#'   Gram matrix is \code{m * lambda}.
#' @param strategy Either \code{"resident"} (default) or \code{"cold"}.
#'
#' @return A list with \code{beta} (\code{p x B} matrix, one column per
#'   subset), \code{info} (integer vector; \code{0} = Cholesky ok),
#'   \code{strategy}, \code{m}, \code{diag} (\code{m * lambda}) and \code{cuda}
#'   (whether a CUDA backend was used). \code{beta} from \code{"resident"} and
#'   \code{"cold"} are numerically identical.
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(60 * 6), 60, 6)
#' y <- drop(X %*% c(1, -1, 0.5, 0, 0, 0)) + rnorm(60, sd = 0.5)
#' idx_list <- replicate(4L, sample.int(60L, 30L), simplify = FALSE)
#' fit <- cuda_resident_subsets(X, y, idx_list, lambda = 0.2)
#' dim(fit$beta)
#'
#' @export
cuda_resident_subsets <- function(x, y, idx_list, lambda,
                                  strategy = c("resident", "cold")) {
  if (!is.matrix(x) || !is.numeric(x) || anyNA(x)) {
    stop("'x' must be a numeric matrix without missing values", call. = FALSE)
  }
  y <- as.numeric(y)
  n <- nrow(x)
  if (length(y) != n || anyNA(y)) {
    stop("'y' must be numeric, of length nrow(x), without NA", call. = FALSE)
  }
  if (!is.numeric(lambda) || length(lambda) != 1L ||
        !is.finite(lambda) || lambda < 0) {
    stop("'lambda' must be a single non-negative number", call. = FALSE)
  }
  if (!is.list(idx_list) || length(idx_list) < 1L) {
    stop("'idx_list' must be a non-empty list of index vectors", call. = FALSE)
  }

  # Validate 1-based indices, then convert to the 0-based convention of the
  # rcupp C ABI. All subsets must share one size (pre-allocated Xs/C0).
  sizes <- lengths(idx_list)
  if (any(sizes < 1L)) {
    stop("each subset in 'idx_list' must have positive size", call. = FALSE)
  }
  if (length(unique(sizes)) != 1L) {
    stop("all subsets in 'idx_list' must have the same positive size",
         call. = FALSE)
  }
  idx_list0 <- lapply(idx_list, function(idx) {
    if (!is.numeric(idx) || anyNA(idx) || any(idx != floor(idx))) {
      stop("each subset must be a numeric vector of whole-number indices",
           call. = FALSE)
    }
    if (any(idx < 1 | idx > n)) {
      stop("'idx_list' indices must be integers in [1, nrow(x)]",
           call. = FALSE)
    }
    as.integer(idx) - 1L
  })

  strategy <- match.arg(strategy)

  # Capability probe (not a version guard): translate a missing B-stage
  # R_GetCCallable() registration into an actionable message.
  res <- tryCatch(
    resident_subsets_cpp(x, y, idx_list0, as.double(lambda), strategy),
    error = function(e) {
      msg <- conditionMessage(e)
      if (grepl("not provided by package|rcupp_d_|rcupp_ctx", msg)) {
        stop("cuda_resident_subsets() needs an 'rcupp' built with the B-stage ",
             "device parts (gather_rows / syrk / potrf / potrs). Reinstall ",
             "'rcupp' from a source tree that includes them. Original error: ",
             msg, call. = FALSE)
      }
      stop(msg, call. = FALSE)
    }
  )

  if (!is.null(colnames(x))) {
    rownames(res$beta) <- colnames(x)
  }
  res
}

# lintr's object_usage_linter checks each source file in isolation, so the
# Rcpp-generated wrapper defined in another R file looks like an undefined
# global. Declare it once at package level (see R/enet_core.R).
utils::globalVariables("resident_subsets_cpp")
