# Part B: a framework-facing subset-ridge fitter that reuses ONE persistent
# rcupp device context across calls, plus an mlr3 regression learner wrapping
# it. All functions here are INTERNAL (no roxygen @export): registration from
# .onLoad in zzz.R, and the learner is reached by its id. We deliberately do
# NOT run roxygenise on cudaGlmnet, to avoid rewriting the package's NAMESPACE
# / man pages while a parallel process edits them.

# Convert R 1-based indices to the 0-based rcupp C ABI convention and validate.
.ctx_subset_idx <- function(idx_list, n) {
  if (!is.list(idx_list) || length(idx_list) < 1L) {
    stop("'idx_list' must be a non-empty list of index vectors", call. = FALSE)
  }
  sizes <- lengths(idx_list)
  if (any(sizes < 1L) || length(unique(sizes)) != 1L) {
    stop("all subsets in 'idx_list' must have the same positive size",
         call. = FALSE)
  }
  lapply(idx_list, function(idx) {
    if (!is.numeric(idx) || anyNA(idx) || any(idx != floor(idx)) ||
          any(idx < 1 | idx > n)) {
      stop("subset indices must be whole numbers in [1, nrow(x)]",
           call. = FALSE)
    }
    as.integer(idx) - 1L
  })
}

# Internal fitter: fit closed-form ridge on B row-subsets of one panel, using
# a caller-supplied persistent ctx (from rcupp::ctx_create()). Same arithmetic
# as cuda_resident_subsets(), but the ctx lifetime belongs to the caller so it
# can be reused across train()/fold calls.
cuda_ctx_subset_ridge <- function(ctx, x, y, idx_list, lambda,
                                  strategy = "resident", method = "auto") {
  if (!is.matrix(x) || !is.numeric(x) || anyNA(x)) {
    stop("'x' must be a numeric matrix without missing values", call. = FALSE)
  }
  y <- as.numeric(y)
  if (length(y) != nrow(x) || anyNA(y)) {
    stop("'y' must be numeric of length nrow(x), without NA", call. = FALSE)
  }
  if (!is.numeric(lambda) || length(lambda) != 1L ||
        !is.finite(lambda) || lambda < 0) {
    stop("'lambda' must be a single non-negative number", call. = FALSE)
  }
  idx0 <- .ctx_subset_idx(idx_list, nrow(x))
  ctx_resident_subsets_cpp(ctx, x, y, idx0, as.double(lambda),
                           strategy, method)
}

# ---- mlr3 learner --------------------------------------------------------

.register_mlr3_subset_learner <- function() {
  if (!requireNamespace("mlr3", quietly = TRUE)) return(invisible(FALSE))
  if (!requireNamespace("rcupp", quietly = TRUE)) return(invisible(FALSE))
  # Runtime guard: an older rcupp without ctx_create() degrades gracefully.
  if (!exists("ctx_create", envir = asNamespace("rcupp"), inherits = FALSE)) {
    return(invisible(FALSE))
  }
  if ("regr.cuda_subset_ridge" %in% mlr3::mlr_learners$keys()) {
    return(invisible(TRUE))
  }

  # Fixed workload constants (p >> m repeated subsets). Deterministic seed so
  # train() is reproducible across folds; no tuning surface for this demo.
  N_SUBSETS <- 8L
  SUBSET_FRAC <- 0.5
  LAMBDA <- 0.2

  LearnerRegrCudaSubsetRidge <- R6::R6Class(
    "LearnerRegrCudaSubsetRidge",
    inherit = mlr3::LearnerRegr,
    public = list(
      initialize = function() {
        super$initialize(
          id = "regr.cuda_subset_ridge",
          packages = c("cudaGlmnet", "rcupp", "mlr3"),
          feature_types = c("numeric", "integer"),
          predict_types = "response",
          properties = character(),
          man = "cudaGlmnet::mlr3_subset_ridge"
        )
      }
    ),
    private = list(
      .ctx = NULL,
      .train = function(task) {
        data <- task$data()
        x <- as.matrix(data[, task$feature_names, drop = FALSE])
        y <- as.numeric(data[[task$target_names]])
        storage.mode(x) <- "double"
        n <- nrow(x)
        m <- max(2L, as.integer(round(SUBSET_FRAC * n)))
        if (m >= n) m <- max(2L, n - 1L)
        set.seed(1234L)
        idx <- lapply(seq_len(N_SUBSETS), function(b) sample.int(n, m))
        # ONE context reused across every train()/fold of this learner instance.
        if (is.null(private$.ctx)) private$.ctx <- rcupp::ctx_create()
        fit <- cuda_ctx_subset_ridge(private$.ctx, x, y, idx,
                                     lambda = LAMBDA,
                                     strategy = "resident", method = "auto")
        if (any(fit$info != 0L)) {
          stop("subset solve failed (info: ",
               paste(unique(fit$info[fit$info != 0L]), collapse = ","), ")")
        }
        list(beta = rowMeans(fit$beta), colnames = colnames(x),
             method = fit$method, sys_dim = fit$sys_dim)
      },
      .predict = function(task) {
        data <- task$data()
        x <- as.matrix(data[, task$feature_names, drop = FALSE])
        model <- self$model
        if (!is.null(model$colnames)) colnames(x) <- model$colnames
        pred <- as.vector(x %*% model$beta)
        mlr3::PredictionRegr$new(task = task, response = pred)
      }
    )
  )
  mlr3::mlr_learners$add("regr.cuda_subset_ridge",
                         LearnerRegrCudaSubsetRidge$new)
  invisible(TRUE)
}
