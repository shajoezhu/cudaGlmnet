# ---------------------------------------------------------------------------
# Optional framework registrations at load time. Each framework is a Suggests
# dependency; we only attempt registration when its namespace is present, and
# wrap every attempt in tryCatch so version / API drift never prevents package
# load. SuperLearner needs no registration: it discovers SL.cudaGlmnet by name.
# ---------------------------------------------------------------------------

.onLoad <- function(libname, pkgname) {
  if (requireNamespace("mlr3", quietly = TRUE)) {
    tryCatch(.register_mlr3_learner(),
             error = function(e) invisible(FALSE))
  }
  if (requireNamespace("parsnip", quietly = TRUE)) {
    tryCatch(.register_parsnip_engine(),
             error = function(e) invisible(FALSE))
  }
  invisible(NULL)
}
