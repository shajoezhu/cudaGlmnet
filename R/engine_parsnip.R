# ---------------------------------------------------------------------------
# parsnip / tidymodels engine registration (optional). parsnip is a Suggests
# dependency only; the registration is attempted from .onLoad when its
# namespace is present, and each attempt is wrapped in tryCatch so version /
# API drift never keeps the package from loading. This code is structurally
# validated only; parsnip is not installed locally, so registration has not
# been exercised end-to-end (register -> fit -> predict).
#
# The parsnip model type is deliberately named "cuda_enet", not "cuda_glmnet":
# set_new_model() may create a model-spec function of that name in the caller
# namespace, which must not shadow this package's own cuda_glmnet() solver.
# The registered engine is still called "cuda_glmnet".
# ---------------------------------------------------------------------------

.register_parsnip_engine <- function() {
  if (!requireNamespace("parsnip", quietly = TRUE)) {
    return(invisible(FALSE))
  }
  ok <- tryCatch({
    parsnip::set_new_model("cuda_enet")
    parsnip::set_model_mode("cuda_enet", mode = "regression")
    parsnip::set_model_engine("cuda_enet", mode = "regression",
                              eng = "cuda_glmnet")
    parsnip::set_dependency("cuda_enet", mode = "regression",
                            eng = "cuda_glmnet", pkg = "cudaGlmnet")
    parsnip::set_fit("cuda_enet", mode = "regression",
                     eng = "cuda_glmnet",
                     func = c(fun = "cuda_glmnet"), defaults = list())
    parsnip::set_pred("cuda_enet", mode = "regression",
                      eng = "cuda_glmnet", type = "numeric",
                      func = c(fun = "predict"),
                      args = list(
                        object = quote(object$fit),
                        newx = quote(new_data),
                        s = quote(min(object$fit$lambda))
                      ))
    TRUE
  }, error = function(e) FALSE)
  invisible(ok)
}
