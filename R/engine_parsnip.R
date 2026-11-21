# ---------------------------------------------------------------------------
# parsnip / tidymodels engine registration (optional). parsnip is a Suggests
# dependency only; the registration is attempted from .onLoad when its
# namespace is present, and each attempt is wrapped in tryCatch so version /
# API drift never keeps the package from loading.
#
# The parsnip model type is deliberately named "cuda_enet", not "cuda_glmnet",
# so the model-spec constructor cuda_enet() cannot shadow this package's own
# cuda_glmnet() solver. The registered engine is still called "cuda_glmnet".
#
# parsnip >= 1.x uses the value=/options= API for set_fit / set_pred /
# set_encoding (older 0.x used func=/defaults=/args=). The registration below
# targets that API and was exercised end-to-end (register -> fit_xy / fit ->
# predict) against parsnip 1.6.1.
# ---------------------------------------------------------------------------

#' Create a GPU elastic-net model specification for parsnip
#'
#' A thin constructor around [parsnip::new_model_spec()] for the
#' \code{"cuda_enet"} model type. The engine registered for it,
#' \code{"cuda_glmnet"}, fits a Gaussian lasso / elastic-net path with
#' [cuda_glmnet()] and predicts at the smallest fitted lambda. Only the
#' \strong{regression} mode and the \strong{Gaussian} family are supported;
#' binomial and sparse designs are future work. There are no tunable
#' parsnip arguments yet: penalty selection (e.g. \code{penalty} /
#' \code{mixture}) is done through [cuda_glmnet()] arguments at fit time, not
#' exposed as model arguments.
#'
#' \pkg{parsnip} must be installed for this constructor to work; the engine
#' itself is registered from \code{.onLoad} only when parsnip is available.
#'
#' @param mode The model mode. Only \code{"regression"} is supported.
#' @param engine The parsnip engine. Only \code{"cuda_glmnet"} is registered.
#'
#' @return A parsnip \code{model_spec} of subclass \code{"cuda_enet"}.
#'
#' @examples
#' \dontrun{
#' spec <- cuda_enet(mode = "regression", engine = "cuda_glmnet")
#' fit <- parsnip::fit_xy(spec, x = x, y = y)
#' predict(fit, new_data = x)
#' }
#'
#' @export
cuda_enet <- function(mode = "regression", engine = "cuda_glmnet") {
  if (!requireNamespace("parsnip", quietly = TRUE)) {
    stop("parsnip must be installed to create a cuda_enet model specification",
         call. = FALSE)
  }
  parsnip::new_model_spec("cuda_enet", args = list(), eng_args = list(),
                          mode = mode, method = NULL, engine = engine)
}

.register_parsnip_engine <- function() {
  if (!requireNamespace("parsnip", quietly = TRUE)) {
    return(invisible(FALSE))
  }
  ok <- tryCatch({
    # set_new_model() errors if the type is already registered in this R
    # session (e.g. a second load_all / library call). Swallow that one error;
    # every set_* call below is idempotent (distinct / is_discordant_info), so
    # re-registering simply re-asserts the same definition.
    tryCatch(parsnip::set_new_model("cuda_enet"),
             error = function(e) invisible(NULL))
    parsnip::set_model_mode("cuda_enet", mode = "regression")
    parsnip::set_model_engine("cuda_enet", mode = "regression",
                              eng = "cuda_glmnet")
    parsnip::set_dependency("cuda_enet", mode = "regression",
                            eng = "cuda_glmnet", pkg = "cudaGlmnet")
    parsnip::set_encoding("cuda_enet", mode = "regression",
                          eng = "cuda_glmnet",
                          options = list(predictor_indicators = "none",
                                         compute_intercept = TRUE,
                                         remove_intercept = TRUE,
                                         allow_sparse_x = FALSE))
    parsnip::set_fit("cuda_enet", mode = "regression",
                     eng = "cuda_glmnet",
                     value = list(
                       interface = "matrix",
                       protect = c("x", "y"),
                       func = c(pkg = "cudaGlmnet", fun = "cuda_glmnet"),
                       defaults = list()
                     ))
    parsnip::set_pred("cuda_enet", mode = "regression",
                      eng = "cuda_glmnet", type = "numeric",
                      value = list(
                        pre = NULL,
                        post = function(x, object) as.numeric(x),
                        func = c(fun = "predict"),
                        args = list(
                          object = quote(object$fit),
                          newx = quote(new_data),
                          s = quote(min(object$fit$lambda))
                        )
                      ))
    TRUE
  }, error = function(e) FALSE)
  invisible(ok)
}
