# ---------------------------------------------------------------------------
# mlr3 learner registration (optional). mlr3 is a Suggests dependency only;
# the registration is attempted from .onLoad when its namespace is present,
# and each attempt is wrapped in tryCatch so version / API drift never keeps
# the package from loading.
# .train / .predict MUST be private methods: mlr3 dispatches them via
# get_private(learner)$.train(task), so defining them as public methods lets
# registration succeed but makes train() fail with "non-function".
# R6 injects self / super into methods at runtime; declare them so static
# checks do not mistake them for undefined globals.
# ---------------------------------------------------------------------------

utils::globalVariables(c("self", "super"))

.register_mlr3_learner <- function() {
  if (!requireNamespace("mlr3", quietly = TRUE)) {
    return(invisible(FALSE))
  }
  ok <- tryCatch({
    LearnerRegrCudaGlmnet <- R6::R6Class(
      "LearnerRegrCudaGlmnet",
      inherit = mlr3::LearnerRegr,
      public = list(
        initialize = function() {
          super$initialize(
            id = "regr.cuda_glmnet",
            feature_types = c("integer", "numeric"),
            predict_types = "response",
            properties = character(),
            packages = c("cudaGlmnet"),
            label = "GPU Gaussian Lasso / Elastic Net",
            man = "cudaGlmnet::cuda_glmnet"
          )
        }
      ),
      private = list(
        .train = function(task) {
          x <- as.matrix(task$data(cols = task$feature_names))
          y <- as.numeric(task$truth())
          cuda_glmnet(x, y)
        },
        .predict = function(task) {
          x <- as.matrix(task$data(cols = task$feature_names))
          s <- min(self$model$lambda)
          pred <- as.numeric(predict(self$model, newx = x, s = s))
          mlr3::PredictionRegr$new(task = task, response = pred)
        }
      )
    )
    mlr3::mlr_learners$add("regr.cuda_glmnet", LearnerRegrCudaGlmnet)
    TRUE
  }, error = function(e) FALSE)
  invisible(ok)
}
