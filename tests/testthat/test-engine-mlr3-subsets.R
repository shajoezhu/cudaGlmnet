# mlr3 regr.cuda_subset_ridge end-to-end (register -> train -> predict -> CV).

test_that("regr.cuda_subset_ridge registers and trains/predicts", {
  skip_if_not_installed("mlr3")
  skip_if_not_installed("R6")
  skip_if_not_installed("rcupp")
  if (!exists("ctx_create", envir = asNamespace("rcupp"), inherits = FALSE)) {
    skip("rcupp without ctx_create(); Part B needs the new rcupp (>= Part A)")
  }
  cudaGlmnet:::.register_mlr3_subset_learner()
  expect_true("regr.cuda_subset_ridge" %in% mlr3::mlr_learners$keys())

  set.seed(42)
  n <- 90L
  p <- 12L
  X <- matrix(rnorm(n * p), n, p)
  y <- drop(X %*% c(rep(1, 3), rep(0, p - 3L))) + rnorm(n, sd = 0.5)
  task <- mlr3::as_task_regr(data.frame(y = y, X = X), target = "y")

  lrn <- mlr3::lrn("regr.cuda_subset_ridge")
  lrn$train(task)
  pred <- lrn$predict(task)
  expect_s3_class(pred, "PredictionRegr")
  expect_length(pred$response, n)
  expect_true(all(is.finite(pred$response)))
})

test_that("regr.cuda_subset_ridge runs 3-fold CV with one ctx", {
  skip_if_not_installed("mlr3")
  skip_if_not_installed("R6")
  skip_if_not_installed("rcupp")
  if (!exists("ctx_create", envir = asNamespace("rcupp"), inherits = FALSE)) {
    skip("rcupp without ctx_create()")
  }
  cudaGlmnet:::.register_mlr3_subset_learner()
  set.seed(7)
  n <- 90L
  p <- 10L
  X <- matrix(rnorm(n * p), n, p)
  y <- drop(X %*% c(1, -1, rep(0, p - 2L))) + rnorm(n, sd = 0.5)
  task <- mlr3::as_task_regr(data.frame(y = y, X = X), target = "y")
  rsmp <- mlr3::rsmp("cv", folds = 3L)
  learners <- list(mlr3::lrn("regr.cuda_subset_ridge"))
  grid <- mlr3::benchmark_grid(task, learners, rsmp)
  bm <- mlr3::benchmark(grid)
  sc <- bm$score(mlr3::msr("regr.mse"))
  expect_equal(nrow(sc), 3L)
  expect_true(all(is.finite(sc$regr.mse)))
})
