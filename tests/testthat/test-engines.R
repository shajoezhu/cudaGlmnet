test_that("SL.cudaGlmnet fits and predicts like a direct cuda_glmnet fit", {
  # This is the only adapter that can be exercised end-to-end locally: it is a
  # plain R function following the SL.* convention and needs no SuperLearner.
  d <- make_lasso_data(n = 80L, p = 6L, seed = 11L)
  newX <- d$X[1:9, , drop = FALSE]
  res <- SL.cudaGlmnet(d$y, d$X, newX, family = gaussian(),
                       obsWeights = rep(1, nrow(d$X)),
                       id = seq_len(nrow(d$X)), nlambda = 10)

  expect_type(res$fit, "list")
  expect_s3_class(res$fit, "SL.cudaGlmnet")
  expect_length(res$pred, nrow(newX))
  expect_true(all(is.finite(res$pred)))

  # Fitting the same object directly and predicting at the same penalty must
  # give numerically identical predictions (cuda_glmnet is deterministic).
  ref <- cuda_glmnet(d$X, d$y, nlambda = 10)
  direct <- as.numeric(predict(ref, newx = newX, s = res$fit$s))
  expect_equal(res$pred, direct, tolerance = 1e-8)

  # The companion S3 predict method returns a vector of the same length.
  p2 <- predict(res$fit, newdata = newX)
  expect_length(p2, nrow(newX))
  expect_equal(as.numeric(p2), res$pred, tolerance = 1e-8)
})

test_that("SL.cudaGlmnet rejects non-Gaussian families", {
  d <- make_lasso_data(n = 40L, p = 4L, seed = 12L)
  expect_error(
    SL.cudaGlmnet(d$y, d$X, d$X, family = binomial(), nlambda = 5),
    "only Gaussian family is supported"
  )
  expect_error(
    SL.cudaGlmnet(d$y, d$X, d$X, family = "binomial", nlambda = 5),
    "only Gaussian family is supported"
  )
})

test_that("registration guards quietly no-op when frameworks are absent", {
  if (!requireNamespace("mlr3", quietly = TRUE)) {
    expect_false(cudaGlmnet:::.register_mlr3_learner())
  }
  if (!requireNamespace("parsnip", quietly = TRUE)) {
    expect_false(cudaGlmnet:::.register_parsnip_engine())
  }
  expect_error(cudaGlmnet:::.register_mlr3_learner(), NA)
  expect_error(cudaGlmnet:::.register_parsnip_engine(), NA)
})

test_that("mlr3 learner registration does not error when mlr3 is present", {
  skip_if_not_installed("mlr3")
  expect_error(cudaGlmnet:::.register_mlr3_learner(), NA)
})

test_that("parsnip engine registration does not error when parsnip present", {
  skip_if_not_installed("parsnip")
  expect_error(cudaGlmnet:::.register_parsnip_engine(), NA)
})

test_that("mlr3 learner trains and predicts end-to-end", {
  skip_if_not_installed("mlr3")
  cudaGlmnet:::.register_mlr3_learner()
  d <- make_lasso_data(n = 120L, p = 5L, seed = 21L)
  df <- data.frame(y = d$y, X = d$X)
  task <- mlr3::as_task_regr(df, target = "y")
  learner <- mlr3::lrn("regr.cuda_glmnet")
  learner$train(task)
  pred <- learner$predict(task)
  expect_s3_class(pred, "PredictionRegr")
  expect_length(pred$response, nrow(d$X))
  expect_true(all(is.finite(pred$response)))
})

test_that("parsnip engine fits and predicts end-to-end", {
  skip_if_not_installed("parsnip")
  cudaGlmnet:::.register_parsnip_engine()
  d <- make_lasso_data(n = 120L, p = 5L, seed = 22L)
  X <- d$X
  colnames(X) <- paste0("V", seq_len(ncol(X)))
  spec <- cuda_enet(mode = "regression", engine = "cuda_glmnet")
  fit <- parsnip::fit_xy(spec, x = X, y = d$y)
  p <- predict(fit, new_data = X[1:5, , drop = FALSE])
  expect_s3_class(p, "data.frame")
  expect_true(".pred" %in% names(p))
  expect_equal(nrow(p), 5L)
  expect_true(all(is.finite(p$.pred)))
})
