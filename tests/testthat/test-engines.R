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
  # Structural check only: a full register -> CV -> predict comparison against
  # a glmnet engine (roadmap section 9) is left to CI with mlr3 installed.
  expect_error(cudaGlmnet:::.register_mlr3_learner(), NA)
})

test_that("parsnip engine registration does not error when parsnip present", {
  skip_if_not_installed("parsnip")
  # Structural check only; end-to-end engine behaviour is a CI follow-up.
  expect_error(cudaGlmnet:::.register_parsnip_engine(), NA)
})
