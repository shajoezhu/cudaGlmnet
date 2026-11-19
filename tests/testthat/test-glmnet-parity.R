test_that("elastic-net fits agree with glmnet within tolerance", {
  skip_if_not_installed("glmnet")
  d <- make_lasso_data(n = 100L, p = 6L, seed = 4L)
  a <- 0.5
  lam <- 0.05
  fit <- cuda_glmnet(d$X, d$y, lambda = lam, alpha = a)
  g <- glmnet::glmnet(d$X, d$y, alpha = a, lambda = lam,
                      standardize = TRUE, intercept = TRUE)
  # FISTA and glmnet's coordinate descent solve the same objective but use
  # different convergence criteria, so compare fitted values, not bit parity.
  ours <- drop(predict(fit, d$X, s = lam))
  theirs <- drop(as.numeric(predict(g, d$X, s = lam)))
  expect_gt(cor(ours, theirs), 0.99)
  ours_coef <- as.numeric(coef(fit, s = lam))
  theirs_coef <- as.numeric(coef(g, s = lam))
  expect_lt(abs(ours_coef[1L] - theirs_coef[1L]), 0.1)
})
