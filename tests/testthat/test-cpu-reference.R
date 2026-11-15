test_that("CPU reference: a large lambda gives all-zero coefficients", {
  d <- make_lasso_data()
  fit <- lasso_fista_cpu(d$X, d$y, lambda = 1e6)
  expect_true(all(abs(fit$coefficients) < 1e-10))
  expect_gt(fit$iterations, 0L)
  expect_true(is.finite(fit$objective))
})

test_that("CPU reference: objective is finite and non-negative", {
  d <- make_lasso_data(seed = 3L)
  fit <- lasso_fista_cpu(d$X, d$y, lambda = 0.05)
  expect_true(is.finite(fit$objective))
  expect_gte(fit$objective, 0)
  expect_true(fit$converged %in% c(0L, 1L))
  expect_gt(fit$iterations, 0L)
})

test_that("CPU reference: lambda = 0 approaches the OLS solution", {
  d <- make_lasso_data(seed = 5L)
  fit <- lasso_fista_cpu(d$X, d$y, lambda = 0,
                         max_iter = 20000L, tol = 1e-10)
  ols <- unname(coef(lm.fit(d$X, d$y)))
  expect_equal(fit$coefficients, ols, tolerance = 1e-3)
})

test_that("CPU reference: objective decreases as lambda shrinks", {
  d <- make_lasso_data(seed = 9L)
  f_hi <- lasso_fista_cpu(d$X, d$y, lambda = 5)
  f_lo <- lasso_fista_cpu(d$X, d$y, lambda = 0.01)
  expect_lte(f_lo$objective, f_hi$objective)
})

test_that("CPU reference: a zero design matrix gives zero coefficients", {
  set.seed(1)
  X <- matrix(0, 30, 4)
  y <- rnorm(30)
  fit <- lasso_fista_cpu(X, y, lambda = 0.05)
  expect_true(all(fit$coefficients == 0))
  expect_true(is.finite(fit$objective))
})

test_that("both solvers name coefficients when x has column names", {
  d <- make_lasso_data(n = 60L, p = 4L, seed = 17L)
  colnames(d$X) <- paste0("v", seq_len(ncol(d$X)))
  fit <- cuda_lasso(d$X, d$y, lambda = 0.05)
  ref <- lasso_fista_cpu(d$X, d$y, lambda = 0.05)
  expect_named(coef(fit), colnames(d$X))
  expect_named(ref$coefficients, colnames(d$X))
  expect_equal(coef(fit), ref$coefficients, tolerance = 1e-5)
})
