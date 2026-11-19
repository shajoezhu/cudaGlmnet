d <- make_lasso_data(n = 60L, p = 5L, seed = 2L)

test_that("alpha = 1 is bit-identical to the default lasso path", {
  path <- c(0.2, 0.1, 0.05, 0.02)
  cold <- cuda_glmnet(d$X, d$y, lambda = path)
  a1 <- cuda_glmnet(d$X, d$y, lambda = path, alpha = 1)
  expect_identical(a1$beta, cold$beta)
  expect_identical(a1$a0, cold$a0)
  expect_identical(a1$iterations, cold$iterations)
  expect_identical(a1$backend, cold$backend)
  expect_identical(a1$lambda, cold$lambda)
})

test_that("enet_fista_cpu(alpha = 1) reproduces lasso_fista_cpu", {
  ref1 <- lasso_fista_cpu(d$X, d$y, lambda = 0.05)
  ref2 <- enet_fista_cpu(d$X, d$y, lambda = 0.05, alpha = 1)
  expect_equal(ref2$coefficients, ref1$coefficients)
  expect_equal(ref2$objective, ref1$objective)
  expect_equal(ref2$iterations, ref1$iterations)
})

test_that("the default lambda grid is unchanged by alpha = 1", {
  f <- cuda_glmnet(d$X, d$y, nlambda = 8)
  a1 <- cuda_glmnet(d$X, d$y, nlambda = 8, alpha = 1)
  expect_identical(a1$lambda, f$lambda)
})
