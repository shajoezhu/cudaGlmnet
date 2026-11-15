test_that("cuda_lasso rejects invalid inputs", {
  set.seed(1)
  X <- matrix(rnorm(60), 20, 3)
  y <- rnorm(20)

  expect_error(cuda_lasso(as.data.frame(X), y, 0.1), "numeric matrix")
  expect_error(cuda_lasso(matrix("a", 20, 3), y, 0.1), "numeric matrix")

  Xna <- X
  Xna[1, 1] <- NA
  expect_error(cuda_lasso(Xna, y, 0.1), "missing values")

  yna <- y
  yna[2] <- NA
  expect_error(cuda_lasso(X, yna, 0.1), "without NA")

  expect_error(cuda_lasso(X, y[-1], 0.1), "length nrow")
  expect_error(cuda_lasso(X, y, -1), "non-negative")
  expect_error(cuda_lasso(X, y, c(0.1, 0.2)), "single non-negative")
  expect_error(cuda_lasso(X, y, NA_real_), "single non-negative")
})

test_that("S3 methods behave", {
  d <- make_lasso_data(n = 60L, p = 5L, seed = 2L)
  fit <- cuda_lasso(d$X, d$y, lambda = 0.05)
  expect_equal(coef(fit), fit$coefficients)
  expect_length(predict(fit, d$X), nrow(d$X))
  expect_error(predict(fit), "required")
  expect_output(print(fit), "Backend:")
  expect_identical(print(fit), fit)
})

test_that("scalar arguments are validated on both solvers", {
  set.seed(1)
  X <- matrix(rnorm(60), 20, 3)
  y <- rnorm(20)

  for (fit in list(cuda_lasso, lasso_fista_cpu)) {
    expect_error(fit(X, y, 0.1, max_iter = 0L), "positive integer")
    expect_error(fit(X, y, 0.1, max_iter = c(10L, 20L)), "positive integer")
    expect_error(fit(X, y, 0.1, tol = 0), "positive number")
    expect_error(fit(X, y, 0.1, tol = -1e-6), "positive number")
    expect_error(fit(X, y, -1), "non-negative")
  }
})

test_that("lasso_fista_cpu validates x and y", {
  set.seed(1)
  X <- matrix(rnorm(60), 20, 3)
  y <- rnorm(20)

  Xna <- X
  Xna[1, 1] <- NA
  expect_error(lasso_fista_cpu(Xna, y, 0.1), "missing values")
  expect_error(lasso_fista_cpu(matrix("a", 20, 3), y, 0.1),
               "numeric matrix")

  yna <- y
  yna[2] <- NA
  expect_error(lasso_fista_cpu(X, yna, 0.1), "without NA")
  expect_error(lasso_fista_cpu(X, y[-1], 0.1), "length nrow")
})

test_that("predict.cuda_lasso validates newx", {
  d <- make_lasso_data(n = 60L, p = 5L, seed = 2L)
  fit <- cuda_lasso(d$X, d$y, lambda = 0.05)
  expect_error(predict(fit, matrix(0, 4, 3)), "must be numeric with 5 columns")
  expect_error(predict(fit, matrix("a", 4, 5)), "must be numeric")
})
