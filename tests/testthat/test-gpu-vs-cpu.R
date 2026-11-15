# Always runs: validates the whole R path (Rcpp marshalling, column-major
# layout, iteration semantics) against the independent pure-R oracle. TODAY
# (no nvcc) this compares the CPU reference shim; after a CUDA build it
# compares the real GPU backend.

test_that("fitted path agrees with the CPU reference oracle", {
  d <- make_lasso_data(seed = 7L)
  fit <- cuda_lasso(d$X, d$y, lambda = 0.05)
  ref <- lasso_fista_cpu(d$X, d$y, lambda = 0.05)
  expect_equal(fit$coefficients, ref$coefficients, tolerance = 1e-5)
  expect_equal(fit$objective, ref$objective, tolerance = 1e-6)
  expect_s3_class(fit, "cuda_lasso")
})

test_that("CUDA backend agrees with the oracle (needs a CUDA build)", {
  skip_if_not(isTRUE(rcupp::cuda_available()), "no CUDA backend")
  d <- make_lasso_data(seed = 11L)
  fit <- cuda_lasso(d$X, d$y, lambda = 0.05)
  expect_identical(fit$backend, "cuda")
  ref <- lasso_fista_cpu(d$X, d$y, lambda = 0.05)
  expect_equal(fit$coefficients, ref$coefficients, tolerance = 1e-5)
})
