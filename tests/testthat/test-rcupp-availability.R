test_that("rcupp is installed and reports a logical backend flag", {
  skip_if_not_installed("rcupp")
  expect_type(rcupp::cuda_available(), "logical")
})

test_that("a fit records which backend was linked", {
  skip_if_not_installed("rcupp")
  d <- make_lasso_data(n = 60L, p = 5L, seed = 13L)
  fit <- cuda_lasso(d$X, d$y, lambda = 0.05)
  expect_true(fit$backend %in% c("cuda", "cpu-shim"))
  expect_identical(fit$backend,
                   if (isTRUE(rcupp::cuda_available())) "cuda" else "cpu-shim")
  expect_true(is.finite(fit$objective))
  expect_true(fit$iterations >= 1L)
})
