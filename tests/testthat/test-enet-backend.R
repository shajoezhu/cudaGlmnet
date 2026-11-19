test_that("elastic-net backend flag matches the linked rcupp backend", {
  skip_if_not_installed("rcupp")
  d <- make_lasso_data(n = 60L, p = 5L, seed = 13L)
  fit <- cuda_glmnet(d$X, d$y, lambda = 0.05, alpha = 0.5)
  expect_identical(fit$backend,
                   if (isTRUE(rcupp::cuda_available())) "cuda" else "cpu-shim")
  expect_true(is.finite(fit$dev.ratio[1L]))
  expect_true(fit$iterations[1L] >= 1L)
})
