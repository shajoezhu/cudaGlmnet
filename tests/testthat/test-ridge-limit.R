d <- make_lasso_data(n = 60L, p = 5L, seed = 2L)
prep <- cudaGlmnet:::.cg_prep(d$X, d$y, TRUE, TRUE)

test_that("alpha = 0 gives non-zero ridge coefficients", {
  fit <- cuda_glmnet(d$X, d$y, lambda = 0.001, alpha = 0)
  expect_equal(dim(fit$beta), c(5L, 1L))
  expect_gt(sum(abs(fit$beta[, 1L]) > 1e-10), 3L)
})

test_that("pure-ridge fit reproduces the ridge oracle", {
  lam <- 0.05
  fit <- cuda_glmnet(d$X, d$y, lambda = lam, alpha = 0)
  ref <- enet_fista_cpu(prep$Xs, prep$ys, lam, alpha = 0)
  expect_equal(as.numeric(fit$beta[, 1L]),
               ref$coefficients / prep$xs, tolerance = 1e-8)
})

test_that("elastic-net objective decreases as lambda shrinks", {
  lam <- c(0.2, 0.05, 0.01)
  obj <- vapply(lam, function(l) {
    enet_fista_cpu(prep$Xs, prep$ys, l, alpha = 0.5)$objective
  }, numeric(1))
  expect_true(all(diff(obj) <= 1e-8))
})
