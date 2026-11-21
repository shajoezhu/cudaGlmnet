# Part B tests: persistent ctx subset-ridge fit (ctx XPtr passed across .Call)
# and the mlr3 regr.cuda_subset_ridge learner. Backend-agnostic numerics.

make_ctx_data <- function(n = 60L, p = 40L, m = 20L, seed = 3L) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  b <- c(2, -1.5, rep(0, p - 2L))
  y <- drop(X %*% b) + rnorm(n, sd = 0.5)
  idx <- lapply(1:4, function(b) sample.int(n, m))
  list(X = X, y = y, idx = idx)
}

test_that("ctx-based subset ridge matches cuda_resident_subsets", {
  skip_if_not_installed("rcupp")
  d <- make_ctx_data()
  ctx <- rcupp::ctx_create()
  on.exit(rcupp::ctx_destroy(ctx), add = TRUE)
  fit1 <- cudaGlmnet:::cuda_ctx_subset_ridge(ctx, d$X, d$y, d$idx, 0.2,
                                             "resident", "dual")
  b <- cuda_resident_subsets(d$X, d$y, d$idx, 0.2, "resident", "dual")
  expect_identical(fit1$method, "dual")
  expect_equal(fit1$beta, b$beta, tolerance = 1e-12)
  expect_true(all(fit1$info == 0L))
})

test_that("one persistent ctx reused across calls is bit-identical", {
  d <- make_ctx_data()
  ctx <- rcupp::ctx_create()
  on.exit(rcupp::ctx_destroy(ctx), add = TRUE)
  a1 <- cudaGlmnet:::cuda_ctx_subset_ridge(ctx, d$X, d$y, d$idx, 0.2,
                                           "resident", "dual")
  b1 <- cudaGlmnet:::cuda_ctx_subset_ridge(ctx, d$X, d$y, d$idx, 0.2,
                                           "resident", "dual")
  # Same ctx, same data -> bit-identical.
  expect_equal(a1$beta, b1$beta, tolerance = 0)
})

test_that("ctx_resident_subsets_cpp rejects a destroyed ctx", {
  d <- make_ctx_data()
  ctx <- rcupp::ctx_create()
  idx0 <- lapply(d$idx, function(i) as.integer(i) - 1L)
  rcupp::ctx_destroy(ctx)
  expect_error(
    cudaGlmnet:::ctx_resident_subsets_cpp(ctx, d$X, d$y, idx0, 0.2,
                                          "resident", "dual"),
    "destroyed"
  )
})
