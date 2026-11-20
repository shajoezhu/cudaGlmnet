make_subset_data <- function() {
  set.seed(7)
  X <- matrix(rnorm(80 * 5), 80, 5)
  y <- drop(X %*% c(1, -1, 0.5, 0, 0)) + rnorm(80, sd = 0.5)
  list(X = X, y = y)
}

subset_cf <- function(X, y, idx, lambda) {
  Xs <- X[idx, , drop = FALSE]
  ys <- y[idx]
  m <- length(idx)
  drop(solve(crossprod(Xs) + m * lambda * diag(ncol(Xs)),
             crossprod(Xs, ys)))
}

test_that("resident and cold agree with each other and the closed form", {
  d <- make_subset_data()
  idx <- list(1:40, 21:60, 41:80, 11:50)
  lambda <- 0.2
  res <- cuda_resident_subsets(d$X, d$y, idx, lambda, "resident")
  cold <- cuda_resident_subsets(d$X, d$y, idx, lambda, "cold")
  expect_equal(res$beta, cold$beta, tolerance = 1e-10)
  expect_true(all(res$info == 0L))
  expect_equal(nrow(res$beta), 5L)
  expect_equal(ncol(res$beta), length(idx))
  for (b in seq_along(idx)) {
    expect_equal(res$beta[, b], subset_cf(d$X, d$y, idx[[b]], lambda),
                 tolerance = 1e-8)
  }
})

test_that("cuda_resident_subsets validates inputs", {
  d <- make_subset_data()
  expect_error(cuda_resident_subsets(d$X, d$y, list(1:10, 1:9), 0.1),
               "same positive size")
  expect_error(cuda_resident_subsets(d$X, d$y, list(c(1, 2, 1000)), 0.1),
               "integers in")
  expect_error(cuda_resident_subsets(d$X, d$y, list(1:10), -1),
               "non-negative")
  expect_error(cuda_resident_subsets(d$X, d$y, list(1:10), 0.1, "bogus"),
               "arg")
})
