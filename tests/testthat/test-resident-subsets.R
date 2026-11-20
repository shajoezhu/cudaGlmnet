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
  expect_error(cuda_resident_subsets(d$X, d$y, list(1:10), 0.1,
                                     method = "bogus"),
               "arg")
})

# Push-through / Woodbury route: for p > m solve the smaller m x m system
# Xs^T (Xs Xs^T + m*lambda I)^-1 ys instead of the p x p Gram. Both routes,
# both strategies, and the exact closed form must agree.
test_that("dual and primal subset routes agree (p > m)", {
  set.seed(11)
  n <- 100L
  p <- 200L
  m <- 40L
  lambda <- 0.25
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)
  idx <- replicate(5L, sample.int(n, m), simplify = FALSE)
  exact <- vapply(idx, function(ii) subset_cf(X, y, ii, lambda), numeric(p))

  auto <- cuda_resident_subsets(X, y, idx, lambda, "resident", "auto")
  dual <- cuda_resident_subsets(X, y, idx, lambda, "resident", "dual")
  primal <- cuda_resident_subsets(X, y, idx, lambda, "resident", "primal")
  cold <- cuda_resident_subsets(X, y, idx, lambda, "cold", "dual")

  # auto resolves to dual (smaller system) and reports the Cholesky dimension.
  expect_equal(auto$method, "dual")
  expect_equal(dual$method, "dual")
  expect_equal(primal$method, "primal")
  expect_equal(auto$sys_dim, m)
  expect_equal(primal$sys_dim, p)
  expect_true(all(auto$info == 0L))
  expect_true(all(dual$info == 0L))
  expect_true(all(primal$info == 0L))

  expect_equal(dual$beta, auto$beta, tolerance = 1e-12)
  expect_equal(dual$beta, cold$beta, tolerance = 1e-10)
  expect_equal(dual$beta, primal$beta, tolerance = 1e-7)
  expect_equal(dual$beta, exact, tolerance = 1e-6)
})

test_that("explicit dual is correct even when p <= m", {
  d <- make_subset_data()
  idx <- list(1:40, 21:60, 41:80, 11:50)
  lambda <- 0.2
  auto <- cuda_resident_subsets(d$X, d$y, idx, lambda)
  dual <- cuda_resident_subsets(d$X, d$y, idx, lambda, "resident", "dual")
  expect_equal(auto$method, "primal")
  expect_equal(auto$sys_dim, ncol(d$X))
  expect_equal(dual$method, "dual")
  expect_equal(dual$sys_dim, 40L)
  expect_equal(dual$beta, auto$beta, tolerance = 1e-8)
})
