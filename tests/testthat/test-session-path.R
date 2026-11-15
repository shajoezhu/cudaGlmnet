# P1: cuda_glmnet / cv.cuda_glmnet now fit their lambda paths through a single
# rcupp session. These tests lock down the anchor: with the default cold start
# the results are IDENTICAL to the original one-shot path, warm starts are a
# relaxed-tolerance opt-in, and object shapes never change.

d <- make_lasso_data(n = 60L, p = 5L, seed = 2L)

test_that("cuda_glmnet default (cold session) is bit-identical to one-shot", {
  lam <- 0.05
  fit <- cuda_glmnet(d$X, d$y, lambda = lam)
  prep <- cudaGlmnet:::.cg_prep(d$X, d$y, TRUE, TRUE)
  # One-shot reference solve (no persistent session).
  one <- cudaGlmnet:::.cg_fit_one(prep, lam, 1000L, 1e-6)

  # cold session fit == the original one-shot solve, element by element
  expect_equal(as.numeric(fit$beta[, 1L]), one$beta, tolerance = 1e-12)
  expect_equal(fit$a0[1L], one$a0, tolerance = 1e-12)
  expect_equal(fit$iterations[1L], one$iterations)
  expect_equal(fit$backend, one$backend)
})

test_that("warm_start is an opt-in that converges to the same path", {
  path <- c(0.2, 0.1, 0.05, 0.02, 0.01)
  cold <- cuda_glmnet(d$X, d$y, lambda = path, warm_start = FALSE)
  warm <- cuda_glmnet(d$X, d$y, lambda = path, warm_start = TRUE)

  expect_equal(dim(warm$beta), dim(cold$beta))
  expect_equal(warm$lambda, cold$lambda)
  expect_equal(warm$beta, cold$beta, tolerance = 1e-3)       # relaxed
  expect_equal(warm$dev.ratio, cold$dev.ratio, tolerance = 1e-3)
  # warm starts should not need MORE total iterations than cold
  expect_lte(sum(warm$iterations), sum(cold$iterations) + length(path) * 2L)
})

test_that("warm_start input is validated", {
  expect_error(cuda_glmnet(d$X, d$y, warm_start = NA), "TRUE or FALSE")
  expect_error(cuda_glmnet(d$X, d$y, warm_start = "yes"), "TRUE or FALSE")
  expect_error(cv.cuda_glmnet(d$X, d$y, nlambda = 4, nfolds = 3,
                              warm_start = 1), "TRUE or FALSE")
})

test_that("cv.cuda_glmnet cold session equals a manual one-shot CV", {
  set.seed(5)
  path <- c(0.2, 0.1, 0.05, 0.02)
  n <- nrow(d$X)
  L <- length(path)
  foldid <- sample(rep_len(1:3, n))
  cv <- cv.cuda_glmnet(d$X, d$y, lambda = path, foldid = foldid,
                       warm_start = FALSE)

  pred <- matrix(NA_real_, n, L)
  for (k in 1:3) {
    te <- which(foldid == k)
    tr <- setdiff(seq_len(n), te)
    pk <- cudaGlmnet:::.cg_prep(d$X[tr, , drop = FALSE], d$y[tr], TRUE, TRUE)
    for (j in seq_len(L)) {
      f <- cudaGlmnet:::.cg_fit_one(pk, path[j], 1000L, 1e-6)  # one-shot
      pred[te, j] <- f$a0 + drop(d$X[te, , drop = FALSE] %*% f$beta)
    }
  }
  cvm_oneshot <- colMeans((pred - d$y)^2)
  expect_equal(cv$cvm, cvm_oneshot, tolerance = 1e-10)
  expect_equal(cv$lambda.min, path[which.min(cvm_oneshot)])
})

test_that("single-column design keeps its shape on the session path", {
  X1 <- matrix(rnorm(40), 40, 1)
  y1 <- drop(X1) * 2 + rnorm(40, sd = 0.5)
  fit <- cuda_glmnet(X1, y1, nlambda = 4)
  expect_equal(dim(fit$beta), c(1L, 4L))
  expect_equal(dim(coef(fit, s = fit$lambda[2])), c(2L, 1L))
  expect_equal(dim(predict(fit, X1[1:3, , drop = FALSE])), c(3L, 4L))
  expect_length(drop(predict(fit, X1[1:3, , drop = FALSE],
                             s = fit$lambda[2])), 3L)
})
