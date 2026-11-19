d <- make_lasso_data(n = 60L, p = 5L, seed = 2L)
prep <- cudaGlmnet:::.cg_prep(d$X, d$y, TRUE, TRUE)

test_that("compiled elastic-net path matches the pure-R oracle", {
  lam <- 0.05
  for (a in c(0.75, 0.5, 0.25)) {
    fit <- cuda_glmnet(d$X, d$y, lambda = lam, alpha = a)
    ref <- enet_fista_cpu(prep$Xs, prep$ys, lam, alpha = a)
    expect_equal(as.numeric(fit$beta[, 1L]),
                 ref$coefficients / prep$xs, tolerance = 1e-8)
    expect_equal(fit$a0[1L],
                 prep$ym - sum(prep$xm * ref$coefficients / prep$xs),
                 tolerance = 1e-8)
  }
})

test_that("elastic-net lambda path matches the oracle at every lambda", {
  lam <- c(0.2, 0.1, 0.05)
  a <- 0.5
  fit <- cuda_glmnet(d$X, d$y, lambda = lam, alpha = a)
  for (j in seq_along(lam)) {
    ref <- enet_fista_cpu(prep$Xs, prep$ys, lam[j], alpha = a)
    expect_equal(as.numeric(fit$beta[, j]),
                 ref$coefficients / prep$xs, tolerance = 1e-8)
  }
})

test_that("an explicit lambda is used unchanged for alpha < 1", {
  lam <- c(0.3, 0.2, 0.1)
  fit <- cuda_glmnet(d$X, d$y, lambda = lam, alpha = 0.3)
  expect_equal(fit$lambda, lam)
})
