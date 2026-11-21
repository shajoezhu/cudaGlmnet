# The numerical contract for alpha < 1 is glmnet compatibility: a given
# (lambda, alpha) must reproduce glmnet::glmnet(lambda, alpha). The remap is
# exact; with tight solver convergence the coefficients agree to high accuracy.
test_that("cuda_glmnet matches glmnet at glmnet-convention (lambda, alpha)", {
  skip_if_not_installed("glmnet")
  d <- make_lasso_data(n = 120L, p = 10L, seed = 4L)
  lam <- 0.03
  for (a in c(0, 0.25, 0.5, 0.75, 1)) {
    fit <- cuda_glmnet(d$X, d$y, lambda = lam, alpha = a,
                       max_iter = 100000L, tol = 1e-12)
    g <- glmnet::glmnet(d$X, d$y, lambda = lam, alpha = a,
                        control = list(thresh = 1e-13))
    d_beta <- max(abs(drop(fit$beta) - drop(as.matrix(g$beta))))
    expect_true(d_beta < 1e-5,
                info = sprintf("alpha = %.2f coefficients: %g", a, d_beta))
    d_a0 <- max(abs(fit$a0 - as.numeric(coef(g)[1L])))
    expect_true(d_a0 < 1e-4,
                info = sprintf("alpha = %.2f intercept: %g", a, d_a0))
  }
})

test_that("glmnet parity holds without standardisation / intercept too", {
  skip_if_not_installed("glmnet")
  d <- make_lasso_data(n = 120L, p = 10L, seed = 7L)
  lam <- 0.03
  for (a in c(0, 0.5, 1)) {
    fit <- cuda_glmnet(d$X, d$y, lambda = lam, alpha = a,
                       standardize = FALSE, intercept = FALSE,
                       max_iter = 100000L, tol = 1e-12)
    g <- glmnet::glmnet(d$X, d$y, lambda = lam, alpha = a,
                        standardize = FALSE, intercept = FALSE,
                        control = list(thresh = 1e-13))
    d_beta <- max(abs(drop(fit$beta) - drop(as.matrix(g$beta))))
    expect_true(d_beta < 1e-5, info = sprintf("alpha = %.2f: %g", a, d_beta))
  }
})
