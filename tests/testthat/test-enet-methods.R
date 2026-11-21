d <- make_lasso_data(n = 60L, p = 5L, seed = 2L)

test_that("elastic-net objects keep glmnet-like shapes and methods", {
  fit <- cuda_glmnet(d$X, d$y, nlambda = 6, alpha = 0.5)
  expect_s3_class(fit, "cuda_glmnet")
  expect_equal(dim(fit$beta), c(5L, 6L))
  expect_length(fit$lambda, 6L)
  expect_true(all(diff(fit$lambda) < 0))
  expect_length(fit$iterations, 6L)
  expect_true(all(fit$converged %in% c(0L, 1L)))
  C <- coef(fit, s = fit$lambda[3L])
  expect_equal(dim(C), c(6L, 1L))
  expect_equal(drop(predict(fit, d$X[1:4, ], s = fit$lambda[3L])),
               C[1L, 1L] + drop(d$X[1:4, ] %*% C[-1L, 1L]),
               tolerance = 1e-10)
})

test_that("standardisation and intercept still map back correctly", {
  lam <- 0.05
  fit <- cuda_glmnet(d$X, d$y, lambda = lam, alpha = 0.5)
  prep <- cudaGlmnet:::.cg_prep(d$X, d$y, TRUE, TRUE)
  conv <- cudaGlmnet:::.cg_glmnet_to_cuda(prep, 0.5)
  ref <- enet_fista_cpu(prep$Xs, prep$ys, lam * conv$lambda_scale,
                        alpha = conv$alpha_c)
  expect_equal(as.numeric(fit$beta), ref$coefficients / prep$xs,
               tolerance = 1e-8)
  expect_equal(fit$a0, prep$ym - sum(prep$xm * ref$coefficients / prep$xs),
               tolerance = 1e-8)
})

test_that("warm_start converges to the same elastic-net path", {
  path <- c(0.2, 0.1, 0.05, 0.02)
  cold <- cuda_glmnet(d$X, d$y, lambda = path, alpha = 0.5,
                      warm_start = FALSE)
  warm <- cuda_glmnet(d$X, d$y, lambda = path, alpha = 0.5,
                      warm_start = TRUE)
  expect_equal(warm$beta, cold$beta, tolerance = 1e-3)
  expect_equal(warm$lambda, cold$lambda)
})
