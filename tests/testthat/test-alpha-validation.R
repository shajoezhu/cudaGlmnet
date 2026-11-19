d <- make_lasso_data(n = 60L, p = 5L, seed = 2L)

test_that("cuda_glmnet validates alpha", {
  expect_error(cuda_glmnet(d$X, d$y, alpha = -0.1), "alpha")
  expect_error(cuda_glmnet(d$X, d$y, alpha = 1.1), "alpha")
  expect_error(cuda_glmnet(d$X, d$y, alpha = NA_real_), "alpha")
  expect_error(cuda_glmnet(d$X, d$y, alpha = c(0.5, 1)), "alpha")
  expect_error(cuda_glmnet(d$X, d$y, alpha = "0.5"), "alpha")
  expect_error(cuda_glmnet(d$X, d$y, alpha = Inf), "alpha")
})

test_that("cv.cuda_glmnet validates alpha", {
  expect_error(cv.cuda_glmnet(d$X, d$y, nfolds = 3, alpha = -0.1), "alpha")
  expect_error(cv.cuda_glmnet(d$X, d$y, nfolds = 3, alpha = 2), "alpha")
  expect_error(cv.cuda_glmnet(d$X, d$y, nfolds = 3, alpha = NA), "alpha")
})

test_that("enet_fista_cpu validates alpha", {
  expect_error(enet_fista_cpu(d$X, d$y, 0.05, alpha = -0.1), "alpha")
  expect_error(enet_fista_cpu(d$X, d$y, 0.05, alpha = 1.5), "alpha")
})
