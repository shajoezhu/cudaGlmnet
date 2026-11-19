d <- make_lasso_data(n = 60L, p = 5L, seed = 2L)

test_that("cv.cuda_glmnet works with alpha < 1", {
  set.seed(11)
  cv <- cv.cuda_glmnet(d$X, d$y, nlambda = 6, nfolds = 3, alpha = 0.5)
  expect_s3_class(cv, "cv.cuda_glmnet")
  expect_length(cv$cvm, 6L)
  expect_equal(cv$lambda.min, cv$lambda[which.min(cv$cvm)])
  expect_true(cv$lambda.1se >= cv$lambda.min)
  expect_s3_class(cv$fit, "cuda_glmnet")
  expect_equal(dim(coef(cv, s = "lambda.min")), c(6L, 1L))
  expect_equal(dim(coef(cv, s = "lambda.1se")), c(6L, 1L))
  expect_length(drop(predict(cv, d$X[1:3, ])), 3L)
})

test_that("cv respects user lambda and foldid with alpha < 1", {
  lam <- c(0.2, 0.1, 0.05)
  cv <- cv.cuda_glmnet(d$X, d$y, lambda = lam, foldid = rep(1:2, 30L),
                       alpha = 0.25)
  expect_equal(cv$lambda, lam)
  expect_equal(cv$nfolds, 2L)
  expect_length(cv$cvm, 3L)
})
