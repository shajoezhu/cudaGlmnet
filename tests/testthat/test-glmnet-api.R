d <- make_lasso_data(n = 60L, p = 5L, seed = 2L)

test_that("cuda_glmnet returns a glmnet-like path object", {
  fit <- cuda_glmnet(d$X, d$y, nlambda = 8)
  expect_s3_class(fit, "cuda_glmnet")
  expect_equal(dim(fit$beta), c(5L, 8L))
  expect_length(fit$lambda, 8L)
  expect_true(all(diff(fit$lambda) < 0))       # decreasing path
  expect_length(fit$a0, 8L)
  expect_equal(fit$nobs, 60L)
  expect_equal(fit$nvars, 5L)
  expect_true(all(fit$dev.ratio >= 0))
  expect_length(fit$nzero, 8L)
  expect_output(print(fit), "cuda_glmnet")
})

test_that("standardisation / intercept match the oracle on transformed data", {
  lam <- 0.05
  fit <- cuda_glmnet(d$X, d$y, lambda = lam)
  xm <- colMeans(d$X)
  ym <- mean(d$y)
  Xc <- d$X - matrix(xm, nrow(d$X), ncol(d$X), byrow = TRUE)
  xs <- sqrt(colMeans(Xc^2))
  Xs <- Xc / matrix(xs, nrow(d$X), ncol(d$X), byrow = TRUE)
  ref <- lasso_fista_cpu(Xs, d$y - ym, lambda = lam)
  expect_equal(as.numeric(fit$beta), ref$coefficients / xs, tolerance = 1e-8)
  expect_equal(fit$a0, ym - sum(xm * ref$coefficients / xs), tolerance = 1e-8)
})

test_that("coef(s=) / predict(s=) agree with manual computation", {
  fit <- cuda_glmnet(d$X, d$y, nlambda = 6)
  s <- fit$lambda[3]
  C <- coef(fit, s = s)
  expect_equal(dim(C), c(6L, 1L))
  expect_equal(rownames(C)[1L], "(Intercept)")
  pred <- predict(fit, d$X[1:4, ], s = s)
  manual <- C[1L, 1L] + drop(d$X[1:4, ] %*% C[-1L, 1L])
  expect_equal(drop(pred), manual, tolerance = 1e-10)
})

test_that("coef(s=NULL) returns the full (p+1) x L matrix", {
  fit <- cuda_glmnet(d$X, d$y, nlambda = 5)
  expect_equal(dim(coef(fit)), c(6L, 5L))
})

test_that("a single-column design keeps beta a matrix", {
  X1 <- matrix(rnorm(40), 40, 1)
  y1 <- drop(X1) * 2 + rnorm(40, sd = 0.5)
  fit <- cuda_glmnet(X1, y1, nlambda = 4)
  expect_equal(dim(fit$beta), c(1L, 4L))
  expect_equal(dim(coef(fit, s = fit$lambda[2])), c(2L, 1L))
  # s = NULL returns predictions at all lambdas (glmnet behaviour): 3 x 4.
  expect_equal(dim(predict(fit, X1[1:3, , drop = FALSE])), c(3L, 4L))
  # A single s collapses to one prediction per row even with p = 1.
  expect_length(drop(predict(fit, X1[1:3, , drop = FALSE],
                             s = fit$lambda[2])), 3L)
})

test_that("cv.cuda_glmnet selects lambda.min / lambda.1se", {
  set.seed(11)
  cv <- cv.cuda_glmnet(d$X, d$y, nlambda = 6, nfolds = 3)
  expect_s3_class(cv, "cv.cuda_glmnet")
  expect_length(cv$cvm, 6L)
  expect_length(cv$cvsd, 6L)
  expect_equal(cv$lambda.min, cv$lambda[which.min(cv$cvm)])
  expect_true(cv$lambda.1se >= cv$lambda.min)
  expect_true(cv$lambda.1se <= max(cv$lambda))
  expect_s3_class(cv$fit, "cuda_glmnet")
  expect_equal(dim(coef(cv, s = "lambda.min")), c(6L, 1L))
  expect_equal(dim(coef(cv, s = "lambda.1se")), c(6L, 1L))
  expect_length(drop(predict(cv, d$X[1:3, ])), 3L)
  expect_output(print(cv), "lambda.min")
})

test_that("user-supplied foldid and lambda are respected", {
  foldid <- rep(1:2, length.out = 60L)
  lam <- c(0.2, 0.1, 0.05)
  cv <- cv.cuda_glmnet(d$X, d$y, lambda = lam, foldid = foldid)
  expect_equal(cv$lambda, lam)
  expect_equal(cv$nfolds, 2L)
  expect_length(cv$cvm, 3L)
})

test_that("glmnet API validates inputs", {
  expect_error(cuda_glmnet(as.data.frame(d$X), d$y), "numeric matrix")
  expect_error(cuda_glmnet(d$X, d$y, nlambda = 0), "positive integer")
  expect_error(cuda_glmnet(d$X, d$y, lambda = -1), "non-negative")
  expect_error(cuda_glmnet(d$X, d$y, standardize = NA), "TRUE or FALSE")
  expect_error(cuda_glmnet(d$X, d$y, lambda.min.ratio = 1.5),
               "lambda.min.ratio")
  expect_error(coef(cuda_glmnet(d$X, d$y, nlambda = 3), s = "lambda.min"),
               "only valid on a cv")
  expect_error(predict(cuda_glmnet(d$X, d$y, nlambda = 3)), "required")
  expect_error(cv.cuda_glmnet(d$X, d$y, nfolds = 1), "between 2")
  expect_error(cv.cuda_glmnet(d$X, d$y, foldid = rep(1, 60)),
               "at least two distinct folds")
})
