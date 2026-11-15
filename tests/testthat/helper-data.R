make_lasso_data <- function(n = 120L, p = 12L, seed = 42L) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  beta <- c(2.5, -1.5, rep(0, p - 2L))
  y <- drop(X %*% beta) + rnorm(n, sd = 0.4)
  list(X = X, y = y)
}
