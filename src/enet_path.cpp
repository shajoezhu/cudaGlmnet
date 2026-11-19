#include <Rcpp.h>
#include <rcupp.h>
#include <algorithm>
#include <cmath>

/*
 * cudaGlmnet Phase A: elastic-net / ridge solved by a DOWNSTREAM FISTA loop
 * that composes rcupp's Phase-0 RESIDENT device operators.
 *
 * This file has NO CUDA code and cudaGlmnet has NO configure / Makevars /
 * CUDA detection / CPU-fallback logic. It reaches the fixed rcupp component
 * catalog purely through <rcupp.h> (LinkingTo: rcupp -> -I.../include) and
 * R_GetCCallable(). It compiles and gives identical numerics whether rcupp
 * was built with a CUDA backend or its CPU fallback.
 *
 * The whole lambda path shares ONE rcupp_ctx: X/y are uploaded once and every
 * operator in the inner FISTA loop runs on resident buffers, with no
 * host<->device round trip per call (the gap documented for the one-shot F0
 * ABI in cudaFin/src/fin.cpp). rcupp never learns what lasso / ridge is.
 *
 * Numerical contract (docs/phaseA-cudaglmnet-ridge-plan.md section 3):
 *   objective 1/(2n)||y - Xb||^2 + lambda[(1-alpha)/2 ||b||^2 + alpha ||b||_1]
 *   L = lambda_max(X^T X)/n, step = 1/(L n), thresh = lambda/L
 *   prox(u) = soft_threshold(u, thresh*alpha) / (1 + thresh*(1-alpha))
 *   alpha = 1 -> lasso; alpha = 0 -> ridge (iterative prox step)
 *
 * The R-level wrapper decides: alpha == 1 uses the existing rcupp lasso
 * session (bit-identical), alpha < 1 calls this function. So this solver is
 * never exercised at alpha == 1 and rcupp's solver is never given `alpha`.
 */

namespace {

struct CtxGuard {
  rcupp_ctx c;
  CtxGuard() : c(rcupp_ctx_create()) {}
  ~CtxGuard() { if (c) rcupp_ctx_destroy(c); }
  CtxGuard(const CtxGuard&) = delete;
  CtxGuard& operator=(const CtxGuard&) = delete;
};

// Buffers are owned by the ctx; rcupp_ctx_destroy frees them all, so no
// per-buffer RAII is needed. Guard the one genuinely fallible creation path.
inline rcupp_buf need_buf(rcupp_ctx c, int n) {
  rcupp_buf b = rcupp_buf_alloc(c, n);
  if (!b) Rcpp::stop("rcupp: buffer allocation failed");
  return b;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List enet_path_cpp(Rcpp::NumericMatrix Xs,
                         Rcpp::NumericVector ys,
                         Rcpp::NumericVector lambda,
                         double alpha, double L,
                         bool warm_start, int max_iter, double tol) {
  if (rcupp_ctx_abi() != 1) {
    Rcpp::stop("rcupp >= 0.1.0 (device-context ABI 1) is required");
  }
  const int n = Xs.nrow();
  const int p = Xs.ncol();
  const int nl = static_cast<int>(lambda.size());
  if (n < 1 || p < 1 || nl < 1 || !(L > 0.0) || max_iter < 1 ||
      !(tol > 0.0) || !(alpha >= 0.0 && alpha <= 1.0) ||
      ys.size() != static_cast<R_xlen_t>(n)) {
    Rcpp::stop("invalid enet_path_cpp parameters");
  }

  CtxGuard g;
  if (!g.c) Rcpp::stop("rcupp: device context creation failed");
  rcupp_ctx c = g.c;

  // Resident design + response (uploaded once for the whole path).
  rcupp_buf X  = need_buf(c, n * p);
  rcupp_buf y  = need_buf(c, n);
  rcupp_buf b  = need_buf(c, p);      // beta (previous iterate)
  rcupp_buf z  = need_buf(c, p);      // FISTA extrapolated point
  rcupp_buf bn = need_buf(c, p);      // beta_new
  rcupp_buf zn = need_buf(c, p);      // z_new
  rcupp_buf u  = need_buf(c, p);
  rcupp_buf gg = need_buf(c, p);      // gradient
  rcupp_buf t1 = need_buf(c, p);      // momentum temp
  rcupp_buf t2 = need_buf(c, p);      // convergence temp
  rcupp_buf r  = need_buf(c, n);
  rcupp_buf Xb = need_buf(c, n);

  rcupp_buf_upload(c, X, Xs.begin(), n * p);
  rcupp_buf_upload(c, y, ys.begin(), n);

  const double step = 1.0 / (L * static_cast<double>(n));

  Rcpp::NumericMatrix beta(p, nl);
  Rcpp::IntegerVector iterations(nl), converged(nl);

  for (int j = 0; j < nl; ++j) {
    const double lam = lambda[j];
    const double thresh = lam / L;
    const double thr1 = thresh * alpha;
    const double inv = 1.0 / (1.0 + thresh * (1.0 - alpha));

    if (!warm_start || j == 0) {
      rcupp_buf_fill(c, b, 0.0, p);
      rcupp_buf_fill(c, z, 0.0, p);
    } else {
      rcupp_d_copy(c, p, b, z);      // seed from previous solution
    }

    double tk = 1.0;
    int it = 0;
    int conv = 0;
    for (it = 1; it <= max_iter; ++it) {
      rcupp_d_gemv(c, X, n, p, z, Xb);                 // Xb = X z
      rcupp_d_copy(c, n, y, r);
      rcupp_d_axpy(c, n, -1.0, Xb, r);                // r = y - Xb
      rcupp_d_scal(c, n, step, r);                    // r *= step
      rcupp_d_gemv_t(c, X, n, p, r, gg);              // g = X^T r
      rcupp_d_copy(c, p, z, u);
      rcupp_d_axpy(c, p, 1.0, gg, u);                 // u = z + g

      rcupp_d_soft_threshold(c, u, thr1, bn, p);      // soft-threshold part
      rcupp_d_scal(c, p, inv, bn);                    // / denom -> enet prox

      // FISTA momentum: z_new = b_new + mom * (b_new - b)
      const double tnew =
          (1.0 + std::sqrt(1.0 + 4.0 * tk * tk)) / 2.0;
      const double mom = (tk - 1.0) / tnew;
      rcupp_d_copy(c, p, bn, t1);
      rcupp_d_axpy(c, p, -1.0, b, t1);                // t1 = bn - b
      rcupp_d_scal(c, p, mom, t1);
      rcupp_d_copy(c, p, bn, zn);
      rcupp_d_axpy(c, p, 1.0, t1, zn);                // zn = bn + mom*(bn-b)

      // Convergence: ||b_new - b|| / max(1, ||b_new||) < tol
      rcupp_d_copy(c, p, bn, t2);
      rcupp_d_axpy(c, p, -1.0, b, t2);
      const double diff = rcupp_d_nrm2(c, p, t2);
      const double nb = rcupp_d_nrm2(c, p, bn);
      const double denom = std::max(1.0, nb);

      rcupp_d_copy(c, p, bn, b);
      rcupp_d_copy(c, p, zn, z);
      tk = tnew;

      if (diff / denom < tol) { conv = 1; break; }
    }

    rcupp_buf_download(c, b, beta.begin() + j * p, p);
    iterations[j] = it;
    converged[j] = conv;
  }

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta,
      Rcpp::Named("iterations") = iterations,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("cuda") = static_cast<bool>(rcupp_cuda_available()));
}
