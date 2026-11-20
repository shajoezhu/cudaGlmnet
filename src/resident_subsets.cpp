#include <Rcpp.h>
#include <rcupp.h>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

/*
 * cudaGlmnet Stage 2B: a DOWNSTREAM composition that keeps ONE full panel
 * resident on the device and refits many row subsets WITHOUT re-uploading.
 *
 * It composes rcupp's B-stage generic parts (gather_rows / syrk / potrf /
 * potrs) plus Phase-0 resident operators through <rcupp.h> (LinkingTo: rcupp
 * -> -I.../include). NO CUDA code, NO configure / Makevars: identical numerics
 * on the CUDA backend and the CPU fallback. rcupp still knows nothing about
 * ridge, CV or reference panels -- the solver loop lives here, downstream
 * (Rcpp glue plus rcupp substrate).
 *
 * This is a "parts demo", not a production p>>n ridge solver: the p x p Gram
 * matrix and its Cholesky are demo-scale (a production path would use the
 * dual/Woodbury form). The point is DATA RESIDENCY:
 *   - strategy "resident": upload the whole (n x p) panel once, then select
 *     each subset with a device-side gather (device -> device, no H->D copy);
 *   - strategy "cold":     assemble each (m x p) subset on the host and
 *     re-upload it on every refit (B full H->D copies -- the old set_data cost).
 * Both strategies share ONE rcupp_ctx, so the comparison isolates the
 * host<->device transfer, not context/handle setup.
 *
 * Closed form per subset (all subsets share size m):
 *   (Xs^T Xs + m*lambda I) beta = Xs^T ys
 * The diagonal m*lambda is constant across subsets, so it is uploaded once
 * into C0 and reused via d_copy on every refit. syrk writes only the LOWER
 * triangle; potrf/potrs MUST use the SAME uplo (RCUPP_FILL_LOWER).
 *
 * idx_list elements are ZERO-BASED row indices (the rcupp C ABI convention);
 * the R wrapper cuda_resident_subsets() converts R's 1-based indices.
 */

namespace {

struct CtxGuard {
  rcupp_ctx c;
  CtxGuard() : c(rcupp_ctx_create()) {}
  ~CtxGuard() { if (c) rcupp_ctx_destroy(c); }
  CtxGuard(const CtxGuard&) = delete;
  CtxGuard& operator=(const CtxGuard&) = delete;
};

// Buffers are owned by the ctx; rcupp_ctx_destroy frees them all. Guard the
// one genuinely fallible creation path.
inline rcupp_buf need_buf(rcupp_ctx c, int n) {
  rcupp_buf b = rcupp_buf_alloc(c, n);
  if (!b) Rcpp::stop("rcupp: buffer allocation failed");
  return b;
}

// Solve one subset from resident buffers Xs (m x p), ys (m). C0 = m*lambda*I
// (uploaded once) -> C; syrk accumulates the Gram into the lower triangle;
// potrf/potrs solve from the Cholesky factor; beta[p] is downloaded last.
// Returns 0 on success, the LAPACK info (>0, not SPD) or a small error code.
inline int solve_subset(rcupp_ctx c, int m, int p,
                        rcupp_buf Xs, rcupp_buf ys,
                        rcupp_buf C0, rcupp_buf C,
                        rcupp_buf Xty, rcupp_buf B,
                        double* beta_out) {
  rcupp_d_gemv_t(c, Xs, m, p, ys, Xty);        // Xty = Xs^T ys
  rcupp_d_copy(c, p * p, C0, C);               // C = m*lambda*I
  rcupp_d_copy(c, p, Xty, B);                  // B = Xty
  // BLAS alpha/beta here are scalars (1,1), NOT the elastic-net mixing alpha.
  rcupp_d_syrk(c, Xs, m, p, RCUPP_FILL_LOWER, 1.0, 1.0, C);  // += Xs^T Xs
  int info = 0;
  if (rcupp_d_potrf(c, RCUPP_FILL_LOWER, p, C, p, &info) != 0) return 2;
  if (info != 0) return info;                  // not positive definite
  if (rcupp_d_potrs(c, RCUPP_FILL_LOWER, p, 1, C, p, B, p, &info) != 0)
    return 3;
  rcupp_buf_download(c, B, beta_out, p);       // 1 length-p transfer per subset
  return info;                                 // 0 = ok
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List resident_subsets_cpp(Rcpp::NumericMatrix x,
                                Rcpp::NumericVector y,
                                Rcpp::List idx_list,
                                double lambda,
                                std::string strategy) {
  if (rcupp_ctx_abi() != 1) {
    Rcpp::stop("rcupp >= 0.1.0 (device-context ABI 1) is required");
  }
  const int n = x.nrow();
  const int p = x.ncol();
  const int B = static_cast<int>(idx_list.size());
  const bool resident = (strategy == "resident");
  if (!resident && strategy != "cold") {
    Rcpp::stop("'strategy' must be \"resident\" or \"cold\"");
  }
  if (n < 1 || p < 1 || B < 1 ||
      y.size() != static_cast<R_xlen_t>(n) ||
      !std::isfinite(lambda) || lambda < 0.0) {
    Rcpp::stop("invalid resident_subsets_cpp parameters");
  }

  // All subsets share one size m (so Xs/C0 are allocated once). Validate the
  // zero-based indices are in [0, n) before any device or host access.
  int m = -1;
  std::vector<std::vector<int> > idx0(B);
  for (int b = 0; b < B; ++b) {
    Rcpp::IntegerVector iv = idx_list[b];
    if (m < 0) m = static_cast<int>(iv.size());
    if (m < 1 || static_cast<int>(iv.size()) != m) {
      Rcpp::stop("all subsets in 'idx_list' must have the same positive size");
    }
    idx0[b].resize(m);
    for (int i = 0; i < m; ++i) {
      const int v = iv[i];
      if (v < 0 || v >= n) Rcpp::stop("subset index out of range [0, n)");
      idx0[b][i] = v;
    }
  }
  const double diag = static_cast<double>(m) * lambda;

  CtxGuard g;
  if (!g.c) Rcpp::stop("rcupp: device context creation failed");
  rcupp_ctx c = g.c;

  // Working buffers (owned by ctx; freed at rcupp_ctx_destroy).
  rcupp_buf Xs  = need_buf(c, m * p);
  rcupp_buf ys  = need_buf(c, m);
  rcupp_buf C0  = need_buf(c, p * p);
  rcupp_buf C   = need_buf(c, p * p);
  rcupp_buf Xty = need_buf(c, p);
  rcupp_buf Buf = need_buf(c, p);

  // C0 = m*lambda*I (constant across subsets), uploaded once.
  std::vector<double> C0h(static_cast<std::size_t>(p) * p, 0.0);
  for (int i = 0; i < p; ++i)
    C0h[i + static_cast<std::size_t>(i) * p] = diag;
  rcupp_buf_upload(c, C0, C0h.data(), p * p);

  // Resident strategy: upload the FULL panel once, keep it for all subsets.
  rcupp_buf Xfull = NULL, yfull = NULL;
  if (resident) {
    Xfull = need_buf(c, n * p);
    yfull = need_buf(c, n);
    rcupp_buf_upload(c, Xfull, x.begin(), n * p);
    rcupp_buf_upload(c, yfull, y.begin(), n);
  }

  Rcpp::NumericMatrix beta(p, B);
  Rcpp::IntegerVector info(B);

  for (int b = 0; b < B; ++b) {
    const int* ix = idx0[b].data();
    if (resident) {
      // Device-side row selection: no host<->device round trip.
      rcupp_d_gather_rows(c, Xfull, n, p, ix, m, Xs);
      rcupp_d_gather_rows(c, yfull, n, 1, ix, m, ys);
    } else {
      // Cold: assemble the contiguous subset on the host, re-upload it.
      std::vector<double> Xsub(static_cast<std::size_t>(m) * p);
      std::vector<double> ysub(m);
      for (int j = 0; j < p; ++j) {
        for (int i = 0; i < m; ++i) {
          Xsub[i + static_cast<std::size_t>(j) * m] =
              x[ix[i] + static_cast<std::size_t>(j) * n];
        }
      }
      for (int i = 0; i < m; ++i) ysub[i] = y[ix[i]];
      rcupp_buf_upload(c, Xs, Xsub.data(), m * p);
      rcupp_buf_upload(c, ys, ysub.data(), m);
    }
    info[b] = solve_subset(c, m, p, Xs, ys, C0, C, Xty, Buf,
                           beta.begin() + static_cast<R_xlen_t>(b) * p);
  }

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta,
      Rcpp::Named("info") = info,
      Rcpp::Named("strategy") = strategy,
      Rcpp::Named("m") = m,
      Rcpp::Named("diag") = diag,
      Rcpp::Named("cuda") = static_cast<bool>(rcupp_cuda_available()));
}
