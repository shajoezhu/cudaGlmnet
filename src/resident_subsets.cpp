#include <Rcpp.h>
#include <rcupp.h>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

/*
 * cudaGlmnet Stage 2B / low-level capability: a DOWNSTREAM composition that
 * keeps ONE full panel resident on the device and refits many row subsets
 * WITHOUT re-uploading.
 *
 * It composes rcupp's generic parts through <rcupp.h> (LinkingTo: rcupp ->
 * -I.../include). NO CUDA code, NO configure / Makevars: identical numerics
 * on the CUDA backend and the CPU fallback. rcupp still knows nothing about
 * ridge, CV or reference panels -- the solver loop lives here, downstream
 * (Rcpp glue plus rcupp substrate).
 *
 * TWO algebraically equivalent closed forms (all subsets share size m):
 *
 *   PRIMAL (p x p, used when p <= m):
 *       (Xs^T Xs + m*lambda I_p) beta = Xs^T ys
 *     Gram via syrk, solve via potrf/potrs. This is the demo-scale path.
 *
 *   DUAL / Woodbury (m x m, used when p > m -- the production p >> n route):
 *       beta = Xs^T (Xs Xs^T + m*lambda I_m)^{-1} ys
 *     push-through identity: (X^T X + tI)^{-1} X^T = X^T (X X^T + tI)^{-1},
 *     t = m*lambda. Sample-space Gram Xs Xs^T via the C-stage gemm(N,T),
 *     solve the SMALLER m x m system via potrf/potrs, then beta = Xs^T alpha
 *     via gemv_t. When p >> m this avoids materialising a p x p Gram/Cholesky.
 *
 * method = "auto" picks dual iff p > m (smaller system); "primal" / "dual"
 * force a route. Both produce the same beta up to rounding. The data-residency
 * strategy ("resident" / "cold") is ORTHOGONAL to method: it only controls
 * how Xs/ys are obtained (device-side gather vs per-subset host re-upload).
 * Both strategies share ONE rcupp_ctx, so ROI isolates transfer, not setup.
 *
 * The m*lambda diagonal is constant across equal-size subsets, so it is
 * uploaded once into `diag` and reused via d_copy on every refit. syrk writes
 * only the LOWER triangle; gemm writes the full symmetric matrix; potrf/potrs
 * MUST use the SAME uplo (RCUPP_FILL_LOWER).
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

// Solve one subset from resident Xs (m x p), ys (m), for BOTH routes.
//   use_dual=false (primal): sys dim sd=p, diagonal p x p, Gram Xs^T Xs.
//   use_dual=true  (dual):   sys dim sd=m, diagonal m x m, Gram Xs Xs^T.
// `diag` = m*lambda*I_sd (uploaded once) -> `work`; Gram accumulated into the
// lower triangle; potrf/potrs solve from the Cholesky factor; beta[p] is
// downloaded last. Returns 0 on success, LAPACK info (>0, not SPD) or a small
// error code.
inline int solve_subset(rcupp_ctx c, int m, int p, bool use_dual,
                        rcupp_buf Xs, rcupp_buf ys,
                        rcupp_buf diag, rcupp_buf work, rcupp_buf rhs,
                        rcupp_buf out, double* beta_out) {
  const int sd = use_dual ? m : p;
  rcupp_d_copy(c, sd * sd, diag, work);            // work = m*lambda*I_sd
  if (use_dual) {
    // work(m x m) += Xs Xs^T : A=Xs (m x p, lda=m, OP_N), B=Xs (m x p
    // stored, ldb=m, OP_T -> p x m). beta=1 accumulates into work.
    // BLAS alpha/beta here are scalars (1,1), NOT the elastic-net mixing alpha.
    rcupp_d_gemm(c, RCUPP_OP_N, RCUPP_OP_T, m, m, p,
                 1.0, Xs, m, Xs, m, 1.0, work, m);
    rcupp_d_copy(c, m, ys, rhs);                    // RHS = ys (m)
  } else {
    rcupp_d_syrk(c, Xs, m, p, RCUPP_FILL_LOWER, 1.0, 1.0, work);  // += Xs^T Xs
    rcupp_d_gemv_t(c, Xs, m, p, ys, rhs);          // RHS = Xs^T ys (p)
  }
  int info = 0;
  if (rcupp_d_potrf(c, RCUPP_FILL_LOWER, sd, work, sd, &info) != 0) return 2;
  if (info != 0) return info;                      // not positive definite
  if (rcupp_d_potrs(c, RCUPP_FILL_LOWER, sd, 1, work, sd, rhs, sd, &info) != 0)
    return 3;
  if (use_dual) {
    rcupp_d_gemv_t(c, Xs, m, p, rhs, out);         // beta = Xs^T alpha (p)
    rcupp_buf_download(c, out, beta_out, p);
  } else {
    rcupp_buf_download(c, rhs, beta_out, p);       // rhs (p) IS beta
  }
  return info;                                     // 0 = ok
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List resident_subsets_cpp(Rcpp::NumericMatrix x,
                                Rcpp::NumericVector y,
                                Rcpp::List idx_list,
                                double lambda,
                                std::string strategy,
                                std::string method) {
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
  if (method != "auto" && method != "primal" && method != "dual") {
    Rcpp::stop("'method' must be \"auto\", \"primal\" or \"dual\"");
  }
  if (n < 1 || p < 1 || B < 1 ||
      y.size() != static_cast<R_xlen_t>(n) ||
      !std::isfinite(lambda) || lambda < 0.0) {
    Rcpp::stop("invalid resident_subsets_cpp parameters");
  }

  // All subsets share one size m (so Xs/diag/work are allocated once). Validate
  // the zero-based indices are in [0, n) before any device or host access.
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
  const double diag_val = static_cast<double>(m) * lambda;

  // Resolve the method: "auto" picks the smaller Cholesky system (dual iff
  // p > m, the production p >> n shape); explicit values force a route.
  const bool use_dual = (method == "dual") || (method == "auto" && p > m);
  const int sd = use_dual ? m : p;                  // Cholesky system dimension
  const std::string method_used = use_dual ? "dual" : "primal";

  CtxGuard g;
  if (!g.c) Rcpp::stop("rcupp: device context creation failed");
  rcupp_ctx c = g.c;

  // Working buffers (owned by ctx; freed at rcupp_ctx_destroy). `diag`/`work`
  // are sd x sd (p x p for primal, m x m for dual); `rhs` is length sd.
  rcupp_buf Xs   = need_buf(c, m * p);
  rcupp_buf ys   = need_buf(c, m);
  rcupp_buf diag = need_buf(c, sd * sd);
  rcupp_buf work = need_buf(c, sd * sd);
  rcupp_buf rhs  = need_buf(c, sd);
  rcupp_buf out  = need_buf(c, p);                  // dual beta staging

  // diag = m*lambda*I_sd (constant across subsets), uploaded once.
  std::vector<double> diag_h(static_cast<std::size_t>(sd) * sd, 0.0);
  for (int i = 0; i < sd; ++i)
    diag_h[i + static_cast<std::size_t>(i) * sd] = diag_val;
  rcupp_buf_upload(c, diag, diag_h.data(), sd * sd);

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
    info[b] = solve_subset(c, m, p, use_dual, Xs, ys, diag, work, rhs, out,
                           beta.begin() + static_cast<R_xlen_t>(b) * p);
  }

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta,
      Rcpp::Named("info") = info,
      Rcpp::Named("strategy") = strategy,
      Rcpp::Named("method") = method_used,
      Rcpp::Named("sys_dim") = sd,
      Rcpp::Named("m") = m,
      Rcpp::Named("diag") = diag_val,
      Rcpp::Named("cuda") = static_cast<bool>(rcupp_cuda_available()));
}
