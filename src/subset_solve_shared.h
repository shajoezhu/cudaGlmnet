#ifndef CG_SUBSET_SOLVE_SHARED_H
#define CG_SUBSET_SOLVE_SHARED_H

/* Shared closed-form subset-ridge solve, used by BOTH
 *   - resident_subsets.cpp  (self-contained CtxGuard version)
 *   - ctx_resident.cpp      (caller supplies a persistent rcupp_ctx XPtr)
 * Kept algorithmically identical so the two entry points agree to the bit.
 * The only difference is WHO owns the rcupp_ctx lifetime.
 */

#include <Rcpp.h>
#include <rcupp.h>

namespace cgsubset {

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
// `diag` = m*lambda*I_sd, `work` = solve buffer, `rhs` = length sd, `out` =
// dual beta staging (p). Downloads beta[p] into beta_out. Returns 0 / LAPACK
// info on success/failure.
inline int solve_subset(rcupp_ctx c, int m, int p, bool use_dual,
                        rcupp_buf Xs, rcupp_buf ys,
                        rcupp_buf diag, rcupp_buf work, rcupp_buf rhs,
                        rcupp_buf out, double* beta_out) {
  const int sd = use_dual ? m : p;
  rcupp_d_copy(c, sd * sd, diag, work);
  if (use_dual) {
    rcupp_d_gemm(c, RCUPP_OP_N, RCUPP_OP_T, m, m, p,
                 1.0, Xs, m, Xs, m, 1.0, work, m);
    rcupp_d_copy(c, m, ys, rhs);
  } else {
    rcupp_d_syrk(c, Xs, m, p, RCUPP_FILL_LOWER, 1.0, 1.0, work);
    rcupp_d_gemv_t(c, Xs, m, p, ys, rhs);
  }
  int info = 0;
  if (rcupp_d_potrf(c, RCUPP_FILL_LOWER, sd, work, sd, &info) != 0) return 2;
  if (info != 0) return info;
  if (rcupp_d_potrs(c, RCUPP_FILL_LOWER, sd, 1, work, sd, rhs, sd, &info) != 0)
    return 3;
  if (use_dual) {
    rcupp_d_gemv_t(c, Xs, m, p, rhs, out);
    rcupp_buf_download(c, out, beta_out, p);
  } else {
    rcupp_buf_download(c, rhs, beta_out, p);
  }
  return info;
}

}  // namespace cgsubset

#endif  // CG_SUBSET_SOLVE_SHARED_H
