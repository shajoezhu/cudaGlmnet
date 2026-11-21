#include <Rcpp.h>
#include <rcupp.h>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>
#include "subset_solve_shared.h"

/*
 * Part B: the SAME closed-form subset-ridge composition as
 * resident_subsets.cpp, but the caller supplies a PERSISTENT rcupp device
 * context (an EXTPTRSXP from rcupp::ctx_create()) instead of one being made
 * and destroyed inside this .Call. That lets a framework learner hold ONE ctx
 * across many train()/fold calls, amortising cublasCreate / cudaMalloc.
 *
 * rcupp itself gains NOTHING here: this is a DOWNSTREAM composition in
 * cudaGlmnet, using only the generic rcupp C ABI. The solve arithmetic is
 * bit-identical to resident_subsets.cpp because both include
 * subset_solve_shared.h. idx_list is ZERO-BASED (the R wrapper converts).
 *
 * Buffers are owned by the supplied ctx and are freed when that ctx's XPtr is
 * finalised/destroyed -- hence rcupp::ctx must outlive this call (guaranteed
 * by the R-level learner holding the XPtr for its whole life).
 */

// [[Rcpp::export]]
Rcpp::List ctx_resident_subsets_cpp(SEXP ctx,
                                    Rcpp::NumericMatrix x,
                                    Rcpp::NumericVector y,
                                    Rcpp::List idx_list,
                                    double lambda,
                                    std::string strategy,
                                    std::string method) {
  if (rcupp_ctx_abi() != 1) {
    Rcpp::stop("rcupp >= 0.1.0 (device-context ABI 1) is required");
  }
  if (TYPEOF(ctx) != EXTPTRSXP) {
    Rcpp::stop("'ctx' is not a valid rcupp device context (external pointer)");
  }
  rcupp_ctx c = static_cast<rcupp_ctx>(R_ExternalPtrAddr(ctx));
  if (c == NULL) {
    Rcpp::stop("this rcupp device context has already been destroyed");
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
    Rcpp::stop("invalid ctx_resident_subsets_cpp parameters");
  }

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
  const bool use_dual = (method == "dual") || (method == "auto" && p > m);
  const int sd = use_dual ? m : p;
  const std::string method_used = use_dual ? "dual" : "primal";

  using cgsubset::need_buf;
  using cgsubset::solve_subset;

  rcupp_buf Xs   = need_buf(c, m * p);
  rcupp_buf ys   = need_buf(c, m);
  rcupp_buf diag = need_buf(c, sd * sd);
  rcupp_buf work = need_buf(c, sd * sd);
  rcupp_buf rhs  = need_buf(c, sd);
  rcupp_buf out  = need_buf(c, p);

  std::vector<double> diag_h(static_cast<std::size_t>(sd) * sd, 0.0);
  for (int i = 0; i < sd; ++i)
    diag_h[i + static_cast<std::size_t>(i) * sd] = diag_val;
  rcupp_buf_upload(c, diag, diag_h.data(), sd * sd);

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
      rcupp_d_gather_rows(c, Xfull, n, p, ix, m, Xs);
      rcupp_d_gather_rows(c, yfull, n, 1, ix, m, ys);
    } else {
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
