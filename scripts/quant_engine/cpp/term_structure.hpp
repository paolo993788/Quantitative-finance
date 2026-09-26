// Kalman filter of the dynamic Nelson-Siegel model (Diebold, Rudebusch and Aruoba, 2006):
//   y_t = L(lambda) f_t + e_t,  e_t ~ N(0, diag(h^2)),
//   f_t - mu = A (f_{t-1} - mu) + u_t,  u_t ~ N(0, Q).
// With a diagonal measurement covariance the update uses the Woodbury identity and the matrix
// determinant lemma, so each step only needs 3 x 3 factorisations:
//   M = P^{-1} + L' H^{-1} L,  log det S = log det H + log det P + log det M,
//   v' S^{-1} v = v' H^{-1} v - b' M^{-1} b with b = L' H^{-1} v,  x+ = x + M^{-1} b,  P+ = M^{-1}.
// Missing observations (NaN) are skipped.
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qe {

struct DnsKalmanResult {
    double loglik = 0.0;
    std::vector<double> filtered;   // T x 3, row-major
    std::vector<double> predicted;  // T x 3, row-major
};

namespace detail {

// Cholesky factor of a symmetric 3 x 3 matrix (row-major); false if not positive definite.
inline bool chol3(const double* a, double* l) {
    for (int i = 0; i < 9; ++i) l[i] = 0.0;
    for (int j = 0; j < 3; ++j) {
        double s = a[j * 3 + j];
        for (int k = 0; k < j; ++k) s -= l[j * 3 + k] * l[j * 3 + k];
        if (!(s > 0.0)) return false;
        l[j * 3 + j] = std::sqrt(s);
        for (int i = j + 1; i < 3; ++i) {
            double t = a[i * 3 + j];
            for (int k = 0; k < j; ++k) t -= l[i * 3 + k] * l[j * 3 + k];
            l[i * 3 + j] = t / l[j * 3 + j];
        }
    }
    return true;
}

// Inverse of a symmetric positive definite 3 x 3 matrix from its Cholesky factor.
inline void chol3_inverse(const double* l, double* inv) {
    double li[9] = {0.0};
    for (int i = 0; i < 3; ++i) {
        li[i * 3 + i] = 1.0 / l[i * 3 + i];
        for (int j = 0; j < i; ++j) {
            double s = 0.0;
            for (int k = j; k < i; ++k) s += l[i * 3 + k] * li[k * 3 + j];
            li[i * 3 + j] = -s / l[i * 3 + i];
        }
    }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += li[k * 3 + i] * li[k * 3 + j];
            inv[i * 3 + j] = s;
        }
}

inline double chol3_logdet(const double* l) {
    return 2.0 * (std::log(l[0]) + std::log(l[4]) + std::log(l[8]));
}

}  // namespace detail

// Nelson-Siegel loadings [1, (1 - e^{-x}) / x, (1 - e^{-x}) / x - e^{-x}], x = lambda * maturity (n x 3, row-major).
inline std::vector<double> ns_loadings(const std::vector<double>& maturities, double lam) {
    std::vector<double> L(maturities.size() * 3);
    for (std::size_t i = 0; i < maturities.size(); ++i) {
        const double x = lam * std::max(maturities[i], 1e-10);
        const double slope = (1.0 - std::exp(-x)) / x;
        L[i * 3] = 1.0;
        L[i * 3 + 1] = slope;
        L[i * 3 + 2] = slope - std::exp(-x);
    }
    return L;
}

inline DnsKalmanResult dns_kalman_filter(const std::vector<double>& Y, std::size_t T, std::size_t n,
                                         const std::vector<double>& maturities, double lam, const std::vector<double>& mu,
                                         const std::vector<double>& A, const std::vector<double>& Q,
                                         const std::vector<double>& h, const std::vector<double>& P0, bool keep_states) {
    if (Y.size() != T * n || maturities.size() != n || h.size() != n || mu.size() != 3 || A.size() != 9 || Q.size() != 9 ||
        P0.size() != 9)
        throw std::invalid_argument("inconsistent dimensions in dns_kalman_filter");
    const double log2pi = std::log(2.0 * 3.14159265358979323846);
    const std::vector<double> L = ns_loadings(maturities, lam);
    std::vector<double> inv_h2(n), log_h2(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!(h[i] > 0.0)) throw std::invalid_argument("measurement standard deviations must be positive");
        inv_h2[i] = 1.0 / (h[i] * h[i]);
        log_h2[i] = 2.0 * std::log(h[i]);
    }
    DnsKalmanResult res;
    if (keep_states) {
        res.filtered.assign(T * 3, 0.0);
        res.predicted.assign(T * 3, 0.0);
    }
    double x[3] = {mu[0], mu[1], mu[2]};
    double P[9];
    for (int i = 0; i < 9; ++i) P[i] = P0[i];
    double lp[9], pinv[9], M[9], lm[9], minv[9];
    for (std::size_t t = 0; t < T; ++t) {
        if (t > 0) {
            double d[3], xn[3], AP[9], Pn[9];
            for (int i = 0; i < 3; ++i) d[i] = x[i] - mu[i];
            for (int i = 0; i < 3; ++i) xn[i] = mu[i] + A[i * 3] * d[0] + A[i * 3 + 1] * d[1] + A[i * 3 + 2] * d[2];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) AP[i * 3 + j] = A[i * 3] * P[j] + A[i * 3 + 1] * P[3 + j] + A[i * 3 + 2] * P[6 + j];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    Pn[i * 3 + j] = AP[i * 3] * A[j * 3] + AP[i * 3 + 1] * A[j * 3 + 1] + AP[i * 3 + 2] * A[j * 3 + 2] + Q[i * 3 + j];
            for (int i = 0; i < 3; ++i) x[i] = xn[i];
            for (int i = 0; i < 9; ++i) P[i] = Pn[i];
        }
        if (keep_states)
            for (int i = 0; i < 3; ++i) res.predicted[t * 3 + i] = x[i];
        double b[3] = {0.0, 0.0, 0.0}, G[9] = {0.0};
        double quad = 0.0, logdet_h = 0.0;
        std::size_t m = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double y = Y[t * n + i];
            if (!std::isfinite(y)) continue;
            const double* li = &L[i * 3];
            const double v = y - (li[0] * x[0] + li[1] * x[1] + li[2] * x[2]);
            quad += v * v * inv_h2[i];
            logdet_h += log_h2[i];
            for (int a = 0; a < 3; ++a) {
                b[a] += li[a] * inv_h2[i] * v;
                for (int c = 0; c < 3; ++c) G[a * 3 + c] += li[a] * inv_h2[i] * li[c];
            }
            ++m;
        }
        if (m > 0) {
            if (!detail::chol3(P, lp)) {
                res.loglik = -std::numeric_limits<double>::infinity();
                return res;
            }
            detail::chol3_inverse(lp, pinv);
            for (int i = 0; i < 9; ++i) M[i] = pinv[i] + G[i];
            if (!detail::chol3(M, lm)) {
                res.loglik = -std::numeric_limits<double>::infinity();
                return res;
            }
            detail::chol3_inverse(lm, minv);
            double mb[3];
            for (int i = 0; i < 3; ++i) mb[i] = minv[i * 3] * b[0] + minv[i * 3 + 1] * b[1] + minv[i * 3 + 2] * b[2];
            const double quad_s = quad - (b[0] * mb[0] + b[1] * mb[1] + b[2] * mb[2]);
            res.loglik -= 0.5 * (static_cast<double>(m) * log2pi + logdet_h + detail::chol3_logdet(lp) +
                                 detail::chol3_logdet(lm) + quad_s);
            for (int i = 0; i < 3; ++i) x[i] += mb[i];
            for (int i = 0; i < 9; ++i) P[i] = minv[i];
        }
        if (keep_states)
            for (int i = 0; i < 3; ++i) res.filtered[t * 3 + i] = x[i];
    }
    return res;
}

}  // namespace qe
