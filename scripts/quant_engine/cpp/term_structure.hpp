// Kalman filter of the dynamic Nelson-Siegel model (Diebold, Rudebusch and Aruoba, 2006):
//   y_t = L(lambda) f_t + e_t,  e_t ~ N(0, diag(h^2)),
//   f_t - mu = A (f_{t-1} - mu) + u_t,  u_t ~ N(0, Q).
// Covariance form with sequential processing: the measurement covariance is diagonal, so the maturities
// observed on a date are assimilated one at a time, each with the scalar innovation variance s = l' P l + h^2,
// the gain k = P l / s and the Joseph update P+ = (I - k l') P (I - k l')' + h^2 k k', which stays symmetric
// and positive semi-definite in floating point. This needs 3 x 3 operations only and keeps full precision
// whatever the measurement standard deviations; the information form M = P^{-1} + L' H^{-1} L used earlier
// lost the prior to rounding once some h approached the 0.01 bp floor allowed by the estimator
// (log-likelihood errors of 1e-3 and wrong gradients). Missing observations (NaN) are skipped.
#pragma once

#include <algorithm>
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

}  // namespace detail

// Nelson-Siegel loadings [1, (1 - e^{-x}) / x, (1 - e^{-x}) / x - e^{-x}], x = lambda * maturity (n x 3, row-major).
inline std::vector<double> ns_loadings(const std::vector<double>& maturities, double lam) {
    std::vector<double> L(maturities.size() * 3);
    for (std::size_t i = 0; i < maturities.size(); ++i) {
        const double x = lam * std::max(maturities[i], 1e-10);
        const double slope = -std::expm1(-x) / x;  // (1 - e^{-x}) / x without cancellation for small x
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
    std::vector<double> h2(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!(h[i] > 0.0)) throw std::invalid_argument("measurement standard deviations must be positive");
        h2[i] = h[i] * h[i];
    }
    DnsKalmanResult res;
    if (keep_states) {
        res.filtered.assign(T * 3, 0.0);
        res.predicted.assign(T * 3, 0.0);
    }
    {
        double lp[9];
        if (!detail::chol3(P0.data(), lp)) {   // the prior must be a covariance matrix
            res.loglik = -std::numeric_limits<double>::infinity();
            return res;
        }
    }
    double x[3] = {mu[0], mu[1], mu[2]};
    double P[9];
    for (int i = 0; i < 9; ++i) P[i] = P0[i];
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
        // Sequential processing of the date's observations: with a diagonal H, updating one maturity at a time
        // gives the same posterior and the same likelihood as the batch update (the innovations are conditioned
        // on the maturities already processed), with 3-vectors and 3 x 3 matrices only.
        for (std::size_t i = 0; i < n; ++i) {
            const double y = Y[t * n + i];
            if (!std::isfinite(y)) continue;
            const double* li = &L[i * 3];
            double Pl[3];
            for (int a = 0; a < 3; ++a) Pl[a] = P[a * 3] * li[0] + P[a * 3 + 1] * li[1] + P[a * 3 + 2] * li[2];
            const double s = li[0] * Pl[0] + li[1] * Pl[1] + li[2] * Pl[2] + h2[i];
            if (!(s > 0.0)) {
                res.loglik = -std::numeric_limits<double>::infinity();
                return res;
            }
            const double v = y - (li[0] * x[0] + li[1] * x[1] + li[2] * x[2]);
            res.loglik -= 0.5 * (log2pi + std::log(s) + v * v / s);
            const double k[3] = {Pl[0] / s, Pl[1] / s, Pl[2] / s};
            for (int a = 0; a < 3; ++a) x[a] += k[a] * v;
            // Joseph form: P = (I - k l') P (I - k l')' + h^2 k k'.
            double IKL[9], B[9];
            for (int a = 0; a < 3; ++a)
                for (int c = 0; c < 3; ++c) IKL[a * 3 + c] = (a == c ? 1.0 : 0.0) - k[a] * li[c];
            for (int a = 0; a < 3; ++a)
                for (int c = 0; c < 3; ++c)
                    B[a * 3 + c] = IKL[a * 3] * P[c] + IKL[a * 3 + 1] * P[3 + c] + IKL[a * 3 + 2] * P[6 + c];
            for (int a = 0; a < 3; ++a)
                for (int c = a; c < 3; ++c) {
                    const double value = B[a * 3] * IKL[c * 3] + B[a * 3 + 1] * IKL[c * 3 + 1] + B[a * 3 + 2] * IKL[c * 3 + 2] +
                                         h2[i] * k[a] * k[c];
                    P[a * 3 + c] = P[c * 3 + a] = value;
                }
        }
        if (keep_states)
            for (int i = 0; i < 3; ++i) res.filtered[t * 3 + i] = x[i];
    }
    return res;
}

}  // namespace qe
