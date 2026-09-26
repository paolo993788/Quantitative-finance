// Heston (1993) stochastic volatility model:
//   dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S
//   dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW_t^v,   d<W^S, W^v>_t = rho dt
//
// Semi-closed-form European prices use the characteristic function in the
// "little Heston trap" formulation (Albrecher, Mayer, Schoutens and Tistaert,
// 2007), which avoids the branch-cut discontinuity of the original paper.
// Monte Carlo prices use the Quadratic-Exponential (QE) scheme of Andersen
// (2008) for the variance and his central discretisation of log S.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "black_scholes.hpp"
#include "parallel.hpp"
#include "random.hpp"

namespace qe {

struct HestonParams {
    double v0;     // initial variance
    double kappa;  // mean-reversion speed
    double theta;  // long-run variance
    double sigma;  // volatility of variance
    double rho;    // correlation between the two Brownian motions

    void validate() const {
        if (!(v0 >= 0.0) || !(kappa > 0.0) || !(theta > 0.0) || !(sigma > 0.0) || !(rho >= -1.0 && rho <= 1.0)) {
            throw std::invalid_argument("invalid Heston parameters: need v0>=0, kappa>0, theta>0, sigma>0, |rho|<=1");
        }
    }
};

// Characteristic function of X_T = ln(S_T / S_0): E[exp(i u X_T)].
inline std::complex<double> heston_cf(std::complex<double> u, double T, double r, double q,
                                      const HestonParams& p) {
    using cd = std::complex<double>;
    const cd i(0.0, 1.0);
    const cd iu = i * u;
    const double s2 = p.sigma * p.sigma;
    const cd a = p.kappa - p.rho * p.sigma * iu;
    const cd d = std::sqrt(a * a + s2 * (iu + u * u));
    const cd g = (a - d) / (a + d);
    const cd e = std::exp(-d * T);
    const cd C = (r - q) * iu * T +
                 (p.kappa * p.theta / s2) * ((a - d) * T - 2.0 * std::log((1.0 - g * e) / (1.0 - g)));
    const cd D = ((a - d) / s2) * (1.0 - e) / (1.0 - g * e);
    return std::exp(C + D * p.v0);
}

// Gauss-Legendre nodes and weights on [-1, 1], computed by Newton iteration.
struct GaussLegendre {
    std::vector<double> nodes;
    std::vector<double> weights;
};

inline GaussLegendre make_gauss_legendre(int n) {
    GaussLegendre gl;
    gl.nodes.assign(n, 0.0);
    gl.weights.assign(n, 0.0);
    for (int i = 0; i < (n + 1) / 2; ++i) {
        double z = std::cos(kPi * (i + 0.75) / (n + 0.5));
        double pp = 0.0;
        for (int iter = 0; iter < 100; ++iter) {
            double p1 = 1.0, p2 = 0.0;
            for (int j = 1; j <= n; ++j) {
                const double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
            }
            pp = n * (z * p1 - p2) / (z * z - 1.0);
            const double z_old = z;
            z = z_old - p1 / pp;
            if (std::abs(z - z_old) < 1e-15) break;
        }
        gl.nodes[i] = -z;
        gl.nodes[n - 1 - i] = z;
        gl.weights[i] = gl.weights[n - 1 - i] = 2.0 / ((1.0 - z * z) * pp * pp);
    }
    return gl;
}

inline const GaussLegendre& gauss_legendre_16() {
    static const GaussLegendre gl = make_gauss_legendre(16);
    return gl;
}

// European call price by Fourier inversion:
//   C = (S e^{-qT} - K e^{-rT}) / 2
//       + e^{-rT} / pi * Int_0^inf Re[ e^{i u k} (S phi(u - i) - K phi(u)) / (i u) ] du,
// with k = ln(S / K) and phi the characteristic function of ln(S_T / S).
// The integral is evaluated with composite 16-point Gauss-Legendre panels and
// stops once several consecutive panels contribute less than abs_tol.
inline double heston_call_price(double S, double K, double T, double r, double q, const HestonParams& p,
                                double abs_tol = 1e-14) {
    p.validate();
    if (S <= 0.0 || K <= 0.0 || T <= 0.0) throw std::invalid_argument("S, K and T must be positive");
    using cd = std::complex<double>;
    const cd i(0.0, 1.0);
    const double k = std::log(S / K);
    auto integrand = [&](double u) {
        const cd phi_shift = heston_cf(cd(u, -1.0), T, r, q, p);
        const cd phi = heston_cf(cd(u, 0.0), T, r, q, p);
        const cd value = std::exp(i * u * k) * (S * phi_shift - K * phi) / (i * u);
        return value.real();
    };
    const GaussLegendre& gl = gauss_legendre_16();
    // Panel width scaled by the typical standard deviation of ln(S_T / S).
    const double scale = std::sqrt(std::max({p.v0, p.theta, 1e-4}) * T);
    const double h = 0.5 / scale;
    const double tol = abs_tol * std::max(S, K);
    double integral = 0.0;
    int small_panels = 0;
    for (int panel = 0; panel < 200000; ++panel) {
        const double a = panel * h;
        double contribution = 0.0;
        for (std::size_t j = 0; j < gl.nodes.size(); ++j) {
            const double u = a + 0.5 * h * (gl.nodes[j] + 1.0);
            contribution += gl.weights[j] * integrand(u);
        }
        contribution *= 0.5 * h;
        integral += contribution;
        small_panels = std::abs(contribution) < tol ? small_panels + 1 : 0;
        if (small_panels >= 4) break;
    }
    const double df_r = std::exp(-r * T);
    return 0.5 * (S * std::exp(-q * T) - K * df_r) + df_r / kPi * integral;
}

inline double heston_price(double S, double K, double T, double r, double q, const HestonParams& p,
                           bool is_call) {
    const double call = heston_call_price(S, K, T, r, q, p);
    if (is_call) return call;
    return call - S * std::exp(-q * T) + K * std::exp(-r * T);  // put-call parity
}

struct HestonMcResult {
    std::vector<double> call;
    std::vector<double> call_se;
    std::vector<double> put;
    std::vector<double> put_se;
    double forward = 0.0;     // Monte Carlo estimate of E[S_T]
    double forward_se = 0.0;  // its standard error
};

// Monte Carlo prices of European calls and puts for several strikes with the
// QE scheme (critical switching value psi_c = 1.5, central discretisation
// gamma_1 = gamma_2 = 1/2). Paths are simulated in blocks of 4096, each with
// its own random stream, so results are reproducible for a given seed
// whatever the number of threads.
inline HestonMcResult heston_mc_qe(double S0, const std::vector<double>& strikes, double T, double r,
                                   double q, const HestonParams& p, int n_steps, long long n_paths,
                                   std::uint64_t seed, int n_threads = 0) {
    p.validate();
    if (n_steps < 1 || n_paths < 2 || T <= 0.0 || S0 <= 0.0) {
        throw std::invalid_argument("need n_steps >= 1, n_paths >= 2, T > 0 and S0 > 0");
    }
    const std::size_t n_strikes = strikes.size();
    const double dt = T / n_steps;
    const double E = std::exp(-p.kappa * dt);
    const double s2 = p.sigma * p.sigma;
    const double gamma1 = 0.5, gamma2 = 0.5, psi_c = 1.5;
    const double K0 = -p.rho * p.kappa * p.theta * dt / p.sigma;
    const double K1 = gamma1 * dt * (p.kappa * p.rho / p.sigma - 0.5) - p.rho / p.sigma;
    const double K2 = gamma2 * dt * (p.kappa * p.rho / p.sigma - 0.5) + p.rho / p.sigma;
    const double K3 = gamma1 * dt * (1.0 - p.rho * p.rho);
    const double K4 = gamma2 * dt * (1.0 - p.rho * p.rho);
    const double drift = (r - q) * dt;
    const double lnS0 = std::log(S0);

    const long long block = 4096;
    const std::size_t n_blocks = static_cast<std::size_t>((n_paths + block - 1) / block);
    const std::size_t stride = 4 * n_strikes + 2;
    std::vector<double> acc(n_blocks * stride, 0.0);

    parallel_for(n_blocks, n_threads, [&](std::size_t b) {
        Xoshiro256 rng(seed, b);
        double* out = acc.data() + b * stride;
        const long long start = static_cast<long long>(b) * block;
        const long long stop = std::min(n_paths, start + block);
        for (long long path = start; path < stop; ++path) {
            double lnS = lnS0;
            double v = p.v0;
            for (int step = 0; step < n_steps; ++step) {
                const double m = p.theta + (v - p.theta) * E;
                const double s_sq = v * s2 * E * (1.0 - E) / p.kappa +
                                    p.theta * s2 * (1.0 - E) * (1.0 - E) / (2.0 * p.kappa);
                const double psi = s_sq / (m * m);
                double v_next;
                if (psi <= psi_c) {
                    const double inv = 2.0 / psi;
                    const double b2 = inv - 1.0 + std::sqrt(inv) * std::sqrt(inv - 1.0);
                    const double a = m / (1.0 + b2);
                    const double z = std::sqrt(b2) + rng.normal();
                    v_next = a * z * z;
                } else {
                    const double prob = (psi - 1.0) / (psi + 1.0);
                    const double beta = (1.0 - prob) / m;
                    const double u = rng.uniform();
                    v_next = u <= prob ? 0.0 : std::log((1.0 - prob) / (1.0 - u)) / beta;
                }
                const double var_term = std::max(K3 * v + K4 * v_next, 0.0);
                lnS += drift + K0 + K1 * v + K2 * v_next + std::sqrt(var_term) * rng.normal();
                v = v_next;
            }
            const double ST = std::exp(lnS);
            for (std::size_t j = 0; j < n_strikes; ++j) {
                const double c = std::max(ST - strikes[j], 0.0);
                const double pu = std::max(strikes[j] - ST, 0.0);
                out[4 * j] += c;
                out[4 * j + 1] += c * c;
                out[4 * j + 2] += pu;
                out[4 * j + 3] += pu * pu;
            }
            out[4 * n_strikes] += ST;
            out[4 * n_strikes + 1] += ST * ST;
        }
    });

    // Combine block sums in a fixed order for bitwise reproducibility.
    std::vector<double> total(stride, 0.0);
    for (std::size_t b = 0; b < n_blocks; ++b)
        for (std::size_t j = 0; j < stride; ++j) total[j] += acc[b * stride + j];

    const double n = static_cast<double>(n_paths);
    const double df = std::exp(-r * T);
    auto mean_se = [n](double sum, double sum_sq, double& mean, double& se) {
        mean = sum / n;
        const double var = std::max(sum_sq / n - mean * mean, 0.0) * n / (n - 1.0);
        se = std::sqrt(var / n);
    };
    HestonMcResult res;
    res.call.resize(n_strikes);
    res.call_se.resize(n_strikes);
    res.put.resize(n_strikes);
    res.put_se.resize(n_strikes);
    for (std::size_t j = 0; j < n_strikes; ++j) {
        double mean, se;
        mean_se(total[4 * j], total[4 * j + 1], mean, se);
        res.call[j] = df * mean;
        res.call_se[j] = df * se;
        mean_se(total[4 * j + 2], total[4 * j + 3], mean, se);
        res.put[j] = df * mean;
        res.put_se[j] = df * se;
    }
    mean_se(total[4 * n_strikes], total[4 * n_strikes + 1], res.forward, res.forward_se);
    return res;
}

}  // namespace qe
