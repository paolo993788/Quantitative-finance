// Black-Scholes-Merton closed-form prices, Greeks and implied volatility for
// European options on an asset with continuous dividend yield q.
// Rates r and q are continuously compounded; T is in years.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace qe {

constexpr double kPi = 3.14159265358979323846;

inline double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

inline double norm_pdf(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * kPi); }

struct BSResult {
    double price;
    double delta;
    double gamma;
    double vega;   // per unit of volatility (not per percentage point)
    double theta;  // per year, dV/dt with calendar time running forward
    double rho;    // per unit of interest rate
};

inline BSResult bs_price_greeks(double S, double K, double T, double r, double q, double sigma,
                                bool is_call) {
    if (S <= 0.0 || K <= 0.0) throw std::invalid_argument("S and K must be positive");
    if (T < 0.0 || sigma < 0.0) throw std::invalid_argument("T and sigma must be non-negative");
    const double df_r = std::exp(-r * T);
    const double df_q = std::exp(-q * T);
    const double sd = sigma * std::sqrt(T);
    BSResult out{};
    if (sd < 1e-14) {
        // Degenerate case: the payoff is known at inception.
        const double forward_intrinsic = S * df_q - K * df_r;
        const double value = is_call ? std::max(forward_intrinsic, 0.0) : std::max(-forward_intrinsic, 0.0);
        const bool in_the_money = is_call ? forward_intrinsic > 0.0 : forward_intrinsic < 0.0;
        out.price = value;
        out.delta = in_the_money ? (is_call ? df_q : -df_q) : 0.0;
        return out;
    }
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sd;
    const double d2 = d1 - sd;
    const double pdf_d1 = norm_pdf(d1);
    out.gamma = df_q * pdf_d1 / (S * sd);
    out.vega = S * df_q * pdf_d1 * std::sqrt(T);
    const double theta_common = -S * df_q * pdf_d1 * sigma / (2.0 * std::sqrt(T));
    if (is_call) {
        out.price = S * df_q * norm_cdf(d1) - K * df_r * norm_cdf(d2);
        out.delta = df_q * norm_cdf(d1);
        out.theta = theta_common - r * K * df_r * norm_cdf(d2) + q * S * df_q * norm_cdf(d1);
        out.rho = K * T * df_r * norm_cdf(d2);
    } else {
        out.price = K * df_r * norm_cdf(-d2) - S * df_q * norm_cdf(-d1);
        out.delta = -df_q * norm_cdf(-d1);
        out.theta = theta_common + r * K * df_r * norm_cdf(-d2) - q * S * df_q * norm_cdf(-d1);
        out.rho = -K * T * df_r * norm_cdf(-d2);
    }
    return out;
}

inline double bs_price(double S, double K, double T, double r, double q, double sigma, bool is_call) {
    return bs_price_greeks(S, K, T, r, q, sigma, is_call).price;
}

// Implied volatility by Newton's method safeguarded with bisection. Returns
// NaN when the price violates the no-arbitrage bounds.
inline double bs_implied_vol(double price, double S, double K, double T, double r, double q,
                             bool is_call, double tol = 1e-12, int max_iter = 200) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(price > 0.0) || T <= 0.0) return nan;
    const double df_r = std::exp(-r * T);
    const double df_q = std::exp(-q * T);
    const double lower = is_call ? std::max(S * df_q - K * df_r, 0.0) : std::max(K * df_r - S * df_q, 0.0);
    const double upper = is_call ? S * df_q : K * df_r;
    if (price <= lower || price >= upper) return nan;

    double lo = 1e-9;
    double hi = 10.0;
    if (bs_price(S, K, T, r, q, hi, is_call) < price) return nan;
    // Brenner-Subrahmanyam style starting point, kept inside the bracket.
    double sigma = std::sqrt(2.0 * std::abs(std::log(S * df_q / (K * df_r))) / T);
    sigma = std::clamp(std::max(sigma, 0.2), lo, hi);
    for (int it = 0; it < max_iter; ++it) {
        const BSResult res = bs_price_greeks(S, K, T, r, q, sigma, is_call);
        const double diff = res.price - price;
        if (std::abs(diff) < tol * std::max(1.0, price)) return sigma;
        if (diff > 0.0) hi = sigma; else lo = sigma;
        double candidate = res.vega > 1e-300 ? sigma - diff / res.vega : 0.5 * (lo + hi);
        if (!(candidate > lo && candidate < hi)) candidate = 0.5 * (lo + hi);
        if (std::abs(candidate - sigma) < 1e-15) return candidate;
        sigma = candidate;
    }
    return sigma;
}

}  // namespace qe
