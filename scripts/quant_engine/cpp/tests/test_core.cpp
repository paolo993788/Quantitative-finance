// Unit tests of the header-only C++ engines, compiled without Python (see ../CMakeLists.txt).
//
// The Python test suite (tests/quant_engine) compares the engines with NumPy and SciPy reference implementations
// through the pybind11 bindings. These tests check the C++ code on its own, so that it can be built with strict
// warnings and run under AddressSanitizer, UndefinedBehaviorSanitizer and ThreadSanitizer:
// * published reference outputs of the random number generators and published option prices
//   (Hull; Longstaff and Schwartz, 2001; Fang and Oosterlee, 2008);
// * exact identities (put-call parity, Greeks as derivatives of the price, martingale property of the
//   characteristic function, cost accounting of the hedging simulation, the Kalman filter in matrix form);
// * limiting cases (small vol of vol, Student-t with many degrees of freedom, constant variance);
// * results that must not depend on the number of threads;
// * argument validation.
// Statistical checks use four standard errors (two-sided type I error about 6e-5 each) with fixed seeds.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "black_scholes.hpp"
#include "finite_difference.hpp"
#include "garch.hpp"
#include "hedging.hpp"
#include "heston.hpp"
#include "optimize.hpp"
#include "parallel.hpp"
#include "random.hpp"
#include "term_structure.hpp"

namespace {

int g_checks = 0;
int g_failures = 0;

void report(bool ok, const char* what, const char* file, int line) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, what);
    }
}

void report_close(double a, double b, double tol, const char* what, const char* file, int line) {
    const bool ok = std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= tol;
    report(ok, what, file, line);
    if (!ok) std::fprintf(stderr, "    %.17g vs %.17g (tolerance %.3g)\n", a, b, tol);
}

#define CHECK(cond) report((cond), #cond, __FILE__, __LINE__)
#define CHECK_CLOSE(a, b, tol) report_close((a), (b), (tol), #a " ~ " #b, __FILE__, __LINE__)

template <class Fn>
bool throws_invalid_argument(Fn&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

double mean_of(const std::vector<double>& x) {
    double s = 0.0;
    for (double v : x) s += v;
    return s / static_cast<double>(x.size());
}

double sd_of(const std::vector<double>& x) {
    const double m = mean_of(x);
    double s = 0.0;
    for (double v : x) s += (v - m) * (v - m);
    return std::sqrt(s / static_cast<double>(x.size() - 1));
}

// ------------------------------------------------------------------------------------------------ random numbers

void test_splitmix64_reference() {
    // Reference sequence of Vigna's splitmix64.c for the seed 1234567.
    std::uint64_t state = 1234567;
    const std::uint64_t expected[] = {6457827717110365317ULL, 3203168211198807973ULL, 9817491932198370423ULL,
                                      4593380528125082431ULL, 16408922859458223821ULL};
    for (std::uint64_t e : expected) CHECK(qe::splitmix64(state) == e);
}

void test_xoshiro_streams() {
    // Values from a transcription of the reference xoshiro256** algorithm (Blackman and Vigna), with the state
    // seeded as in qe::Xoshiro256; the transcription reproduces the published outputs from the state {1, 2, 3, 4}
    // (11520, 0, 1509978240, 1215971899390074240).
    qe::Xoshiro256 a(42, 0), b(42, 7);
    const std::uint64_t ea[] = {17604071880264941726ULL, 13049929662915288091ULL, 16314220431934199612ULL,
                                16017857136869241811ULL};
    const std::uint64_t eb[] = {15260628496888913177ULL, 12080209494996618911ULL, 16192203974094658541ULL,
                                4943040130464974858ULL};
    for (int k = 0; k < 4; ++k) {
        CHECK(a.next() == ea[k]);
        CHECK(b.next() == eb[k]);
    }
}

void test_uniform_and_normal_moments() {
    const std::size_t n = 1000000;
    const double dn = static_cast<double>(n);
    qe::Xoshiro256 rng(2026);
    double s = 0.0, q = 0.0, lo = 1.0, hi = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double u = rng.uniform();
        s += u;
        q += u * u;
        lo = std::min(lo, u);
        hi = std::max(hi, u);
    }
    CHECK(lo > 0.0 && hi < 1.0);  // open interval
    CHECK_CLOSE(s / dn, 0.5, 4.0 * std::sqrt(1.0 / 12.0 / dn));
    // Sample variance 1/12 with standard error sqrt((mu4 - sigma^4) / n) = sqrt(1 / (180 n)).
    CHECK_CLOSE(q / dn - (s / dn) * (s / dn), 1.0 / 12.0, 4.0 * std::sqrt(1.0 / 180.0 / dn));

    qe::Xoshiro256 g(7);
    double m = 0.0, v = 0.0, tail = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double z = g.normal();
        m += z;
        v += z * z;
        tail += std::abs(z) > 1.959963984540054 ? 1.0 : 0.0;
    }
    m /= dn;
    v = v / dn - m * m;
    CHECK_CLOSE(m, 0.0, 4.0 / std::sqrt(dn));
    CHECK_CLOSE(v, 1.0, 4.0 * std::sqrt(2.0 / dn));
    CHECK_CLOSE(tail / dn, 0.05, 4.0 * std::sqrt(0.05 * 0.95 / dn));
}

void test_gamma_variates() {
    // Marsaglia-Tsang: mean = variance = shape; shape < 1 exercises the boosting branch.
    const std::size_t n = 200000;
    for (double shape : {0.7, 3.5}) {
        qe::Xoshiro256 rng(11);
        std::vector<double> x(n);
        for (auto& v : x) v = qe::gamma_variate(rng, shape);
        CHECK(*std::min_element(x.begin(), x.end()) > 0.0);
        CHECK_CLOSE(mean_of(x), shape, 4.0 * std::sqrt(shape / static_cast<double>(n)));
        // Variance of the sample variance of a gamma variable: (mu4 - sigma^4) / n with mu4 = 3 k^2 + 6 k.
        const double sd = sd_of(x);
        CHECK_CLOSE(sd * sd, shape, 4.0 * std::sqrt((2.0 * shape * shape + 6.0 * shape) / static_cast<double>(n)));
    }
}

// ------------------------------------------------------------------------------------------------ parallel loop

void test_parallel_for() {
    const std::size_t n = 1000;
    for (int threads : {1, 3, 8}) {
        std::vector<double> out(n, 0.0);
        qe::parallel_for(n, threads, [&](std::size_t i) { out[i] = static_cast<double>(i) * static_cast<double>(i); });
        double total = 0.0;
        for (double v : out) total += v;
        CHECK(total == 332833500.0);  // sum of i^2 for i < 1000
    }
    bool rethrown = false;
    try {
        qe::parallel_for(100, 4, [](std::size_t i) {
            if (i == 57) throw std::runtime_error("task failed");
        });
    } catch (const std::runtime_error&) {
        rethrown = true;
    }
    CHECK(rethrown);
    CHECK(qe::resolve_threads(3) == 3);
    CHECK(qe::resolve_threads(0) >= 1);
}

// ------------------------------------------------------------------------------------------------ Black-Scholes

void test_black_scholes_reference_values() {
    // Hull, Options, Futures and Other Derivatives: S = K = 100, T = 1, r = 5%, sigma = 20% gives 10.4506
    // (published to four decimals), and Example 15.6 (S = 42, K = 40, T = 0.5, r = 10%, sigma = 20%) gives
    // c = 4.76 and p = 0.81 (two decimals).
    CHECK_CLOSE(qe::bs_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, true), 10.4506, 5e-5);
    CHECK_CLOSE(qe::bs_price(42.0, 40.0, 0.5, 0.10, 0.0, 0.2, true), 4.76, 5e-3);
    CHECK_CLOSE(qe::bs_price(42.0, 40.0, 0.5, 0.10, 0.0, 0.2, false), 0.81, 5e-3);
}

void test_black_scholes_identities() {
    const double cases[][6] = {{100, 100, 1.0, 0.05, 0.0, 0.2}, {100, 80, 0.25, 0.02, 0.03, 0.35},
                               {50, 70, 2.0, 0.0, 0.01, 0.15}, {1.1, 1.05, 0.5, 0.04, 0.02, 0.08}};
    for (const auto& c : cases) {
        const double S = c[0], K = c[1], T = c[2], r = c[3], q = c[4], sigma = c[5];
        const qe::BSResult call = qe::bs_price_greeks(S, K, T, r, q, sigma, true);
        const qe::BSResult put = qe::bs_price_greeks(S, K, T, r, q, sigma, false);
        // Put-call parity and the parity relations of the Greeks.
        CHECK_CLOSE(call.price - put.price, S * std::exp(-q * T) - K * std::exp(-r * T), 1e-12 * S);
        CHECK_CLOSE(call.delta - put.delta, std::exp(-q * T), 1e-14);
        CHECK_CLOSE(call.gamma, put.gamma, 1e-14);
        CHECK_CLOSE(call.vega, put.vega, 1e-12 * S);
        // Greeks as central differences of the price: truncation O(h^2), rounding O(eps / h).
        for (bool is_call : {true, false}) {
            const qe::BSResult g = is_call ? call : put;
            auto price = [&](double s_, double t_, double r_, double v_) {
                return qe::bs_price(s_, K, t_, r_, q, v_, is_call);
            };
            // Steps: 1e-4 S for the spot (truncation about hs^2 / 6 times the third derivative), 1e-5 otherwise.
            const double hs = 1e-4 * S, h = 1e-5;
            CHECK_CLOSE(g.delta, (price(S + hs, T, r, sigma) - price(S - hs, T, r, sigma)) / (2 * hs), 1e-6);
            CHECK_CLOSE(g.gamma, (price(S + hs, T, r, sigma) - 2 * g.price + price(S - hs, T, r, sigma)) / (hs * hs),
                        1e-5 / S);
            CHECK_CLOSE(g.vega, (price(S, T, r, sigma + h) - price(S, T, r, sigma - h)) / (2 * h), 1e-6 * S);
            CHECK_CLOSE(g.rho, (price(S, T, r + h, sigma) - price(S, T, r - h, sigma)) / (2 * h), 1e-6 * S);
            // theta is dV/dt with calendar time running forward, i.e. -dV/dT.
            CHECK_CLOSE(g.theta, -(price(S, T + h, r, sigma) - price(S, T - h, r, sigma)) / (2 * h), 1e-6 * S);
            // No-arbitrage bounds.
            CHECK(g.price >= (is_call ? std::max(S * std::exp(-q * T) - K * std::exp(-r * T), 0.0)
                                      : std::max(K * std::exp(-r * T) - S * std::exp(-q * T), 0.0)) - 1e-12);
            CHECK(g.price <= (is_call ? S * std::exp(-q * T) : K * std::exp(-r * T)));
        }
    }
    // Degenerate cases: no volatility or no time left.
    CHECK_CLOSE(qe::bs_price(100, 90, 1.0, 0.05, 0.0, 0.0, true), 100 - 90 * std::exp(-0.05), 1e-12);
    CHECK(qe::bs_price(100, 110, 1.0, 0.0, 0.0, 0.0, true) == 0.0);
    CHECK_CLOSE(qe::bs_price(100, 110, 0.0, 0.05, 0.0, 0.3, false), 10.0, 1e-12);
    CHECK(throws_invalid_argument([] { qe::bs_price(-1.0, 100, 1, 0, 0, 0.2, true); }));
    CHECK(throws_invalid_argument([] { qe::bs_price(100, 100, 1, 0, 0, -0.2, true); }));
}

void test_implied_volatility() {
    for (double K : {60.0, 90.0, 100.0, 115.0, 160.0})
        for (double T : {0.05, 0.5, 3.0})
            for (double sigma : {0.05, 0.2, 0.8})
                for (bool is_call : {true, false}) {
                    const qe::BSResult g = qe::bs_price_greeks(100.0, K, T, 0.03, 0.01, sigma, is_call);
                    const double lower = std::max(is_call ? 100.0 * std::exp(-0.01 * T) - K * std::exp(-0.03 * T)
                                                          : K * std::exp(-0.03 * T) - 100.0 * std::exp(-0.01 * T), 0.0);
                    // A time value below 1e-8 carries no usable information about the volatility.
                    if (g.price - lower < 1e-8) continue;
                    const double iv = qe::bs_implied_vol(g.price, 100.0, K, T, 0.03, 0.01, is_call);
                    // The solver stops when the price matches to 1e-12; the volatility is then accurate to about
                    // 1e-12 / vega, which is checked where the vega exceeds 1e-3.
                    CHECK_CLOSE(qe::bs_price(100.0, K, T, 0.03, 0.01, iv, is_call), g.price, 1e-11 * std::max(1.0, g.price));
                    if (g.vega > 1e-3) CHECK_CLOSE(iv, sigma, 1e-7);
                }
    // Prices outside the no-arbitrage bounds have no implied volatility.
    CHECK(std::isnan(qe::bs_implied_vol(0.5 * (100 - 90 * std::exp(-0.03)), 100, 90, 1, 0.03, 0, true)));
    CHECK(std::isnan(qe::bs_implied_vol(100.5, 100, 90, 1, 0.03, 0, true)));
    CHECK(std::isnan(qe::bs_implied_vol(-1.0, 100, 90, 1, 0.03, 0, true)));
}

// ------------------------------------------------------------------------------------------------ finite differences

void test_tridiagonal_solver() {
    // Constant tridiagonal system with diagonal 4 and off-diagonals -1: recover a known solution.
    const std::vector<double> x = {1.0, -2.0, 0.5, 3.0, 4.0, -1.0};
    std::vector<double> rhs(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        rhs[i] = 4.0 * x[i] - (i > 0 ? x[i - 1] : 0.0) - (i + 1 < x.size() ? x[i + 1] : 0.0);
    qe::solve_tridiagonal(-1.0, 4.0, -1.0, rhs);
    for (std::size_t i = 0; i < x.size(); ++i) CHECK_CLOSE(rhs[i], x[i], 1e-14);
}

void test_finite_difference_european() {
    const double S = 100, K = 105, T = 0.75, r = 0.04, q = 0.01, sigma = 0.25;
    for (bool is_call : {true, false}) {
        const qe::BSResult exact = qe::bs_price_greeks(S, K, T, r, q, sigma, is_call);
        const qe::FdResult fine = qe::bs_finite_difference(S, K, T, r, q, sigma, is_call, false, 400, 400);
        const qe::FdResult coarse = qe::bs_finite_difference(S, K, T, r, q, sigma, is_call, false, 100, 100);
        // Crank-Nicolson with a Rannacher start is second order: refining the grid four times in space and time
        // must cut the error by at least 16. (Beyond about 400 nodes the error stops falling monotonically,
        // because the truncation of the domain at five standard deviations and the position of the strike between
        // nodes dominate.)
        CHECK_CLOSE(fine.price, exact.price, 1e-4);
        CHECK_CLOSE(fine.delta, exact.delta, 5e-5);
        CHECK_CLOSE(fine.gamma, exact.gamma, 1e-6);
        CHECK(16.0 * std::abs(fine.price - exact.price) < std::abs(coarse.price - exact.price));
    }
    CHECK(throws_invalid_argument([] { qe::bs_finite_difference(100, 100, 0.0, 0.0, 0.0, 0.2, true, false); }));
    CHECK(throws_invalid_argument([] { qe::bs_finite_difference(100, 100, 1.0, 0.0, 0.0, 0.2, true, false, 4); }));
}

// American put on a Cox-Ross-Rubinstein binomial tree, an independent method (error O(1/n), oscillating in n).
double crr_american_put(double S, double K, double T, double r, double sigma, int n) {
    const double dt = T / n, u = std::exp(sigma * std::sqrt(dt)), p = (std::exp(r * dt) - 1.0 / u) / (u - 1.0 / u);
    const double disc = std::exp(-r * dt);
    const std::size_t m = static_cast<std::size_t>(n);
    std::vector<double> spot(2 * m + 1);  // S u^k for k = -n, ..., n
    for (std::size_t k = 0; k <= 2 * m; ++k)
        spot[k] = S * std::exp((static_cast<double>(k) - n) * sigma * std::sqrt(dt));
    std::vector<double> v(m + 1);
    for (std::size_t j = 0; j <= m; ++j) v[j] = std::max(K - spot[2 * j], 0.0);
    for (std::size_t i = m; i-- > 0;)
        for (std::size_t j = 0; j <= i; ++j)  // node (i, j) has spot S u^(2j - i)
            v[j] = std::max(disc * (p * v[j + 1] + (1.0 - p) * v[j]), K - spot[2 * j + m - i]);
    return v[0];
}

void test_finite_difference_american() {
    // S = 36, K = 40, r = 6%, sigma = 20%, T = 1, the first case of Longstaff and Schwartz (2001, Table 1), whose
    // European value 3.844 is published to three decimals. The American value is checked against a binomial tree
    // averaged over 4,000 and 4,001 steps to damp its odd-even oscillation (about 4.48669; the two trees differ by
    // 4e-5), with a tolerance of 2e-4 for the discretisation error of both methods.
    const qe::FdResult am = qe::bs_finite_difference(36, 40, 1.0, 0.06, 0.0, 0.2, false, true, 800, 800);
    const qe::FdResult eu = qe::bs_finite_difference(36, 40, 1.0, 0.06, 0.0, 0.2, false, false, 800, 800);
    CHECK_CLOSE(qe::bs_price(36, 40, 1.0, 0.06, 0.0, 0.2, false), 3.844, 5e-4);
    const double tree = 0.5 * (crr_american_put(36, 40, 1.0, 0.06, 0.2, 4000) +
                               crr_american_put(36, 40, 1.0, 0.06, 0.2, 4001));
    CHECK_CLOSE(am.price, tree, 2e-4);
    CHECK(am.price > eu.price);
    // The American value never falls below the payoff, and the critical price stays below the strike and falls
    // (weakly) as the time to maturity grows.
    for (std::size_t j = 0; j < am.values.size(); ++j)
        CHECK(am.values[j] >= std::max(40.0 - am.spot_grid[j], 0.0) - 1e-12);
    bool below_strike = true, monotone = true;
    double previous = std::numeric_limits<double>::infinity();
    for (double b : am.exercise_boundary) {
        if (!std::isfinite(b)) continue;
        below_strike = below_strike && b <= 40.0;
        monotone = monotone && b <= previous + 1e-12;
        previous = b;
    }
    CHECK(below_strike);
    CHECK(monotone);
    CHECK(am.exercise_boundary.size() == am.tau_grid.size());
    // Without dividends early exercise of a call is never optimal (Merton, 1973).
    const qe::FdResult am_call = qe::bs_finite_difference(100, 100, 1.0, 0.05, 0.0, 0.3, true, true);
    const qe::FdResult eu_call = qe::bs_finite_difference(100, 100, 1.0, 0.05, 0.0, 0.3, true, false);
    CHECK_CLOSE(am_call.price, eu_call.price, 1e-6);
}

// ------------------------------------------------------------------------------------------------ Heston

const qe::HestonParams kFangOosterlee{0.0175, 1.5768, 0.0398, 0.5751, -0.5711};

void test_gauss_legendre() {
    // The 16-point rule integrates polynomials up to degree 31 exactly on [-1, 1].
    const qe::GaussLegendre& gl = qe::gauss_legendre_16();
    double w = 0.0, x30 = 0.0, x31 = 0.0;
    for (std::size_t j = 0; j < gl.nodes.size(); ++j) {
        w += gl.weights[j];
        x30 += gl.weights[j] * std::pow(gl.nodes[j], 30);
        x31 += gl.weights[j] * std::pow(gl.nodes[j], 31);
    }
    CHECK_CLOSE(w, 2.0, 1e-14);
    CHECK_CLOSE(x30, 2.0 / 31.0, 1e-14);
    CHECK_CLOSE(x31, 0.0, 1e-14);
}

void test_heston_characteristic_function() {
    const double T = 1.3, r = 0.02, q = 0.01;
    const std::complex<double> one = qe::heston_cf(std::complex<double>(0.0, 0.0), T, r, q, kFangOosterlee);
    CHECK_CLOSE(one.real(), 1.0, 1e-14);
    CHECK_CLOSE(one.imag(), 0.0, 1e-14);
    // Martingale property: E[S_T / S_0] = exp((r - q) T), i.e. phi(-i) = exp((r - q) T).
    const std::complex<double> growth = qe::heston_cf(std::complex<double>(0.0, -1.0), T, r, q, kFangOosterlee);
    CHECK_CLOSE(growth.real(), std::exp((r - q) * T), 1e-13);
    CHECK_CLOSE(growth.imag(), 0.0, 1e-13);
    // Symmetry of a characteristic function: phi(-u) = conj(phi(u)) for real u.
    const std::complex<double> a = qe::heston_cf(std::complex<double>(3.7, 0.0), T, r, q, kFangOosterlee);
    const std::complex<double> b = qe::heston_cf(std::complex<double>(-3.7, 0.0), T, r, q, kFangOosterlee);
    CHECK_CLOSE(a.real(), b.real(), 1e-14);
    CHECK_CLOSE(a.imag(), -b.imag(), 1e-14);
}

void test_heston_prices() {
    // Fang and Oosterlee (2008): S = K = 100, T = 1, r = q = 0 gives 5.785155450 (ten significant digits, itself a
    // numerical result); the Fourier price agrees to 2e-8.
    CHECK_CLOSE(qe::heston_price(100, 100, 1.0, 0.0, 0.0, kFangOosterlee, true), 5.785155450, 2e-8);
    // With sigma -> 0 and rho = 0 the variance is deterministic, v(t) = theta + (v0 - theta) e^{-kappa t}, and the
    // price is Black-Scholes with the average variance; the first correction is O(sigma^2).
    const double v0 = 0.09, kappa = 1.5, theta = 0.04, T = 1.5;
    const qe::HestonParams nearly_deterministic{v0, kappa, theta, 1e-3, 0.0};
    const double avg_var = theta + (v0 - theta) * (1.0 - std::exp(-kappa * T)) / (kappa * T);
    for (double K : {70.0, 100.0, 140.0})
        CHECK_CLOSE(qe::heston_price(100, K, T, 0.02, 0.0, nearly_deterministic, true),
                    qe::bs_price(100, K, T, 0.02, 0.0, std::sqrt(avg_var), true), 1e-5);
    // Put-call parity by construction, and the no-arbitrage bounds of the call.
    const qe::HestonParams p{0.04, 1.2, 0.06, 0.8, -0.5};
    for (double K : {60.0, 100.0, 150.0}) {
        const double call = qe::heston_price(100, K, 0.7, 0.03, 0.01, p, true);
        const double put = qe::heston_price(100, K, 0.7, 0.03, 0.01, p, false);
        CHECK(call >= std::max(100 * std::exp(-0.01 * 0.7) - K * std::exp(-0.03 * 0.7), 0.0) - 1e-10);
        CHECK(call <= 100 * std::exp(-0.01 * 0.7));
        CHECK(put >= -1e-10);
    }
    CHECK(throws_invalid_argument([] { qe::heston_price(100, 100, 1, 0, 0, {0.04, 1.0, 0.04, 0.0, 0.0}, true); }));
    CHECK(throws_invalid_argument([] { qe::heston_price(100, 100, 1, 0, 0, {0.04, 1.0, 0.04, 0.3, 1.5}, true); }));
}

void test_heston_monte_carlo() {
    const std::vector<double> strikes = {80.0, 100.0, 120.0};
    const qe::HestonMcResult mc = qe::heston_mc_qe(100.0, strikes, 1.0, 0.0, 0.0, kFangOosterlee, 50, 200000, 7, 4);
    for (std::size_t j = 0; j < strikes.size(); ++j) {
        // Four standard errors (sampling) plus 0.01 for the QE discretisation bias at 50 steps per year, which
        // Andersen (2008) reports to be of this order or smaller.
        CHECK_CLOSE(mc.call[j], qe::heston_price(100, strikes[j], 1.0, 0, 0, kFangOosterlee, true),
                    4.0 * mc.call_se[j] + 0.01);
        CHECK_CLOSE(mc.put[j], qe::heston_price(100, strikes[j], 1.0, 0, 0, kFangOosterlee, false),
                    4.0 * mc.put_se[j] + 0.01);
        // Put-call parity holds path by path in expectation: C - P = E[S_T] - K (r = q = 0).
        CHECK_CLOSE(mc.call[j] - mc.put[j], mc.forward - strikes[j], 1e-9);
    }
    CHECK_CLOSE(mc.forward, 100.0, 4.0 * mc.forward_se + 0.01);
    // Each block of paths has its own random stream: the result does not depend on the number of threads.
    const qe::HestonMcResult one = qe::heston_mc_qe(100.0, strikes, 1.0, 0.0, 0.0, kFangOosterlee, 20, 20000, 3, 1);
    const qe::HestonMcResult many = qe::heston_mc_qe(100.0, strikes, 1.0, 0.0, 0.0, kFangOosterlee, 20, 20000, 3, 5);
    CHECK(one.call == many.call && one.put == many.put && one.forward == many.forward);
    CHECK(throws_invalid_argument([&] { qe::heston_mc_qe(100.0, strikes, 1.0, 0, 0, kFangOosterlee, 0, 100, 1); }));
}

// ------------------------------------------------------------------------------------------------ GARCH

void test_log_gamma_and_nelder_mead() {
    for (double x : {0.3, 1.0, 2.5, 7.2, 20.0, 150.5}) CHECK_CLOSE(qe::log_gamma(x), std::lgamma(x), 1e-12);
    // Rosenbrock function from the classical starting point (-1.2, 1).
    auto rosenbrock = [](const std::vector<double>& x) {
        return 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) + (1.0 - x[0]) * (1.0 - x[0]);
    };
    const qe::NelderMeadResult res = qe::nelder_mead(rosenbrock, {-1.2, 1.0}, {0.5, 0.5}, 1e-14, 1e-10, 10000);
    CHECK(res.converged);
    CHECK_CLOSE(res.x[0], 1.0, 1e-5);
    CHECK_CLOSE(res.x[1], 1.0, 1e-5);
}

void test_garch_likelihood() {
    std::vector<double> r(500);
    qe::Xoshiro256 rng(5);
    for (auto& v : r) v = 0.8 * rng.normal() + 0.05;
    const double n = static_cast<double>(r.size());
    // With alpha = gamma = beta = 0 the variance is omega after the first (backcast) observation.
    qe::GarchParams flat;
    flat.mu = 0.05;
    flat.omega = 0.64;
    double backcast = 0.0;
    for (double v : r) backcast += (v - flat.mu) * (v - flat.mu);
    backcast /= n;
    const double log_2pi = std::log(2.0 * qe::kPi);
    double expected = 0.5 * (log_2pi + std::log(backcast) + (r[0] - flat.mu) * (r[0] - flat.mu) / backcast);
    for (std::size_t t = 1; t < r.size(); ++t)
        expected += 0.5 * (log_2pi + std::log(flat.omega) + (r[t] - flat.mu) * (r[t] - flat.mu) / flat.omega);
    const qe::GarchSpec gaussian{true, false};
    std::vector<double> s2(r.size() + 1);
    CHECK_CLOSE(qe::garch_nll(r.data(), r.size(), flat, gaussian, s2.data()), expected, 1e-9);
    CHECK(s2[0] == backcast && s2[r.size()] == flat.omega);
    // A standardised Student-t with many degrees of freedom converges to the normal: the difference in the
    // log-likelihood is O(n / nu).
    qe::GarchParams p;
    p.mu = 0.05;
    p.omega = 0.05;
    p.alpha = 0.05;
    p.gamma = 0.05;
    p.beta = 0.85;
    p.nu = 1e7;
    const double t_nll = qe::garch_nll(r.data(), r.size(), p, qe::GarchSpec{true, true});
    const double n_nll = qe::garch_nll(r.data(), r.size(), p, gaussian);
    CHECK_CLOSE(t_nll, n_nll, 1e-3);
    // GJR recursion written out: the leverage term gamma applies after negative shocks only.
    double s2_manual = backcast, gjr_nll = 0.0;
    for (double v : r) {
        const double e = v - p.mu;
        gjr_nll += 0.5 * (log_2pi + std::log(s2_manual) + e * e / s2_manual);
        s2_manual = p.omega + (e < 0.0 ? p.alpha + p.gamma : p.alpha) * e * e + p.beta * s2_manual;
    }
    CHECK_CLOSE(n_nll, gjr_nll, 1e-9);
    p.nu = 2.0;  // no finite variance: rejected with a large value
    CHECK(qe::garch_nll(r.data(), r.size(), p, qe::GarchSpec{true, true}) >= 1e300);
    CHECK(throws_invalid_argument([&] { qe::garch_nll(r.data(), 1, p, gaussian); }));
    // The reparametrisation is a bijection.
    p.nu = 7.0;
    const qe::GarchSpec gjr_t{true, true};
    const qe::GarchParams back = qe::garch_from_unconstrained(qe::garch_to_unconstrained(p, gjr_t, 0.8), gjr_t, 0.8);
    CHECK_CLOSE(back.mu, p.mu, 1e-14);
    CHECK_CLOSE(back.omega, p.omega, 1e-14);
    CHECK_CLOSE(back.alpha, p.alpha, 1e-14);
    CHECK_CLOSE(back.gamma, p.gamma, 1e-14);
    CHECK_CLOSE(back.beta, p.beta, 1e-14);
    CHECK_CLOSE(back.nu, p.nu, 1e-12);
}

void test_garch_simulation_and_fit() {
    qe::GarchParams truth;
    truth.mu = 0.02;
    truth.omega = 0.02;
    truth.alpha = 0.04;
    truth.gamma = 0.08;
    truth.beta = 0.88;
    truth.nu = 7.0;
    const qe::GarchSpec spec{true, true};
    const std::vector<double> r = qe::garch_simulate(truth, spec, 4000, 2024);
    CHECK(r.size() == 4000);
    const qe::GarchFit fit = qe::garch_fit(r.data(), r.size(), spec);
    // The optimiser must do at least as well as the true parameters (the maximum likelihood estimate is the best
    // fit to this sample), and the estimates are close to the truth for 4,000 observations.
    CHECK(fit.nll <= qe::garch_nll(r.data(), r.size(), truth, spec) + 1e-6);
    CHECK(fit.params.persistence() < 1.0);
    CHECK_CLOSE(fit.params.persistence(), truth.persistence(), 0.03);
    CHECK_CLOSE(fit.params.beta, truth.beta, 0.05);
    CHECK_CLOSE(fit.params.gamma, truth.gamma, 0.05);  // leverage: larger response to negative shocks
    CHECK_CLOSE(fit.params.nu, truth.nu, 3.0);
    qe::GarchParams explosive = truth;
    explosive.beta = 0.99;
    CHECK(throws_invalid_argument([&] { qe::garch_simulate(explosive, spec, 10, 1); }));
}

void test_garch_rolling_forecast() {
    qe::GarchParams truth;
    truth.omega = 0.03;
    truth.alpha = 0.05;
    truth.gamma = 0.05;
    truth.beta = 0.88;
    truth.nu = 6.0;
    const qe::GarchSpec spec{true, true};
    const std::vector<double> r = qe::garch_simulate(truth, spec, 460, 99);
    const std::vector<double> alphas = {0.01, 0.025};
    const qe::RollingGarchResult one = qe::garch_rolling_forecast(r, 400, 20, spec, alphas, 1);
    const qe::RollingGarchResult many = qe::garch_rolling_forecast(r, 400, 20, spec, alphas, 4);
    CHECK(one.n_forecasts == 60 && one.params.size() == 3 * 6);
    // Independent fits written to their own slots: identical results whatever the number of threads.
    CHECK(one.sigma == many.sigma && one.fhs_var == many.fhs_var && one.fhs_es == many.fhs_es && one.nll == many.nll);
    // Expected shortfall is the mean loss beyond the VaR, so ES >= VaR; a smaller tail probability gives a larger VaR.
    bool es_above_var = true, ordered = true;
    for (std::size_t i = 0; i < one.n_forecasts; ++i) {
        for (std::size_t a = 0; a < alphas.size(); ++a)
            es_above_var = es_above_var && one.fhs_es[i * 2 + a] >= one.fhs_var[i * 2 + a] - 1e-12;
        ordered = ordered && one.fhs_var[i * 2] >= one.fhs_var[i * 2 + 1] - 1e-12;
    }
    CHECK(es_above_var);
    CHECK(ordered);
    CHECK(throws_invalid_argument([&] { qe::garch_rolling_forecast(r, 40, 20, spec, alphas); }));
    CHECK(throws_invalid_argument([&] { qe::garch_rolling_forecast(r, 400, 20, spec, {1.5}); }));
    CHECK(throws_invalid_argument([&] { qe::garch_rolling_forecast(r, 460, 20, spec, alphas); }));
}

// ------------------------------------------------------------------------------------------------ delta hedging

qe::HedgeContract atm_call() {
    qe::HedgeContract c;
    c.S0 = 100.0;
    c.K = 100.0;
    c.T = 0.5;
    c.r_dom = 0.03;
    c.r_for = 0.01;
    c.sigma_price = 0.2;
    c.sigma_hedge = 0.2;
    c.n_steps = 125;
    c.rebalance_every = 1;
    return c;
}

void test_hedging_pnl() {
    qe::HedgeDynamics gbm;
    gbm.mu = 0.03 - 0.01;  // risk-neutral drift of the underlying: r_dom - r_for
    gbm.sigma = 0.2;
    const long long n = 40000;
    qe::HedgeContract unhedged = atm_call();
    unhedged.rebalance_every = 0;
    const qe::HedgeResult u = qe::simulate_hedge(unhedged, gbm, n, 17, 4);
    CHECK_CLOSE(u.premium, qe::bs_price(100, 100, 0.5, 0.03, 0.01, 0.2, true), 1e-14);
    // Unhedged, priced at the true volatility and simulated with the risk-neutral drift: zero expected P&L.
    CHECK_CLOSE(mean_of(u.pnl), 0.0, 4.0 * sd_of(u.pnl) / std::sqrt(static_cast<double>(n)));
    // Discrete delta hedging: the hedging error variance is proportional to the rebalancing interval, so weekly
    // (every 5 steps) versus daily rebalancing multiplies the standard deviation by about sqrt(5) = 2.24.
    const qe::HedgeResult daily = qe::simulate_hedge(atm_call(), gbm, n, 17, 4);
    qe::HedgeContract weekly_c = atm_call();
    weekly_c.rebalance_every = 5;
    const qe::HedgeResult weekly = qe::simulate_hedge(weekly_c, gbm, n, 17, 4);
    CHECK(sd_of(daily.pnl) < 0.15 * sd_of(u.pnl));
    const double ratio = sd_of(weekly.pnl) / sd_of(daily.pnl);
    CHECK(ratio > 1.9 && ratio < 2.6);
    CHECK_CLOSE(mean_of(daily.pnl), 0.0, 4.0 * sd_of(daily.pnl) / std::sqrt(static_cast<double>(n)) + 0.01);
    // Costs do not change the simulated paths or the hedge ratios: the P&L falls by exactly the discounted costs.
    qe::HedgeContract costly = atm_call();
    costly.cost_rate = 0.0005;
    const qe::HedgeResult with_costs = qe::simulate_hedge(costly, gbm, 2000, 23, 2);
    const qe::HedgeResult without = qe::simulate_hedge(atm_call(), gbm, 2000, 23, 2);
    bool identity = true;
    for (std::size_t i = 0; i < with_costs.pnl.size(); ++i)
        identity = identity && std::abs(without.pnl[i] - with_costs.pnl[i] - with_costs.costs[i]) < 1e-10 &&
                   without.costs[i] == 0.0 && with_costs.costs[i] > 0.0;
    CHECK(identity);
}

void test_hedging_dynamics_and_threads() {
    qe::HedgeDynamics heston;
    heston.kind = qe::Dynamics::Heston;
    heston.heston = qe::HestonParams{0.04, 2.0, 0.04, 0.5, -0.6};
    qe::HedgeDynamics fhs;
    fhs.kind = qe::Dynamics::GarchFhs;
    fhs.g_omega = 0.01;
    fhs.g_alpha = 0.04;
    fhs.g_gamma = 0.02;
    fhs.g_beta = 0.93;
    fhs.g_var0 = 0.4;
    qe::Xoshiro256 rng(3);
    fhs.residuals.resize(1000);
    for (auto& v : fhs.residuals) v = rng.normal();
    for (const qe::HedgeDynamics& d : {heston, fhs}) {
        const qe::HedgeResult one = qe::simulate_hedge(atm_call(), d, 5000, 41, 1);
        const qe::HedgeResult many = qe::simulate_hedge(atm_call(), d, 5000, 41, 3);
        CHECK(one.pnl == many.pnl && one.final_spot == many.final_spot);
        bool finite = true;
        for (double v : one.pnl) finite = finite && std::isfinite(v);
        CHECK(finite);
    }
    qe::HedgeDynamics empty = fhs;
    empty.residuals.clear();
    CHECK(throws_invalid_argument([&] { qe::simulate_hedge(atm_call(), empty, 10, 1); }));
    qe::HedgeContract bad = atm_call();
    bad.rebalance_every = -1;
    CHECK(throws_invalid_argument([&] { qe::simulate_hedge(bad, qe::HedgeDynamics{}, 10, 1); }));
}

// ------------------------------------------------------------------------------------------------ Nelson-Siegel filter

// Dense Kalman filter of the same model (general n x n innovation covariance), for comparison.
bool cholesky(std::vector<double> a, std::size_t n, std::vector<double>& l) {
    l.assign(n * n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        double s = a[j * n + j];
        for (std::size_t k = 0; k < j; ++k) s -= l[j * n + k] * l[j * n + k];
        if (!(s > 0.0)) return false;
        l[j * n + j] = std::sqrt(s);
        for (std::size_t i = j + 1; i < n; ++i) {
            double t = a[i * n + j];
            for (std::size_t k = 0; k < j; ++k) t -= l[i * n + k] * l[j * n + k];
            l[i * n + j] = t / l[j * n + j];
        }
    }
    return true;
}

std::vector<double> solve_spd(const std::vector<double>& l, std::size_t n, std::vector<double> b) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < i; ++k) b[i] -= l[i * n + k] * b[k];
        b[i] /= l[i * n + i];
    }
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t k = i + 1; k < n; ++k) b[i] -= l[k * n + i] * b[k];
        b[i] /= l[i * n + i];
    }
    return b;
}

double dense_dns_loglik(const std::vector<double>& Y, std::size_t T, std::size_t n, const std::vector<double>& L,
                        const std::vector<double>& mu, const std::vector<double>& A, const std::vector<double>& Q,
                        const std::vector<double>& h, const std::vector<double>& P0, std::vector<double>& filtered) {
    std::vector<double> x = mu, P = P0;
    double loglik = 0.0;
    filtered.assign(T * 3, 0.0);
    for (std::size_t t = 0; t < T; ++t) {
        if (t > 0) {
            std::vector<double> xn(3), AP(9, 0.0), Pn(9, 0.0);
            for (std::size_t i = 0; i < 3; ++i) {
                xn[i] = mu[i];
                for (std::size_t k = 0; k < 3; ++k) xn[i] += A[i * 3 + k] * (x[k] - mu[k]);
            }
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                    for (std::size_t k = 0; k < 3; ++k) AP[i * 3 + j] += A[i * 3 + k] * P[k * 3 + j];
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j) {
                    Pn[i * 3 + j] = Q[i * 3 + j];
                    for (std::size_t k = 0; k < 3; ++k) Pn[i * 3 + j] += AP[i * 3 + k] * A[j * 3 + k];
                }
            x = xn;
            P = Pn;
        }
        std::vector<std::size_t> obs;
        for (std::size_t i = 0; i < n; ++i)
            if (std::isfinite(Y[t * n + i])) obs.push_back(i);
        const std::size_t m = obs.size();
        if (m > 0) {
            // v = y - L x, S = L P L' + H, K = P L' S^{-1}.
            std::vector<double> v(m), PLt(3 * m, 0.0), S(m * m, 0.0);
            for (std::size_t a = 0; a < m; ++a) {
                const std::size_t i = obs[a];
                v[a] = Y[t * n + i];
                for (std::size_t k = 0; k < 3; ++k) v[a] -= L[i * 3 + k] * x[k];
                for (std::size_t r = 0; r < 3; ++r)
                    for (std::size_t k = 0; k < 3; ++k) PLt[r * m + a] += P[r * 3 + k] * L[i * 3 + k];
            }
            for (std::size_t a = 0; a < m; ++a)
                for (std::size_t b = 0; b < m; ++b) {
                    for (std::size_t k = 0; k < 3; ++k) S[a * m + b] += L[obs[a] * 3 + k] * PLt[k * m + b];
                    if (a == b) S[a * m + b] += h[obs[a]] * h[obs[a]];
                }
            std::vector<double> ls;
            if (!cholesky(S, m, ls)) return -std::numeric_limits<double>::infinity();
            const std::vector<double> sv = solve_spd(ls, m, v);
            double logdet = 0.0, quad = 0.0;
            for (std::size_t a = 0; a < m; ++a) {
                logdet += 2.0 * std::log(ls[a * m + a]);
                quad += v[a] * sv[a];
            }
            loglik -= 0.5 * (static_cast<double>(m) * std::log(2.0 * qe::kPi) + logdet + quad);
            // x += P L' S^{-1} v; P -= P L' S^{-1} L P.
            for (std::size_t r = 0; r < 3; ++r)
                for (std::size_t a = 0; a < m; ++a) x[r] += PLt[r * m + a] * sv[a];
            std::vector<double> Pn = P;
            for (std::size_t c = 0; c < 3; ++c) {
                std::vector<double> col(m);
                for (std::size_t a = 0; a < m; ++a) col[a] = PLt[c * m + a];
                const std::vector<double> scol = solve_spd(ls, m, col);
                for (std::size_t r = 0; r < 3; ++r)
                    for (std::size_t a = 0; a < m; ++a) Pn[r * 3 + c] -= PLt[r * m + a] * scol[a];
            }
            P = Pn;
        }
        for (std::size_t i = 0; i < 3; ++i) filtered[t * 3 + i] = x[i];
    }
    return loglik;
}

void test_nelson_siegel_filter() {
    const std::vector<double> maturities = {0.25, 1.0, 3.0, 7.0, 15.0};
    const double lam = 0.6;
    const std::vector<double> L = qe::ns_loadings(maturities, lam);
    CHECK(L[0] == 1.0);
    const std::vector<double> limits = qe::ns_loadings({1.0 / 12.0, 1000.0}, lam);
    // One-month maturity: (1 - e^{-x}) / x loses about eps / x = 4e-15 to cancellation at x = 0.05, which is
    // negligible; expm1 gives the reference value.
    const double x0 = lam / 12.0;
    CHECK_CLOSE(limits[1], -std::expm1(-x0) / x0, 1e-13);
    CHECK_CLOSE(limits[2], -std::expm1(-x0) / x0 - std::exp(-x0), 1e-13);
    CHECK_CLOSE(limits[4], 1.0 / (lam * 1000.0), 1e-12);
    CHECK_CLOSE(limits[5], 1.0 / (lam * 1000.0), 1e-12);

    const std::size_t T = 40, n = maturities.size();
    const std::vector<double> mu = {4.0, -1.0, 0.5};
    const std::vector<double> A = {0.95, 0.02, 0.0, 0.01, 0.9, 0.05, 0.0, 0.03, 0.8};
    const std::vector<double> Q = {0.04, 0.01, 0.0, 0.01, 0.09, 0.02, 0.0, 0.02, 0.25};
    const std::vector<double> h = {0.08, 0.03, 0.02, 0.03, 0.06};
    const std::vector<double> P0 = {1.0, 0.1, 0.0, 0.1, 1.5, 0.2, 0.0, 0.2, 2.0};
    // Simulate yields from the model, with a few missing observations and one empty date.
    qe::Xoshiro256 rng(8);
    std::vector<double> f = mu, Y(T * n);
    for (std::size_t t = 0; t < T; ++t) {
        std::vector<double> next(3);
        for (std::size_t i = 0; i < 3; ++i) {
            next[i] = mu[i] + 0.2 * rng.normal();
            for (std::size_t k = 0; k < 3; ++k) next[i] += A[i * 3 + k] * (f[k] - mu[k]);
        }
        f = next;
        for (std::size_t i = 0; i < n; ++i)
            Y[t * n + i] = L[i * 3] * f[0] + L[i * 3 + 1] * f[1] + L[i * 3 + 2] * f[2] + h[i] * rng.normal();
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Y[3 * n + 1] = nan;
    Y[10 * n + 4] = nan;
    for (std::size_t i = 0; i < n; ++i) Y[20 * n + i] = nan;

    const qe::DnsKalmanResult fast = qe::dns_kalman_filter(Y, T, n, maturities, lam, mu, A, Q, h, P0, true);
    std::vector<double> filtered;
    const double dense = dense_dns_loglik(Y, T, n, L, mu, A, Q, h, P0, filtered);
    // The Woodbury form and the dense form are the same filter: agreement to rounding error.
    CHECK_CLOSE(fast.loglik, dense, 1e-9 * std::abs(dense));
    double max_diff = 0.0;
    for (std::size_t i = 0; i < filtered.size(); ++i) max_diff = std::max(max_diff, std::abs(fast.filtered[i] - filtered[i]));
    CHECK(max_diff < 1e-10);
    // On a date without observations the filtered state is the prediction.
    for (std::size_t i = 0; i < 3; ++i) CHECK(fast.filtered[20 * 3 + i] == fast.predicted[20 * 3 + i]);
    // Without keep_states only the likelihood is returned, unchanged.
    const qe::DnsKalmanResult lean = qe::dns_kalman_filter(Y, T, n, maturities, lam, mu, A, Q, h, P0, false);
    CHECK(lean.loglik == fast.loglik && lean.filtered.empty());
    // A prior covariance that is not positive definite makes the likelihood -inf.
    const std::vector<double> bad_P0 = {-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    CHECK(std::isinf(qe::dns_kalman_filter(Y, T, n, maturities, lam, mu, A, Q, h, bad_P0, false).loglik));
    CHECK(throws_invalid_argument([&] { qe::dns_kalman_filter(Y, T, n - 1, maturities, lam, mu, A, Q, h, P0, false); }));
    std::vector<double> zero_h = h;
    zero_h[2] = 0.0;
    CHECK(throws_invalid_argument([&] { qe::dns_kalman_filter(Y, T, n, maturities, lam, mu, A, Q, zero_h, P0, false); }));
}

}  // namespace

int main() {
    test_splitmix64_reference();
    test_xoshiro_streams();
    test_uniform_and_normal_moments();
    test_gamma_variates();
    test_parallel_for();
    test_black_scholes_reference_values();
    test_black_scholes_identities();
    test_implied_volatility();
    test_tridiagonal_solver();
    test_finite_difference_european();
    test_finite_difference_american();
    test_gauss_legendre();
    test_heston_characteristic_function();
    test_heston_prices();
    test_heston_monte_carlo();
    test_log_gamma_and_nelder_mead();
    test_garch_likelihood();
    test_garch_simulation_and_fit();
    test_garch_rolling_forecast();
    test_hedging_pnl();
    test_hedging_dynamics_and_threads();
    test_nelson_siegel_filter();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
