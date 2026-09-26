// Crank-Nicolson finite-difference solver for European and American options
// under Black-Scholes dynamics, written in the log-price x = ln S:
//   dV/dtau = 0.5 sigma^2 V_xx + (r - q - 0.5 sigma^2) V_x - r V,
// where tau is the time to maturity. The first steps use Rannacher smoothing
// (implicit Euler half steps) to damp the oscillations caused by the
// non-smooth payoff. American exercise is handled by projected successive
// over-relaxation (PSOR), which solves the linear complementarity problem at
// every time step.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qe {

struct FdResult {
    double price = 0.0;
    double delta = 0.0;
    double gamma = 0.0;
    std::vector<double> spot_grid;          // S values of the space grid
    std::vector<double> values;             // option values at tau = T on the grid
    std::vector<double> tau_grid;           // time to maturity after each step
    std::vector<double> exercise_boundary;  // critical spot price (American only)
    long long psor_iterations = 0;
};

inline void solve_tridiagonal(double lower, double diag, double upper, std::vector<double>& rhs) {
    // Thomas algorithm for a constant-coefficient tridiagonal system.
    const std::size_t n = rhs.size();
    std::vector<double> c_prime(n, 0.0);
    double denom = diag;
    c_prime[0] = upper / denom;
    rhs[0] /= denom;
    for (std::size_t i = 1; i < n; ++i) {
        denom = diag - lower * c_prime[i - 1];
        c_prime[i] = upper / denom;
        rhs[i] = (rhs[i] - lower * rhs[i - 1]) / denom;
    }
    for (std::size_t i = n - 1; i-- > 0;) rhs[i] -= c_prime[i] * rhs[i + 1];
}

inline FdResult bs_finite_difference(double S0, double K, double T, double r, double q, double sigma,
                                     bool is_call, bool american, int n_space = 400, int n_time = 400,
                                     double n_sd = 5.0, int rannacher_steps = 2, double omega = 1.2,
                                     double psor_tol = 1e-12) {
    if (S0 <= 0.0 || K <= 0.0 || T <= 0.0 || sigma <= 0.0) {
        throw std::invalid_argument("S0, K, T and sigma must be positive");
    }
    if (n_space < 10 || n_time < 1) throw std::invalid_argument("grid too small");
    if (n_space % 2 != 0) ++n_space;  // keep S0 on a grid node
    const int M = n_space;
    const double x0 = std::log(S0);
    const double half_width = n_sd * sigma * std::sqrt(T) + std::abs(std::log(K / S0));
    const double dx = 2.0 * half_width / M;

    std::vector<double> S(M + 1), payoff(M + 1), V(M + 1);
    for (int j = 0; j <= M; ++j) {
        S[j] = std::exp(x0 + (j - M / 2) * dx);
        payoff[j] = is_call ? std::max(S[j] - K, 0.0) : std::max(K - S[j], 0.0);
        V[j] = payoff[j];
    }

    const double mu = r - q - 0.5 * sigma * sigma;
    const double a = 0.5 * sigma * sigma / (dx * dx) - mu / (2.0 * dx);
    const double b = -sigma * sigma / (dx * dx) - r;
    const double c = 0.5 * sigma * sigma / (dx * dx) + mu / (2.0 * dx);

    // Time-stepping schedule: Rannacher start, then Crank-Nicolson.
    const double dt = T / n_time;
    std::vector<std::pair<double, double>> schedule;  // (theta, step)
    const int n_smooth = std::min(rannacher_steps, n_time);
    for (int k = 0; k < 2 * n_smooth; ++k) schedule.emplace_back(1.0, 0.5 * dt);
    for (int k = n_smooth; k < n_time; ++k) schedule.emplace_back(0.5, dt);

    FdResult res;
    std::vector<double> rhs(M - 1);
    double tau = 0.0;
    for (const auto& [theta, step] : schedule) {
        tau += step;
        // Boundary values at the new time level.
        double left, right;
        if (is_call) {
            left = 0.0;
            right = american ? S[M] - K : S[M] * std::exp(-q * tau) - K * std::exp(-r * tau);
        } else {
            left = american ? K - S[0] : K * std::exp(-r * tau) - S[0] * std::exp(-q * tau);
            right = 0.0;
        }
        const double explicit_w = (1.0 - theta) * step;
        for (int j = 1; j < M; ++j) {
            rhs[j - 1] = V[j] + explicit_w * (a * V[j - 1] + b * V[j] + c * V[j + 1]);
        }
        const double lo = -theta * step * a;
        const double di = 1.0 - theta * step * b;
        const double up = -theta * step * c;
        V[0] = left;
        V[M] = right;
        if (!american) {
            rhs[0] -= lo * left;
            rhs[M - 2] -= up * right;
            solve_tridiagonal(lo, di, up, rhs);
            for (int j = 1; j < M; ++j) V[j] = rhs[j - 1];
        } else {
            // PSOR, warm-started from the previous time level.
            for (int iter = 0; iter < 100000; ++iter) {
                double max_change = 0.0;
                for (int j = 1; j < M; ++j) {
                    const double gs = (rhs[j - 1] - lo * V[j - 1] - up * V[j + 1]) / di;
                    const double updated = std::max(payoff[j], V[j] + omega * (gs - V[j]));
                    max_change = std::max(max_change, std::abs(updated - V[j]));
                    V[j] = updated;
                }
                ++res.psor_iterations;
                if (max_change < psor_tol * K) break;
            }
            // Critical price: edge of the region where V equals the payoff.
            double boundary = std::numeric_limits<double>::quiet_NaN();
            const double tol = 1e-9 * K;
            if (is_call) {
                for (int j = M; j >= 0; --j) {
                    if (payoff[j] > 0.0 && V[j] - payoff[j] < tol) boundary = S[j]; else if (payoff[j] > 0.0) break;
                }
            } else {
                for (int j = 0; j <= M; ++j) {
                    if (payoff[j] > 0.0 && V[j] - payoff[j] < tol) boundary = S[j]; else if (payoff[j] > 0.0) break;
                }
            }
            res.tau_grid.push_back(tau);
            res.exercise_boundary.push_back(boundary);
        }
    }

    const int mid = M / 2;
    const double v_x = (V[mid + 1] - V[mid - 1]) / (2.0 * dx);
    const double v_xx = (V[mid + 1] - 2.0 * V[mid] + V[mid - 1]) / (dx * dx);
    res.price = V[mid];
    res.delta = v_x / S0;
    res.gamma = (v_xx - v_x) / (S0 * S0);
    res.spot_grid = S;
    res.values = V;
    return res;
}

}  // namespace qe
