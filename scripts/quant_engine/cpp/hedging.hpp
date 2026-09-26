// Monte Carlo simulation of the profit and loss of a dealer who sells a
// European option and delta-hedges it in discrete time.
//
// Conventions (Garman-Kohlhagen for currencies, Black-Scholes-Merton for
// assets with a yield): S is the price of one unit of the underlying in
// domestic currency, r_dom the domestic and r_for the foreign rate (or
// dividend yield), both continuously compounded.
//
// Trading rules for one short option:
//   t = 0: receive the premium priced at vol `sigma_price`, buy the
//          Black-Scholes delta computed at vol `sigma_hedge`;
//   every `rebalance_every` simulation steps: rebalance to the new delta;
//   the cash account accrues at r_dom and the underlying position accrues
//   the foreign rate (units grow by exp(r_for dt));
//   every trade in the underlying pays `cost_rate` times its value;
//   at maturity: pay the option payoff and liquidate the hedge.
// rebalance_every = 0 means no hedge at all. The reported P&L is discounted
// to time 0 at r_dom.
//
// Real-world dynamics of the underlying (one simulation step = T / n_steps):
//   GBM:       d ln S = (mu - sigma^2 / 2) dt + sigma dW;
//   Heston:    Andersen (2008) QE scheme with drift mu;
//   GARCH-FHS: daily log returns r = mu_d + sigma_t z with z resampled from
//              standardised residuals and GJR-GARCH(1,1) variance, inputs
//              in percent as produced by the garch module (filtered
//              historical simulation, Barone-Adesi et al., 1999).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "black_scholes.hpp"
#include "heston.hpp"
#include "parallel.hpp"
#include "random.hpp"

namespace qe {

enum class Dynamics { GBM, Heston, GarchFhs };

struct HedgeContract {
    double S0 = 1.0, K = 1.0, T = 1.0;
    double r_dom = 0.0, r_for = 0.0;
    bool is_call = true;
    double sigma_price = 0.1;  // volatility used to price the option sold
    double sigma_hedge = 0.1;  // volatility used to compute the hedge ratio
    int n_steps = 252;
    int rebalance_every = 1;   // 0 = unhedged
    double cost_rate = 0.0;    // proportional cost per unit of value traded
};

struct HedgeDynamics {
    Dynamics kind = Dynamics::GBM;
    double mu = 0.0;                 // real-world drift (GBM, Heston), per year
    double sigma = 0.1;              // GBM volatility
    HestonParams heston{0.01, 1.0, 0.01, 0.1, 0.0};
    // GARCH-FHS (percent units, daily): mean, omega, alpha, gamma, beta, initial variance.
    double g_mu = 0.0, g_omega = 0.0, g_alpha = 0.0, g_gamma = 0.0, g_beta = 0.0, g_var0 = 0.0;
    std::vector<double> residuals;   // standardised residuals to resample
};

struct HedgeResult {
    std::vector<double> pnl;          // discounted P&L of the short option position
    std::vector<double> costs;        // discounted transaction costs (included in pnl)
    std::vector<double> final_spot;
    double premium = 0.0;             // option value at sigma_price
};

inline double bs_delta(double S, double K, double tau, double r, double q, double sigma, bool is_call) {
    if (tau <= 0.0) return is_call ? (S > K ? 1.0 : 0.0) : (S < K ? -1.0 : 0.0);
    return bs_price_greeks(S, K, tau, r, q, sigma, is_call).delta;
}

inline HedgeResult simulate_hedge(const HedgeContract& c, const HedgeDynamics& d, long long n_paths, std::uint64_t seed,
                                  int n_threads = 0) {
    if (c.n_steps < 1 || n_paths < 1 || c.T <= 0.0 || c.S0 <= 0.0 || c.K <= 0.0 || c.rebalance_every < 0 || c.cost_rate < 0.0)
        throw std::invalid_argument("invalid hedging contract");
    if (d.kind == Dynamics::Heston) d.heston.validate();
    if (d.kind == Dynamics::GarchFhs && (d.residuals.empty() || !(d.g_var0 > 0.0)))
        throw std::invalid_argument("GARCH-FHS needs residuals and a positive initial variance");

    const double dt = c.T / c.n_steps;
    const double grow_for = std::exp(c.r_for * dt), grow_dom = std::exp(c.r_dom * dt);
    HedgeResult res;
    res.premium = bs_price(c.S0, c.K, c.T, c.r_dom, c.r_for, c.sigma_price, c.is_call);
    const std::size_t n = static_cast<std::size_t>(n_paths);
    res.pnl.assign(n, 0.0);
    res.costs.assign(n, 0.0);
    res.final_spot.assign(n, 0.0);

    // Heston QE constants (real-world drift mu).
    const HestonParams& h = d.heston;
    const double E = std::exp(-h.kappa * dt), s2 = h.sigma * h.sigma;
    const double K0 = -h.rho * h.kappa * h.theta * dt / h.sigma;
    const double K1 = 0.5 * dt * (h.kappa * h.rho / h.sigma - 0.5) - h.rho / h.sigma;
    const double K2 = 0.5 * dt * (h.kappa * h.rho / h.sigma - 0.5) + h.rho / h.sigma;
    const double K3 = 0.5 * dt * (1.0 - h.rho * h.rho), K4 = K3;

    const long long block = 1024;
    const std::size_t n_blocks = static_cast<std::size_t>((n_paths + block - 1) / block);
    parallel_for(n_blocks, n_threads, [&](std::size_t b) {
        Xoshiro256 rng(seed, b);
        const long long start = static_cast<long long>(b) * block, stop = std::min(n_paths, start + block);
        for (long long p = start; p < stop; ++p) {
            double S = c.S0;
            double v = h.v0;           // Heston variance
            double g_var = d.g_var0;   // GARCH variance (percent^2)
            double units = c.rebalance_every > 0 ? bs_delta(S, c.K, c.T, c.r_dom, c.r_for, c.sigma_hedge, c.is_call) : 0.0;
            double cost = c.cost_rate * std::abs(units) * S;
            double cash = res.premium - units * S - cost;
            double total_cost = cost;
            for (int step = 1; step <= c.n_steps; ++step) {
                // Advance the underlying one step.
                double log_ret;
                if (d.kind == Dynamics::GBM) {
                    log_ret = (d.mu - 0.5 * d.sigma * d.sigma) * dt + d.sigma * std::sqrt(dt) * rng.normal();
                } else if (d.kind == Dynamics::Heston) {
                    const double m = h.theta + (v - h.theta) * E;
                    const double s_sq = v * s2 * E * (1.0 - E) / h.kappa + h.theta * s2 * (1.0 - E) * (1.0 - E) / (2.0 * h.kappa);
                    const double psi = s_sq / (m * m);
                    double v_next;
                    if (psi <= 1.5) {
                        const double inv = 2.0 / psi;
                        const double b2 = inv - 1.0 + std::sqrt(inv) * std::sqrt(inv - 1.0);
                        const double z = std::sqrt(b2) + rng.normal();
                        v_next = m / (1.0 + b2) * z * z;
                    } else {
                        const double prob = (psi - 1.0) / (psi + 1.0);
                        const double u = rng.uniform();
                        v_next = u <= prob ? 0.0 : std::log((1.0 - prob) / (1.0 - u)) * m / (1.0 - prob);
                    }
                    log_ret = d.mu * dt + K0 + K1 * v + K2 * v_next +
                              std::sqrt(std::max(K3 * v + K4 * v_next, 0.0)) * rng.normal();
                    v = v_next;
                } else {
                    const std::size_t idx = std::min(static_cast<std::size_t>(rng.uniform() * d.residuals.size()),
                                                     d.residuals.size() - 1);
                    const double e = std::sqrt(g_var) * d.residuals[idx];  // percent
                    log_ret = (d.g_mu + e) / 100.0;
                    g_var = d.g_omega + (d.g_alpha + (e < 0.0 ? d.g_gamma : 0.0)) * e * e + d.g_beta * g_var;
                }
                S *= std::exp(log_ret);
                cash *= grow_dom;
                units *= grow_for;
                if (step < c.n_steps && c.rebalance_every > 0 && step % c.rebalance_every == 0) {
                    const double target = bs_delta(S, c.K, c.T - step * dt, c.r_dom, c.r_for, c.sigma_hedge, c.is_call);
                    const double trade = target - units;
                    const double trade_cost = c.cost_rate * std::abs(trade) * S;
                    cash -= trade * S + trade_cost;
                    total_cost += trade_cost * std::exp(-c.r_dom * step * dt);
                    units = target;
                }
            }
            const double payoff = c.is_call ? std::max(S - c.K, 0.0) : std::max(c.K - S, 0.0);
            const double liquidation = c.cost_rate * std::abs(units) * S;
            cash += units * S - liquidation - payoff;
            total_cost += liquidation * std::exp(-c.r_dom * c.T);
            res.pnl[p] = cash * std::exp(-c.r_dom * c.T);
            res.costs[p] = total_cost;
            res.final_spot[p] = S;
        }
    });
    return res;
}

}  // namespace qe
