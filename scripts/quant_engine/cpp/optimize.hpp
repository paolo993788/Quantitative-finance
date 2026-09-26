// Derivative-free Nelder-Mead simplex minimiser (Nelder and Mead, 1965) with
// the standard coefficients discussed by Lagarias, Reeds, Wright and Wright
// (1998). Used for the many independent GARCH fits of the rolling backtest,
// where a self-contained C++ optimiser avoids calling back into Python.
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace qe {

struct NelderMeadResult {
    std::vector<double> x;
    double fun = 0.0;
    int iterations = 0;
    int evaluations = 0;
    bool converged = false;
};

template <class Objective>
NelderMeadResult nelder_mead(Objective&& f, std::vector<double> x0, const std::vector<double>& step,
                             double ftol = 1e-10, double xtol = 1e-8, int max_iter = 5000) {
    const double rho = 1.0, chi = 2.0, gamma = 0.5, shrink = 0.5;
    const std::size_t n = x0.size();
    std::vector<std::vector<double>> simplex(n + 1, x0);
    std::vector<double> values(n + 1);
    NelderMeadResult res;
    auto eval = [&](const std::vector<double>& x) {
        ++res.evaluations;
        const double v = f(x);
        return std::isfinite(v) ? v : 1e300;
    };
    for (std::size_t i = 0; i < n; ++i) simplex[i + 1][i] += step[i];
    for (std::size_t i = 0; i <= n; ++i) values[i] = eval(simplex[i]);

    std::vector<std::size_t> order(n + 1);
    std::vector<double> centroid(n), xr(n), xe(n), xc(n);
    for (res.iterations = 0; res.iterations < max_iter; ++res.iterations) {
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return values[a] < values[b]; });
        const std::size_t best = order.front(), worst = order.back(), second = order[n - 1];

        double f_spread = std::abs(values[worst] - values[best]);
        double x_spread = 0.0;
        for (std::size_t i = 0; i <= n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                x_spread = std::max(x_spread, std::abs(simplex[i][j] - simplex[best][j]));
        if (f_spread <= ftol * (1.0 + std::abs(values[best])) && x_spread <= xtol) {
            res.converged = true;
            break;
        }

        std::fill(centroid.begin(), centroid.end(), 0.0);
        for (std::size_t i = 0; i <= n; ++i) {
            if (i == worst) continue;
            for (std::size_t j = 0; j < n; ++j) centroid[j] += simplex[i][j] / n;
        }
        for (std::size_t j = 0; j < n; ++j) xr[j] = centroid[j] + rho * (centroid[j] - simplex[worst][j]);
        const double fr = eval(xr);
        if (fr < values[best]) {
            for (std::size_t j = 0; j < n; ++j) xe[j] = centroid[j] + chi * (xr[j] - centroid[j]);
            const double fe = eval(xe);
            if (fe < fr) { simplex[worst] = xe; values[worst] = fe; }
            else { simplex[worst] = xr; values[worst] = fr; }
            continue;
        }
        if (fr < values[second]) {
            simplex[worst] = xr;
            values[worst] = fr;
            continue;
        }
        bool accepted = false;
        if (fr < values[worst]) {  // outside contraction
            for (std::size_t j = 0; j < n; ++j) xc[j] = centroid[j] + gamma * (xr[j] - centroid[j]);
            const double fc = eval(xc);
            if (fc <= fr) { simplex[worst] = xc; values[worst] = fc; accepted = true; }
        } else {  // inside contraction
            for (std::size_t j = 0; j < n; ++j) xc[j] = centroid[j] + gamma * (simplex[worst][j] - centroid[j]);
            const double fc = eval(xc);
            if (fc < values[worst]) { simplex[worst] = xc; values[worst] = fc; accepted = true; }
        }
        if (!accepted) {
            for (std::size_t i = 0; i <= n; ++i) {
                if (i == best) continue;
                for (std::size_t j = 0; j < n; ++j)
                    simplex[i][j] = simplex[best][j] + shrink * (simplex[i][j] - simplex[best][j]);
                values[i] = eval(simplex[i]);
            }
        }
    }
    const std::size_t best =
        static_cast<std::size_t>(std::min_element(values.begin(), values.end()) - values.begin());
    res.x = simplex[best];
    res.fun = values[best];
    return res;
}

}  // namespace qe
