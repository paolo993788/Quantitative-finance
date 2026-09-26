// GARCH(1,1) and GJR-GARCH(1,1) (Glosten, Jagannathan and Runkle, 1993)
// with Gaussian or standardised Student-t innovations:
//   r_t = mu + e_t,  e_t = sigma_t z_t,
//   sigma_t^2 = omega + (alpha + gamma 1{e_{t-1} < 0}) e_{t-1}^2 + beta sigma_{t-1}^2.
// The recursion is initialised with the sample variance of e_t (backcast).
// Parameters are estimated by maximum likelihood with Nelder-Mead on an
// unconstrained reparametrisation that enforces omega > 0, alpha > 0,
// beta > 0 and alpha + gamma > 0. The leverage coefficient gamma may be
// negative ("inverse leverage", common for exchange rates). Covariance
// stationarity (alpha + gamma / 2 + beta < 1) is imposed through a penalty.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "optimize.hpp"
#include "parallel.hpp"
#include "random.hpp"

namespace qe {

struct GarchSpec {
    bool gjr = true;        // include the leverage term gamma
    bool student_t = true;  // standardised Student-t instead of Gaussian innovations
};

struct GarchParams {
    double mu = 0.0;
    double omega = 0.0;
    double alpha = 0.0;
    double gamma = 0.0;
    double beta = 0.0;
    double nu = 0.0;  // degrees of freedom (Student-t only)

    double persistence() const { return alpha + 0.5 * gamma + beta; }
};

// ln Gamma(x) for x > 0: recurrence up to x >= 15, then the Stirling series.
// Implemented here because std::lgamma may write the global `signgam` and is
// therefore not guaranteed to be thread safe.
inline double log_gamma(double x) {
    double shift = 0.0;
    while (x < 15.0) {
        shift -= std::log(x);
        x += 1.0;
    }
    const double x2 = 1.0 / (x * x);
    const double series = (1.0 / x) * (1.0 / 12.0 - x2 * (1.0 / 360.0 - x2 * (1.0 / 1260.0 - x2 / 1680.0)));
    return (x - 0.5) * std::log(x) - x + 0.91893853320467274178 + series + shift;
}

// Negative log-likelihood. If sigma2_out is not null it receives n + 1
// conditional variances; the last one is the one-step-ahead forecast.
inline double garch_nll(const double* r, std::size_t n, const GarchParams& p, const GarchSpec& spec,
                        double* sigma2_out = nullptr) {
    if (n < 2) throw std::invalid_argument("need at least two observations");
    double backcast = 0.0;
    for (std::size_t t = 0; t < n; ++t) backcast += (r[t] - p.mu) * (r[t] - p.mu);
    backcast /= static_cast<double>(n);
    const double gamma = spec.gjr ? p.gamma : 0.0;
    double t_const = 0.0;
    if (spec.student_t) {
        if (!(p.nu > 2.0)) return 1e300;
        t_const = log_gamma(0.5 * (p.nu + 1.0)) - log_gamma(0.5 * p.nu) - 0.5 * std::log(3.14159265358979323846 * (p.nu - 2.0));
    }
    const double log_2pi = 1.83787706640934548356;
    double s2 = backcast;
    double nll = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
        if (!(s2 > 0.0) || !std::isfinite(s2)) return 1e300;
        if (sigma2_out) sigma2_out[t] = s2;
        const double e = r[t] - p.mu;
        if (spec.student_t) {
            nll -= t_const - 0.5 * std::log(s2) - 0.5 * (p.nu + 1.0) * std::log1p(e * e / ((p.nu - 2.0) * s2));
        } else {
            nll += 0.5 * (log_2pi + std::log(s2) + e * e / s2);
        }
        s2 = p.omega + (p.alpha + (e < 0.0 ? gamma : 0.0)) * e * e + p.beta * s2;
    }
    if (sigma2_out) sigma2_out[n] = s2;
    return nll;
}

// Unconstrained coordinates: mu / scale, ln(omega / scale^2), ln alpha,
// ln beta, [ln(alpha + gamma)], [ln(nu - 2)].
inline GarchParams garch_from_unconstrained(const std::vector<double>& z, const GarchSpec& spec, double scale) {
    auto safe_exp = [](double v) { return std::exp(std::clamp(v, -50.0, 50.0)); };
    GarchParams p;
    std::size_t k = 0;
    p.mu = z[k++] * scale;
    p.omega = safe_exp(z[k++]) * scale * scale;
    p.alpha = safe_exp(z[k++]);
    p.beta = safe_exp(z[k++]);
    p.gamma = spec.gjr ? safe_exp(z[k++]) - p.alpha : 0.0;
    p.nu = spec.student_t ? 2.0 + std::exp(std::clamp(z[k++], -10.0, std::log(998.0))) : 0.0;
    return p;
}

inline std::vector<double> garch_to_unconstrained(const GarchParams& p, const GarchSpec& spec, double scale) {
    std::vector<double> z{p.mu / scale, std::log(p.omega / (scale * scale)), std::log(p.alpha), std::log(p.beta)};
    if (spec.gjr) z.push_back(std::log(p.alpha + p.gamma));
    if (spec.student_t) z.push_back(std::log(p.nu - 2.0));
    return z;
}

struct GarchFit {
    GarchParams params;
    double nll = 0.0;
    int evaluations = 0;
    bool converged = false;
};

inline GarchFit garch_fit(const double* r, std::size_t n, const GarchSpec& spec) {
    double mean = 0.0;
    for (std::size_t t = 0; t < n; ++t) mean += r[t];
    mean /= static_cast<double>(n);
    double var = 0.0;
    for (std::size_t t = 0; t < n; ++t) var += (r[t] - mean) * (r[t] - mean);
    var /= static_cast<double>(n);
    if (!(var > 0.0)) throw std::invalid_argument("returns have zero variance");
    const double scale = std::sqrt(var);

    GarchParams start;
    start.mu = mean;
    start.alpha = spec.gjr ? 0.03 : 0.06;
    start.gamma = spec.gjr ? 0.06 : 0.0;
    start.beta = 0.90;
    start.omega = var * (1.0 - start.persistence());
    start.nu = 8.0;

    auto objective = [&](const std::vector<double>& z) {
        const GarchParams p = garch_from_unconstrained(z, spec, scale);
        const double pers = p.persistence();
        if (pers >= 1.0 - 1e-6) return 1e10 * (1.0 + pers);
        return garch_nll(r, n, p, spec);
    };
    std::vector<double> z = garch_to_unconstrained(start, spec, scale);
    std::vector<double> step(z.size(), 0.5);
    step[0] = 0.05;

    GarchFit fit;
    NelderMeadResult best = nelder_mead(objective, z, step, 1e-12, 1e-7, 4000);
    fit.evaluations = best.evaluations;
    // Restart from the best vertex until the objective stops improving.
    for (int restart = 0; restart < 3; ++restart) {
        NelderMeadResult again = nelder_mead(objective, best.x, step, 1e-12, 1e-7, 4000);
        fit.evaluations += again.evaluations;
        const bool improved = again.fun < best.fun - 1e-9;
        if (again.fun <= best.fun) best = again;
        if (!improved) break;
    }
    fit.params = garch_from_unconstrained(best.x, spec, scale);
    fit.nll = best.fun;
    fit.converged = best.converged;
    return fit;
}

struct RollingGarchResult {
    std::size_t n_forecasts = 0;
    std::size_t n_alphas = 0;
    std::vector<double> mu, sigma, nu;   // one-step-ahead forecasts
    std::vector<double> fhs_var, fhs_es; // filtered historical simulation, row-major (forecast, alpha)
    std::vector<double> params;          // (fit, [mu, omega, alpha, gamma, beta, nu])
    std::vector<double> nll;
    std::vector<int> converged;
};

// Rolling one-step-ahead forecasts. Forecast i refers to observation
// window + i and uses parameters estimated on the `window` observations
// ending at the most recent refit date (refits every `refit_every` days).
// The fits are independent and run in parallel.
inline RollingGarchResult garch_rolling_forecast(const std::vector<double>& r, std::size_t window,
                                                 std::size_t refit_every, const GarchSpec& spec,
                                                 const std::vector<double>& alphas, int n_threads = 0) {
    if (window < 50 || refit_every < 1) throw std::invalid_argument("window must be >= 50 and refit_every >= 1");
    if (r.size() <= window) throw std::invalid_argument("not enough observations for the chosen window");
    for (double a : alphas)
        if (!(a > 0.0 && a < 1.0)) throw std::invalid_argument("tail probabilities must lie in (0, 1)");
    RollingGarchResult res;
    res.n_forecasts = r.size() - window;
    res.n_alphas = alphas.size();
    const std::size_t n_fits = (res.n_forecasts + refit_every - 1) / refit_every;
    std::vector<GarchFit> fits(n_fits);
    parallel_for(n_fits, n_threads, [&](std::size_t k) {
        fits[k] = garch_fit(r.data() + k * refit_every, window, spec);
    });

    res.mu.resize(res.n_forecasts);
    res.sigma.resize(res.n_forecasts);
    res.nu.resize(res.n_forecasts);
    res.fhs_var.resize(res.n_forecasts * res.n_alphas);
    res.fhs_es.resize(res.n_forecasts * res.n_alphas);
    parallel_for(res.n_forecasts, n_threads, [&](std::size_t i) {
        const GarchParams& p = fits[i / refit_every].params;
        const double* start = r.data() + i;
        std::vector<double> s2(window + 1);
        garch_nll(start, window, p, spec, s2.data());
        std::vector<double> z(window);
        for (std::size_t j = 0; j < window; ++j) z[j] = (start[j] - p.mu) / std::sqrt(s2[j]);
        std::sort(z.begin(), z.end());
        const double sig = std::sqrt(s2[window]);
        res.mu[i] = p.mu;
        res.sigma[i] = sig;
        res.nu[i] = p.nu;
        for (std::size_t a = 0; a < res.n_alphas; ++a) {
            // Empirical quantile with linear interpolation (numpy's default).
            const double h = (window - 1) * alphas[a];
            const std::size_t lo = static_cast<std::size_t>(std::floor(h));
            const std::size_t hi = std::min(lo + 1, window - 1);
            const double q = z[lo] + (h - lo) * (z[hi] - z[lo]);
            double tail_sum = 0.0;
            std::size_t tail_n = 0;
            for (std::size_t j = 0; j < window && z[j] <= q; ++j) {
                tail_sum += z[j];
                ++tail_n;
            }
            res.fhs_var[i * res.n_alphas + a] = -(p.mu + sig * q);
            res.fhs_es[i * res.n_alphas + a] = -(p.mu + sig * tail_sum / tail_n);
        }
    });

    for (const GarchFit& f : fits) {
        res.params.insert(res.params.end(), {f.params.mu, f.params.omega, f.params.alpha, f.params.gamma, f.params.beta, f.params.nu});
        res.nll.push_back(f.nll);
        res.converged.push_back(f.converged ? 1 : 0);
    }
    return res;
}

// Gamma(shape, 1) variate by Marsaglia and Tsang (2000).
inline double gamma_variate(Xoshiro256& rng, double shape) {
    if (shape < 1.0) return gamma_variate(rng, shape + 1.0) * std::pow(rng.uniform(), 1.0 / shape);
    const double d = shape - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
        const double x = rng.normal();
        double v = 1.0 + c * x;
        if (v <= 0.0) continue;
        v = v * v * v;
        const double u = rng.uniform();
        if (std::log(u) < 0.5 * x * x + d - d * v + d * std::log(v)) return d * v;
    }
}

// Simulates n returns, discarding `burn` initial observations; the recursion
// starts from the unconditional variance.
inline std::vector<double> garch_simulate(const GarchParams& p, const GarchSpec& spec, std::size_t n,
                                          std::uint64_t seed, std::size_t burn = 1000) {
    const double gamma = spec.gjr ? p.gamma : 0.0;
    const double pers = p.alpha + 0.5 * gamma + p.beta;
    if (!(pers < 1.0) || !(p.omega > 0.0)) throw std::invalid_argument("parameters must be covariance stationary");
    if (spec.student_t && !(p.nu > 2.0)) throw std::invalid_argument("nu must exceed 2");
    Xoshiro256 rng(seed);
    std::vector<double> out;
    out.reserve(n);
    double s2 = p.omega / (1.0 - pers);
    for (std::size_t t = 0; t < n + burn; ++t) {
        double z = rng.normal();
        if (spec.student_t) {
            const double chi2 = 2.0 * gamma_variate(rng, 0.5 * p.nu);
            z = z / std::sqrt(chi2 / p.nu) * std::sqrt((p.nu - 2.0) / p.nu);
        }
        const double e = std::sqrt(s2) * z;
        if (t >= burn) out.push_back(p.mu + e);
        s2 = p.omega + (p.alpha + (e < 0.0 ? gamma : 0.0)) * e * e + p.beta * s2;
    }
    return out;
}

}  // namespace qe
