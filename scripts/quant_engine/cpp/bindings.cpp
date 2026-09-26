// pybind11 bindings exposing the C++ engines as the module quant_engine._core.
// Heavy computations release the GIL so that notebooks stay responsive and
// the engines can use all available cores.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "black_scholes.hpp"
#include "finite_difference.hpp"
#include "garch.hpp"
#include "hedging.hpp"
#include "heston.hpp"

namespace py = pybind11;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

std::vector<double> to_vector(const DoubleArray& a) {
    const auto buf = a.request();
    const double* ptr = static_cast<const double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.size);
}

py::array_t<double> to_array(const std::vector<double>& v) {
    py::array_t<double> out(v.size());
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

py::array_t<double> to_matrix(const std::vector<double>& v, std::size_t rows, std::size_t cols) {
    py::array_t<double> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

bool parse_option_type(const std::string& kind) {
    if (kind == "call") return true;
    if (kind == "put") return false;
    throw py::value_error("option_type must be 'call' or 'put'");
}

qe::HestonParams heston_params(double v0, double kappa, double theta, double sigma, double rho) {
    qe::HestonParams p{v0, kappa, theta, sigma, rho};
    p.validate();
    return p;
}

qe::GarchSpec garch_spec(const std::string& model, const std::string& dist) {
    qe::GarchSpec spec;
    if (model == "gjr") spec.gjr = true;
    else if (model == "garch") spec.gjr = false;
    else throw py::value_error("model must be 'garch' or 'gjr'");
    if (dist == "t") spec.student_t = true;
    else if (dist == "normal") spec.student_t = false;
    else throw py::value_error("dist must be 'normal' or 't'");
    return spec;
}

py::dict garch_params_dict(const qe::GarchParams& p) {
    py::dict d;
    d["mu"] = p.mu;
    d["omega"] = p.omega;
    d["alpha"] = p.alpha;
    d["gamma"] = p.gamma;
    d["beta"] = p.beta;
    d["nu"] = p.nu;
    return d;
}

qe::GarchParams garch_params_from(double mu, double omega, double alpha, double gamma, double beta, double nu) {
    qe::GarchParams p;
    p.mu = mu;
    p.omega = omega;
    p.alpha = alpha;
    p.gamma = gamma;
    p.beta = beta;
    p.nu = nu;
    return p;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ pricing and risk engines for quant_engine (Black-Scholes, Heston, finite differences, GARCH).";

    // ------------------------------------------------------------------ Black-Scholes
    m.def(
        "bs_price_greeks",
        [](double S, double K, double T, double r, double q, double sigma, const std::string& option_type) {
            const qe::BSResult res = qe::bs_price_greeks(S, K, T, r, q, sigma, parse_option_type(option_type));
            py::dict d;
            d["price"] = res.price;
            d["delta"] = res.delta;
            d["gamma"] = res.gamma;
            d["vega"] = res.vega;
            d["theta"] = res.theta;
            d["rho"] = res.rho;
            return d;
        },
        py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("q"), py::arg("sigma"),
        py::arg("option_type") = "call", "Black-Scholes-Merton price and Greeks of a European option.");

    m.def(
        "bs_implied_vol",
        [](const DoubleArray& prices, double S, const DoubleArray& strikes, const DoubleArray& maturities,
           const DoubleArray& rates, double q, const std::string& option_type) {
            const auto p = to_vector(prices), K = to_vector(strikes), T = to_vector(maturities), r = to_vector(rates);
            if (K.size() != p.size() || T.size() != p.size() || r.size() != p.size())
                throw py::value_error("prices, strikes, maturities and rates must have the same length");
            const bool is_call = parse_option_type(option_type);
            std::vector<double> out(p.size());
            for (std::size_t i = 0; i < p.size(); ++i) out[i] = qe::bs_implied_vol(p[i], S, K[i], T[i], r[i], q, is_call);
            return to_array(out);
        },
        py::arg("prices"), py::arg("S"), py::arg("strikes"), py::arg("maturities"), py::arg("rates"), py::arg("q"),
        py::arg("option_type") = "call",
        "Black-Scholes implied volatilities (NaN where the price violates no-arbitrage bounds).");

    // ------------------------------------------------------------------ Heston
    m.def(
        "heston_price",
        [](double S, const DoubleArray& strikes, const DoubleArray& maturities, const DoubleArray& rates, double q,
           double v0, double kappa, double theta, double sigma, double rho, const std::string& option_type) {
            const auto K = to_vector(strikes), T = to_vector(maturities), r = to_vector(rates);
            if (T.size() != K.size() || r.size() != K.size())
                throw py::value_error("strikes, maturities and rates must have the same length");
            const qe::HestonParams p = heston_params(v0, kappa, theta, sigma, rho);
            const bool is_call = parse_option_type(option_type);
            std::vector<double> out(K.size());
            {
                py::gil_scoped_release release;
                for (std::size_t i = 0; i < K.size(); ++i) out[i] = qe::heston_price(S, K[i], T[i], r[i], q, p, is_call);
            }
            return to_array(out);
        },
        py::arg("S"), py::arg("strikes"), py::arg("maturities"), py::arg("rates"), py::arg("q"), py::arg("v0"),
        py::arg("kappa"), py::arg("theta"), py::arg("sigma"), py::arg("rho"), py::arg("option_type") = "call",
        "Heston European option prices by Fourier inversion (vectorised over strikes and maturities).");

    m.def(
        "heston_cf",
        [](const py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>& u, double T, double r,
           double q, double v0, double kappa, double theta, double sigma, double rho) {
            const qe::HestonParams p = heston_params(v0, kappa, theta, sigma, rho);
            const auto buf = u.request();
            const auto* ptr = static_cast<const std::complex<double>*>(buf.ptr);
            py::array_t<std::complex<double>> out(buf.size);
            auto* dst = out.mutable_data();
            for (py::ssize_t i = 0; i < buf.size; ++i) dst[i] = qe::heston_cf(ptr[i], T, r, q, p);
            return out;
        },
        py::arg("u"), py::arg("T"), py::arg("r"), py::arg("q"), py::arg("v0"), py::arg("kappa"), py::arg("theta"),
        py::arg("sigma"), py::arg("rho"), "Characteristic function of ln(S_T / S_0) under Heston.");

    m.def(
        "heston_mc",
        [](double S, const DoubleArray& strikes, double T, double r, double q, double v0, double kappa, double theta,
           double sigma, double rho, int n_steps, long long n_paths, std::uint64_t seed, int n_threads) {
            const auto K = to_vector(strikes);
            const qe::HestonParams p = heston_params(v0, kappa, theta, sigma, rho);
            qe::HestonMcResult res;
            {
                py::gil_scoped_release release;
                res = qe::heston_mc_qe(S, K, T, r, q, p, n_steps, n_paths, seed, n_threads);
            }
            py::dict d;
            d["call"] = to_array(res.call);
            d["call_se"] = to_array(res.call_se);
            d["put"] = to_array(res.put);
            d["put_se"] = to_array(res.put_se);
            d["forward"] = res.forward;
            d["forward_se"] = res.forward_se;
            return d;
        },
        py::arg("S"), py::arg("strikes"), py::arg("T"), py::arg("r"), py::arg("q"), py::arg("v0"), py::arg("kappa"),
        py::arg("theta"), py::arg("sigma"), py::arg("rho"), py::arg("n_steps"), py::arg("n_paths"), py::arg("seed"),
        py::arg("n_threads") = 0,
        "Heston Monte Carlo (Andersen QE scheme): call/put prices and standard errors for several strikes.");

    // ------------------------------------------------------------------ Finite differences
    m.def(
        "bs_finite_difference",
        [](double S, double K, double T, double r, double q, double sigma, const std::string& option_type,
           bool american, int n_space, int n_time, double n_sd, int rannacher_steps, double omega) {
            const bool is_call = parse_option_type(option_type);
            qe::FdResult res;
            {
                py::gil_scoped_release release;
                res = qe::bs_finite_difference(S, K, T, r, q, sigma, is_call, american, n_space, n_time, n_sd,
                                               rannacher_steps, omega);
            }
            py::dict d;
            d["price"] = res.price;
            d["delta"] = res.delta;
            d["gamma"] = res.gamma;
            d["spot_grid"] = to_array(res.spot_grid);
            d["values"] = to_array(res.values);
            d["tau_grid"] = to_array(res.tau_grid);
            d["exercise_boundary"] = to_array(res.exercise_boundary);
            d["psor_iterations"] = res.psor_iterations;
            return d;
        },
        py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("q"), py::arg("sigma"),
        py::arg("option_type") = "put", py::arg("american") = false, py::arg("n_space") = 400,
        py::arg("n_time") = 400, py::arg("n_sd") = 5.0, py::arg("rannacher_steps") = 2, py::arg("omega") = 1.2,
        "Crank-Nicolson (Rannacher start, PSOR for American exercise) Black-Scholes solver.");

    // ------------------------------------------------------------------ GARCH
    m.def(
        "garch_nll",
        [](const DoubleArray& returns, double mu, double omega, double alpha, double gamma, double beta, double nu,
           const std::string& model, const std::string& dist) {
            const auto r = to_vector(returns);
            const qe::GarchSpec spec = garch_spec(model, dist);
            std::vector<double> s2(r.size() + 1);
            const double nll = qe::garch_nll(r.data(), r.size(), garch_params_from(mu, omega, alpha, gamma, beta, nu),
                                             spec, s2.data());
            return py::make_tuple(nll, to_array(s2));
        },
        py::arg("returns"), py::arg("mu"), py::arg("omega"), py::arg("alpha"), py::arg("gamma"), py::arg("beta"),
        py::arg("nu"), py::arg("model") = "gjr", py::arg("dist") = "t",
        "Negative log-likelihood and conditional variances (n + 1 values, the last is the forecast).");

    m.def(
        "garch_fit",
        [](const DoubleArray& returns, const std::string& model, const std::string& dist) {
            const auto r = to_vector(returns);
            const qe::GarchSpec spec = garch_spec(model, dist);
            qe::GarchFit fit;
            {
                py::gil_scoped_release release;
                fit = qe::garch_fit(r.data(), r.size(), spec);
            }
            py::dict d = garch_params_dict(fit.params);
            d["nll"] = fit.nll;
            d["evaluations"] = fit.evaluations;
            d["converged"] = fit.converged;
            return d;
        },
        py::arg("returns"), py::arg("model") = "gjr", py::arg("dist") = "t",
        "Maximum-likelihood fit of GARCH(1,1) or GJR-GARCH(1,1) with Nelder-Mead.");

    m.def(
        "garch_rolling_forecast",
        [](const DoubleArray& returns, std::size_t window, std::size_t refit_every, const std::string& model,
           const std::string& dist, const DoubleArray& alphas, int n_threads) {
            const auto r = to_vector(returns);
            const auto a = to_vector(alphas);
            const qe::GarchSpec spec = garch_spec(model, dist);
            qe::RollingGarchResult res;
            {
                py::gil_scoped_release release;
                res = qe::garch_rolling_forecast(r, window, refit_every, spec, a, n_threads);
            }
            py::dict d;
            d["mu"] = to_array(res.mu);
            d["sigma"] = to_array(res.sigma);
            d["nu"] = to_array(res.nu);
            d["fhs_var"] = to_matrix(res.fhs_var, res.n_forecasts, res.n_alphas);
            d["fhs_es"] = to_matrix(res.fhs_es, res.n_forecasts, res.n_alphas);
            d["params"] = to_matrix(res.params, res.nll.size(), 6);
            d["nll"] = to_array(res.nll);
            d["converged"] = res.converged;
            return d;
        },
        py::arg("returns"), py::arg("window"), py::arg("refit_every"), py::arg("model") = "gjr",
        py::arg("dist") = "t", py::arg("alphas") = std::vector<double>{0.01, 0.025}, py::arg("n_threads") = 0,
        "Rolling one-step-ahead GARCH forecasts with filtered historical simulation VaR and ES.");

    // ------------------------------------------------------------------ Hedging
    m.def(
        "hedge_simulation",
        [](double S0, double K, double T, double r_dom, double r_for, const std::string& option_type, double sigma_price,
           double sigma_hedge, int n_steps, int rebalance_every, double cost_rate, const std::string& dynamics, double mu,
           double sigma, const DoubleArray& heston, const DoubleArray& garch, const DoubleArray& residuals,
           long long n_paths, std::uint64_t seed, int n_threads) {
            qe::HedgeContract c;
            c.S0 = S0; c.K = K; c.T = T; c.r_dom = r_dom; c.r_for = r_for;
            c.is_call = parse_option_type(option_type);
            c.sigma_price = sigma_price; c.sigma_hedge = sigma_hedge;
            c.n_steps = n_steps; c.rebalance_every = rebalance_every; c.cost_rate = cost_rate;
            qe::HedgeDynamics d;
            d.mu = mu;
            d.sigma = sigma;
            if (dynamics == "gbm") {
                d.kind = qe::Dynamics::GBM;
            } else if (dynamics == "heston") {
                const auto hp = to_vector(heston);
                if (hp.size() != 5) throw py::value_error("heston must be (v0, kappa, theta, sigma, rho)");
                d.kind = qe::Dynamics::Heston;
                d.heston = heston_params(hp[0], hp[1], hp[2], hp[3], hp[4]);
            } else if (dynamics == "garch_fhs") {
                const auto gp = to_vector(garch);
                if (gp.size() != 6) throw py::value_error("garch must be (mu, omega, alpha, gamma, beta, var0) in percent units");
                d.kind = qe::Dynamics::GarchFhs;
                d.g_mu = gp[0]; d.g_omega = gp[1]; d.g_alpha = gp[2]; d.g_gamma = gp[3]; d.g_beta = gp[4]; d.g_var0 = gp[5];
                d.residuals = to_vector(residuals);
            } else {
                throw py::value_error("dynamics must be 'gbm', 'heston' or 'garch_fhs'");
            }
            qe::HedgeResult res;
            {
                py::gil_scoped_release release;
                res = qe::simulate_hedge(c, d, n_paths, seed, n_threads);
            }
            py::dict out;
            out["pnl"] = to_array(res.pnl);
            out["costs"] = to_array(res.costs);
            out["final_spot"] = to_array(res.final_spot);
            out["premium"] = res.premium;
            return out;
        },
        py::arg("S0"), py::arg("K"), py::arg("T"), py::arg("r_dom"), py::arg("r_for"), py::arg("option_type"),
        py::arg("sigma_price"), py::arg("sigma_hedge"), py::arg("n_steps"), py::arg("rebalance_every"),
        py::arg("cost_rate"), py::arg("dynamics"), py::arg("mu") = 0.0, py::arg("sigma") = 0.1,
        py::arg("heston") = std::vector<double>{}, py::arg("garch") = std::vector<double>{},
        py::arg("residuals") = std::vector<double>{}, py::arg("n_paths") = 100000, py::arg("seed") = 12345,
        py::arg("n_threads") = 0,
        "P&L of a short European option delta-hedged in discrete time under GBM, Heston or GARCH-FHS dynamics.");

    m.def(
        "garch_simulate",
        [](double mu, double omega, double alpha, double gamma, double beta, double nu, std::size_t n,
           std::uint64_t seed, const std::string& model, const std::string& dist, std::size_t burn) {
            const qe::GarchSpec spec = garch_spec(model, dist);
            return to_array(qe::garch_simulate(garch_params_from(mu, omega, alpha, gamma, beta, nu), spec, n, seed, burn));
        },
        py::arg("mu"), py::arg("omega"), py::arg("alpha"), py::arg("gamma"), py::arg("beta"), py::arg("nu"),
        py::arg("n"), py::arg("seed"), py::arg("model") = "gjr", py::arg("dist") = "t", py::arg("burn") = 1000,
        "Simulates a GARCH(1,1) or GJR-GARCH(1,1) return series.");
}
