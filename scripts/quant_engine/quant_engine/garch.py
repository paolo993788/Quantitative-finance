"""GARCH(1,1) and GJR-GARCH(1,1) estimation and forecasting.

The C++ engine performs maximum-likelihood estimation (Nelder-Mead) and the
rolling out-of-sample forecasts. Constraints: omega > 0, alpha >= 0,
beta >= 0, alpha + gamma >= 0 (gamma may be negative) and
alpha + gamma / 2 + beta < 1; this module adds a pure Python reference
likelihood, a SciPy-based reference estimator, standard errors and the news
impact curve. Returns are expected in percent (100 x log-returns), which keeps
the optimisation well scaled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, special

from . import require_cpp

PARAM_NAMES = ("mu", "omega", "alpha", "gamma", "beta", "nu")


def _free_names(model: str, dist: str):
    names = ["mu", "omega", "alpha", "beta"]
    if model == "gjr":
        names.insert(3, "gamma")
    if dist == "t":
        names.append("nu")
    return names


@dataclass
class GarchResult:
    model: str
    dist: str
    params: dict
    std_errors: dict
    loglik: float
    n_obs: int
    converged: bool

    @property
    def n_params(self) -> int:
        return len(_free_names(self.model, self.dist))

    @property
    def aic(self) -> float:
        return -2.0 * self.loglik + 2.0 * self.n_params

    @property
    def bic(self) -> float:
        return -2.0 * self.loglik + self.n_params * math.log(self.n_obs)

    @property
    def persistence(self) -> float:
        p = self.params
        return p["alpha"] + 0.5 * p["gamma"] + p["beta"]

    @property
    def half_life(self) -> float:
        """Half-life of a volatility shock, in observations."""
        return math.log(0.5) / math.log(self.persistence)

    def summary(self) -> pd.DataFrame:
        names = _free_names(self.model, self.dist)
        return pd.DataFrame({"estimate": [self.params[n] for n in names],
                             "std_error": [self.std_errors.get(n, np.nan) for n in names]}, index=names)


def nll(returns, params: dict, model="gjr", dist="t"):
    """Negative log-likelihood from the C++ engine."""
    core = require_cpp()
    p = {name: params.get(name, 0.0) for name in PARAM_NAMES}
    value, _ = core.garch_nll(np.asarray(returns, dtype=float), p["mu"], p["omega"], p["alpha"],
                              p["gamma"], p["beta"], p["nu"], model, dist)
    return value


def conditional_variance(returns, params: dict, model="gjr", dist="t"):
    """Conditional variances (n + 1 values; the last is the one-step forecast)."""
    core = require_cpp()
    p = {name: params.get(name, 0.0) for name in PARAM_NAMES}
    _, s2 = core.garch_nll(np.asarray(returns, dtype=float), p["mu"], p["omega"], p["alpha"],
                           p["gamma"], p["beta"], p["nu"], model, dist)
    return s2


def nll_reference(returns, params: dict, model="gjr", dist="t"):
    """Pure Python negative log-likelihood (reference for the C++ version)."""
    r = np.asarray(returns, dtype=float)
    mu, omega, alpha = params["mu"], params["omega"], params["alpha"]
    gamma = params.get("gamma", 0.0) if model == "gjr" else 0.0
    beta, nu = params["beta"], params.get("nu", 0.0)
    e = r - mu
    s2 = float(np.mean(e**2))
    total = 0.0
    if dist == "t":
        const = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    for et in e:
        if not s2 > 0.0:
            return 1e10  # infeasible trial point (the optimiser may step outside the constraints)
        if dist == "t":
            total -= const - 0.5 * math.log(s2) - 0.5 * (nu + 1) * math.log1p(et * et / ((nu - 2) * s2))
        else:
            total += 0.5 * (math.log(2 * math.pi) + math.log(s2) + et * et / s2)
        s2 = omega + (alpha + (gamma if et < 0 else 0.0)) * et * et + beta * s2
    return total


def _numerical_std_errors(returns, params: dict, model: str, dist: str) -> dict:
    """Standard errors from the inverse of a central-difference Hessian of the
    negative log-likelihood (classical, non-robust)."""
    names = _free_names(model, dist)
    x0 = np.array([params[n] for n in names])
    h = np.maximum(np.abs(x0) * 1e-4, 1e-6)

    def f(x):
        return nll(returns, {**params, **dict(zip(names, x))}, model, dist)

    k = len(x0)
    H = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = h[i], h[j]
            H[i, j] = H[j, i] = (f(x0 + ei + ej) - f(x0 + ei - ej) - f(x0 - ei + ej) + f(x0 - ei - ej)) / (4 * h[i] * h[j])
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    return dict(zip(names, se))


def fit(returns, model="gjr", dist="t", std_errors=True) -> GarchResult:
    """Maximum-likelihood fit with the C++ Nelder-Mead estimator."""
    core = require_cpp()
    r = np.asarray(returns, dtype=float)
    res = core.garch_fit(r, model, dist)
    params = {n: res[n] for n in PARAM_NAMES}
    se = _numerical_std_errors(r, params, model, dist) if std_errors else {}
    return GarchResult(model, dist, params, se, -res["nll"], len(r), bool(res["converged"]))


def fit_reference(returns, model="gjr", dist="t") -> GarchResult:
    """Reference estimator: SciPy SLSQP on the pure Python likelihood."""
    r = np.asarray(returns, dtype=float)
    names = _free_names(model, dist)
    var = r.var()
    start = {"mu": r.mean(), "omega": 0.05 * var, "alpha": 0.05, "gamma": 0.05, "beta": 0.9, "nu": 8.0}
    bounds = {"mu": (None, None), "omega": (1e-8 * var, 10 * var), "alpha": (0.0, 1.0),
              "gamma": (-1.0, 1.0), "beta": (0.0, 1.0), "nu": (2.05, 500.0)}

    def unpack(x):
        p = dict(zip(names, x))
        p.setdefault("gamma", 0.0)
        p.setdefault("nu", 0.0)
        return p

    def objective(x):
        return nll_reference(r, unpack(x), model, dist)

    def stationarity(x):
        p = unpack(x)
        return 1.0 - 1e-6 - (p["alpha"] + 0.5 * p["gamma"] + p["beta"])

    def positive_news_response(x):
        p = unpack(x)
        return p["alpha"] + p["gamma"]

    sol = optimize.minimize(objective, [start[n] for n in names], method="SLSQP",
                            bounds=[bounds[n] for n in names],
                            constraints=[{"type": "ineq", "fun": stationarity},
                                         {"type": "ineq", "fun": positive_news_response}],
                            options={"ftol": 1e-12, "maxiter": 1000})
    params = unpack(sol.x)
    return GarchResult(model, dist, params, {}, -sol.fun, len(r), bool(sol.success))


def simulate(params: dict, n: int, seed: int, model="gjr", dist="t", burn=1000):
    """Simulated returns from the C++ engine."""
    core = require_cpp()
    p = {name: params.get(name, 0.0) for name in PARAM_NAMES}
    return core.garch_simulate(p["mu"], p["omega"], p["alpha"], p["gamma"], p["beta"], p["nu"], n, seed, model, dist, burn)


def news_impact_curve(result: GarchResult, shocks):
    """Next-period variance as a function of today's shock e_t, holding the
    current variance at its unconditional level (Engle and Ng, 1993)."""
    p = result.params
    gamma = p["gamma"] if result.model == "gjr" else 0.0
    uncond = p["omega"] / (1.0 - result.persistence)
    e = np.asarray(shocks, dtype=float)
    return p["omega"] + (p["alpha"] + gamma * (e < 0)) * e**2 + p["beta"] * uncond


def rolling_forecast(returns: pd.Series, window=1000, refit_every=20, model="gjr", dist="t",
                     alphas=(0.01, 0.025), n_threads=0) -> pd.DataFrame:
    """Rolling one-step-ahead forecasts computed in parallel by the C++ engine.

    The row for date t uses only returns strictly before t. Columns: mu,
    sigma, nu and filtered historical simulation VaR/ES (positive losses)
    for each tail probability in `alphas`.
    """
    core = require_cpp()
    out = core.garch_rolling_forecast(returns.to_numpy(dtype=float), window, refit_every, model, dist,
                                      np.asarray(alphas, dtype=float), n_threads)
    frame = pd.DataFrame({"mu": out["mu"], "sigma": out["sigma"], "nu": out["nu"]}, index=returns.index[window:])
    for j, a in enumerate(alphas):
        frame[f"fhs_var_{a:g}"] = out["fhs_var"][:, j]
        frame[f"fhs_es_{a:g}"] = out["fhs_es"][:, j]
    frame.attrs["params"] = pd.DataFrame(out["params"], columns=PARAM_NAMES)
    frame.attrs["converged"] = np.asarray(out["converged"], dtype=bool)
    return frame
