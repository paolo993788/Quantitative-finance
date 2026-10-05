"""Arbitrage-free dynamic Nelson-Siegel model (Christensen, Diebold and Rudebusch, 2011).

The dynamic Nelson-Siegel (DNS) model of ``term_structure`` is not arbitrage-free. Christensen, Diebold and
Rudebusch (CDR) show that the Nelson-Siegel loadings are the factor loadings of a three-factor Gaussian affine
model whose risk-neutral mean-reversion matrix is

    K^Q = [[0, 0, 0], [0, lambda, -lambda], [0, 0, lambda]],    r_t = X1_t + X2_t,

so that the absence of arbitrage only adds a deterministic yield-adjustment term to the Nelson-Siegel curve:

    y_t(tau) = X1 + X2 (1 - e^{-lambda tau}) / (lambda tau) + X3 [(1 - e^{-lambda tau}) / (lambda tau) - e^{-lambda tau}]
               - C(tau) / tau,
    C(tau) / tau = (1 / (2 tau)) int_0^tau sum_j sigma_j^2 B_j(s)^2 ds,

with B_1(s) = s, B_2(s) = (1 - e^{-lambda s}) / lambda, B_3(s) = B_2(s) - s e^{-lambda s} (the convexity of the
bond price, -Var(int_0^tau r) / (2 tau)). This module implements the independent-factor AFNS model of CDR
(Sigma and the physical mean-reversion matrix K^P diagonal), for which C(tau) / tau has the closed form of
CDR (2011, section 4.2):

    sigma_1^2 tau^2 / 6
    + sigma_2^2 [1 / (2 lambda^2) - (1 - e^{-lambda tau}) / (lambda^3 tau) + (1 - e^{-2 lambda tau}) / (4 lambda^3 tau)]
    + sigma_3^2 [1 / (2 lambda^2) + e^{-lambda tau} / lambda^2 - tau e^{-2 lambda tau} / (4 lambda)
                 - 3 e^{-2 lambda tau} / (4 lambda^2) - 2 (1 - e^{-lambda tau}) / (lambda^3 tau)
                 + 5 (1 - e^{-2 lambda tau}) / (8 lambda^3 tau)].

Physical dynamics: dX = K^P (theta - X) dt + Sigma dW with K^P = diag(kappa), discretised exactly at the sampling
interval dt (monthly: 1/12): X_t - theta = e^{-kappa dt} (X_{t-1} - theta) + eta_t, Var(eta_i) =
sigma_i^2 (1 - e^{-2 kappa_i dt}) / (2 kappa_i). The model is a DNS state-space model with these transition
parameters and adjusted yields y - adjustment, so it is filtered with the Kalman filter of ``term_structure``.

Units as in ``term_structure``: yields and factors in percent, sigma in percent per square-root year, maturities
in years, lambda and kappa per year.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from . import term_structure as ts

MONTHLY = 1.0 / 12.0


def yield_adjustment(maturities, lam: float, sigma) -> np.ndarray:
    """-C(tau)/tau of the independent-factor AFNS model, in percent (sigma in percent per sqrt-year)."""
    tau = np.maximum(np.asarray(maturities, dtype=float), 1e-10)
    s1, s2, s3 = np.asarray(sigma, dtype=float) / 100.0      # decimals
    l = float(lam)
    e1, e2 = np.exp(-l * tau), np.exp(-2.0 * l * tau)
    # 1 - e^{-x} written with expm1 to keep precision at short maturities.
    a1, a2 = -np.expm1(-l * tau), -np.expm1(-2.0 * l * tau)
    t1 = s1**2 * tau**2 / 6.0
    t2 = s2**2 * (1.0 / (2 * l**2) - a1 / (l**3 * tau) + a2 / (4 * l**3 * tau))
    t3 = s3**2 * (1.0 / (2 * l**2) + e1 / l**2 - tau * e2 / (4 * l) - 3 * e2 / (4 * l**2)
                  - 2 * a1 / (l**3 * tau) + 5 * a2 / (8 * l**3 * tau))
    return -100.0 * (t1 + t2 + t3)


@dataclass
class AFNSModel:
    """Independent-factor AFNS model: decay lam, means theta, physical mean reversions kappa and volatilities
    sigma of the three factors, measurement standard deviations h (percent)."""

    lam: float
    theta: np.ndarray
    kappa: np.ndarray
    sigma: np.ndarray
    h: np.ndarray
    maturities: np.ndarray
    dt: float = MONTHLY
    loglik: float = float("nan")
    converged: bool = False
    common_h: bool = True
    arbitrage_free: bool = True   # False: the same model without the yield adjustment (independent-factor DNS)

    def adjustment(self) -> np.ndarray:
        if not self.arbitrage_free:
            return np.zeros(len(self.maturities))
        return yield_adjustment(self.maturities, self.lam, self.sigma)

    def transition(self):
        """(A, Q) of the exact discretisation at the sampling interval dt."""
        a = np.exp(-self.kappa * self.dt)
        q = self.sigma**2 * (-np.expm1(-2.0 * self.kappa * self.dt)) / (2.0 * self.kappa)
        return np.diag(a), np.diag(q)

    def to_dns(self) -> ts.DNSModel:
        """The state-space model of the adjusted yields y - adjustment."""
        A, Q = self.transition()
        return ts.DNSModel(self.lam, np.asarray(self.theta, float), A, Q, np.asarray(self.h, float),
                           np.asarray(self.maturities, float), common_h=self.common_h)

    def pack(self) -> np.ndarray:
        log_h = [math.log(float(np.mean(self.h)))] if self.common_h else np.log(self.h)
        return np.concatenate([[math.log(self.lam)], self.theta, np.log(self.kappa), np.log(self.sigma), log_h])

    @classmethod
    def unpack(cls, theta, maturities, dt=MONTHLY, arbitrage_free=True) -> "AFNSModel":
        theta = np.asarray(theta, dtype=float)
        n = len(maturities)
        common = theta.size == 11 and n > 1
        h = np.full(n, math.exp(theta[10])) if common else np.exp(theta[10:10 + n])
        return cls(math.exp(theta[0]), theta[1:4].copy(), np.exp(theta[4:7]), np.exp(theta[7:10]), h,
                   np.asarray(maturities, dtype=float), dt, common_h=common, arbitrage_free=arbitrage_free)


def kalman_filter(model: AFNSModel, yields, engine: str = "cpp", keep_states: bool = True) -> dict:
    """Kalman filter of the AFNS model: the DNS filter applied to the adjusted yields (missing values skipped)."""
    Y = np.asarray(yields, dtype=float) - model.adjustment()[None, :]
    return ts.kalman_filter(model.to_dns(), Y, engine=engine, keep_states=keep_states)


def forecast(model: AFNSModel, last_state, horizon: int) -> np.ndarray:
    """h-step-ahead yield forecast (percent): Nelson-Siegel loadings of E[X_{t+h}] plus the yield adjustment."""
    x = model.theta + np.exp(-model.kappa * model.dt * horizon) * (np.asarray(last_state, float) - model.theta)
    return ts.ns_loadings(model.maturities, model.lam) @ x + model.adjustment()


def fit_afns(yields: pd.DataFrame, lam0: float = ts.DIEBOLD_LI_LAMBDA, common_h: bool = True, dt: float = MONTHLY,
             max_rounds: int = 10, tol: float = 1e-6, grad_tol: float = 1e-2, arbitrage_free: bool = True) -> AFNSModel:
    """Maximum-likelihood estimation of the independent-factor AFNS model (with `arbitrage_free=False`, of the
    same model without the yield adjustment, the independent-factor DNS model of CDR's comparison).

    Starting values from the two-step DNS estimates (per-factor AR(1) coefficients mapped to kappa, innovation
    variances to sigma). The optimisation follows ``term_structure.fit_dns_kalman``: BFGS restarted from its own
    solution with forward and then central differences, then Newton steps with the numerical Hessian while it is
    positive definite. Mean reversions below 0.01 per year (half-life above 69 years), decays outside (0.02, 5)
    per year and measurement standard deviations below 0.01 bp are excluded. `converged` reports whether the
    largest central-difference gradient component is below `grad_tol`.
    """
    maturities = yields.columns.to_numpy(dtype=float)
    Y = yields.to_numpy(dtype=float)
    factors = ts.fit_ns_cross_section(yields, lam0).to_numpy()
    theta0 = factors.mean(axis=0)
    kappa0, sigma0 = np.empty(3), np.empty(3)
    for j in range(3):
        x, y = factors[:-1, j] - theta0[j], factors[1:, j] - theta0[j]
        a = float(np.clip(x @ y / (x @ x), 0.5, 0.995))
        q = float(np.var(y - a * x))
        kappa0[j] = -math.log(a) / dt
        sigma0[j] = math.sqrt(q * 2 * kappa0[j] / -math.expm1(-2 * kappa0[j] * dt))
    resid = Y - factors @ ts.ns_loadings(maturities, lam0).T
    h0 = np.maximum(np.nanstd(resid, axis=0), 1e-3)
    start = AFNSModel(lam0, theta0, kappa0, sigma0,
                      np.full(maturities.size, float(np.sqrt(np.mean(h0**2)))) if common_h else h0,
                      maturities, dt, common_h=common_h, arbitrage_free=arbitrage_free)

    def objective(th):
        if not np.all(np.isfinite(th)):
            return 1e12
        model = AFNSModel.unpack(th, maturities, dt, arbitrage_free)
        if not 0.02 < model.lam < 5.0 or np.min(model.kappa) < 0.01 or np.max(model.kappa) > 50 or np.min(model.h) < 1e-4:
            return 1e12
        ll = kalman_filter(model, Y, keep_states=False)["loglik"]
        return -ll if np.isfinite(ll) else 1e12

    def gradient(th, step=1e-5):
        return np.array([(objective(th + e) - objective(th - e)) / (2 * step) for e in step * np.eye(th.size)])

    th = start.pack()
    best = objective(th)
    for jac in (None, gradient):
        for _ in range(max_rounds):
            res = optimize.minimize(objective, th, jac=jac, method="BFGS", options={"gtol": 1e-6, "maxiter": 5000})
            improvement = best - res.fun
            if res.fun <= best:
                th, best = res.x, res.fun
            if improvement < tol:
                break
    for _ in range(max_rounds):
        grad = gradient(th)
        if np.max(np.abs(grad)) < 0.1 * grad_tol:
            break
        H = ts._hessian(objective, th, 1e-4)
        try:
            np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            break
        direction = -np.linalg.solve(H, grad)
        for t in 0.5 ** np.arange(12):
            value = objective(th + t * direction)
            if value <= best:
                th, best = th + t * direction, value
                break
        else:
            break
    model = AFNSModel.unpack(th, maturities, dt, arbitrage_free)
    model.loglik = -float(best)
    model.converged = bool(np.max(np.abs(gradient(th))) < grad_tol)
    return model


def recursive_forecasts(yields: pd.DataFrame, first_origin, horizons=(1, 6, 12), refit_every: int = 12,
                        min_obs: int = 60, lam0: float = ts.DIEBOLD_LI_LAMBDA, common_h: bool = True,
                        arbitrage_free: bool = True) -> dict:
    """Out-of-sample forecast errors of the AFNS model (actual - forecast, percent), with the same expanding
    window, re-estimation schedule and timing as ``term_structure.recursive_kalman_forecasts``: parameters
    re-estimated every `refit_every` months on data up to the origin, the state filtered on data up to the origin."""
    Y = yields.to_numpy(dtype=float)
    dates = yields.index
    start = max(int(np.searchsorted(dates, pd.Timestamp(first_origin))), min_obs)
    errors = {h: [] for h in horizons}
    targets = {h: [] for h in horizons}
    fits = []
    model = None
    for t in range(start, len(dates) - min(horizons)):
        if model is None or (t - start) % refit_every == 0:
            model = fit_afns(yields.iloc[:t + 1], lam0=lam0 if model is None else model.lam, common_h=common_h,
                             arbitrage_free=arbitrage_free)
            fits.append({"origin": dates[t], "lambda": model.lam, "loglik": model.loglik, "converged": model.converged})
        state = kalman_filter(model, Y[:t + 1])["filtered"][-1]
        for h in horizons:
            if t + h < len(dates):
                errors[h].append(Y[t + h] - forecast(model, state, h))
                targets[h].append(dates[t + h])
    out = {h: pd.DataFrame(errors[h], index=pd.DatetimeIndex(targets[h]), columns=yields.columns) for h in horizons}
    out["fits"] = pd.DataFrame(fits).set_index("origin")
    return out
