"""Yield-curve factor models, forecasting and interest-rate risk in the banking book.

* Zero curves from the ECB Svensson parameters (continuously compounded, percent).
* Principal components of yield changes (Litterman and Scheinkman, 1991).
* Dynamic Nelson-Siegel model (Diebold and Li, 2006): two-step estimation (cross-sectional
  OLS with a fixed decay, then autoregressive factors) and one-step state-space estimation by
  maximum likelihood with the Kalman filter (Diebold, Rudebusch and Aruoba, 2006).
* Forecast evaluation: recursive out-of-sample forecasts and the Diebold-Mariano test with
  the small-sample correction of Harvey, Leybourne and Newbold (1997).
* Interest-rate risk in the banking book (IRRBB): economic value of equity under the six
  standardised shock scenarios of the Basel Committee (2016) with the post-shock floor of the
  EBA guidelines (EBA/GL/2022/14).

Units: yields in percent in the factor models, decimals in the IRRBB functions; maturities in
years; the Nelson-Siegel decay lambda is per year.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .curves import SvenssonCurve

DIEBOLD_LI_LAMBDA = 0.0609 * 12   # per year: curvature loading peaks at 30 months (Diebold and Li, 2006)


# --------------------------------------------------------------------------- data


def zero_curves(svensson: pd.DataFrame, maturities, freq: str | None = "ME") -> pd.DataFrame:
    """Zero rates (percent, continuous compounding) at the given maturities from Svensson parameters.

    With `freq` the parameters are first sampled at the last available day of each period.
    """
    params = svensson.resample(freq).last().dropna() if freq else svensson.dropna()
    maturities = np.asarray(maturities, dtype=float)
    rows = [100.0 * SvenssonCurve.from_series(row).zero_rate(maturities) for _, row in params.iterrows()]
    return pd.DataFrame(rows, index=params.index, columns=maturities)


# --------------------------------------------------------------------------- principal components


def principal_components(changes: pd.DataFrame, n: int = 3) -> dict:
    """PCA of yield changes (covariance matrix). Signs: level loadings positive on average,
    slope rising with maturity, curvature positive at the middle maturity."""
    X = changes.dropna().to_numpy(dtype=float)
    X = X - X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if vectors[:, 0].mean() < 0:
        vectors[:, 0] *= -1
    if n > 1 and vectors[-1, 1] - vectors[0, 1] < 0:
        vectors[:, 1] *= -1
    if n > 2 and vectors[vectors.shape[0] // 2, 2] < 0:
        vectors[:, 2] *= -1
    names = ["level", "slope", "curvature"] + [f"PC{i + 1}" for i in range(3, vectors.shape[1])]
    loadings = pd.DataFrame(vectors[:, :n], index=changes.columns, columns=names[:n])
    explained = pd.Series(values / values.sum(), index=names[:len(values)], name="share of variance")
    scores = pd.DataFrame(X @ vectors[:, :n], index=changes.dropna().index, columns=names[:n])
    return {"loadings": loadings, "explained": explained, "scores": scores, "eigenvalues": values}


# --------------------------------------------------------------------------- Nelson-Siegel


def ns_loadings(maturities, lam: float = DIEBOLD_LI_LAMBDA) -> np.ndarray:
    """Nelson-Siegel loadings [level, slope, curvature] for maturities in years."""
    tau = np.maximum(np.asarray(maturities, dtype=float), 1e-10)
    x = lam * tau
    slope = (1.0 - np.exp(-x)) / x
    return np.column_stack([np.ones_like(tau), slope, slope - np.exp(-x)])


def fit_ns_cross_section(yields: pd.DataFrame, lam: float = DIEBOLD_LI_LAMBDA) -> pd.DataFrame:
    """Level, slope and curvature factors by OLS at each date, with the decay fixed (Diebold and Li, 2006)."""
    X = ns_loadings(yields.columns.to_numpy(dtype=float), lam)
    beta = np.linalg.lstsq(X, yields.to_numpy(dtype=float).T, rcond=None)[0].T
    return pd.DataFrame(beta, index=yields.index, columns=["level", "slope", "curvature"])


def fit_var1(factors) -> dict:
    """VAR(1) x_t = c + A x_{t-1} + e_t by OLS; returns c, A, residual covariance and the mean."""
    F = np.asarray(factors, dtype=float)
    Z = np.column_stack([np.ones(F.shape[0] - 1), F[:-1]])
    B = np.linalg.lstsq(Z, F[1:], rcond=None)[0]
    resid = F[1:] - Z @ B
    c, A = B[0], B[1:].T
    k = F.shape[1]
    Sigma = resid.T @ resid / (resid.shape[0] - k - 1)
    mean = np.linalg.solve(np.eye(k) - A, c) if np.all(np.abs(np.linalg.eigvals(A)) < 1) else F.mean(axis=0)
    return {"c": c, "A": A, "Sigma": Sigma, "mean": mean}


@dataclass
class DNSModel:
    """State-space dynamic Nelson-Siegel model:
    y_t = L(lam) f_t + e_t, e_t ~ N(0, diag(h^2));  f_t - mu = A (f_{t-1} - mu) + u_t, u_t ~ N(0, Q).

    With `common_h` the measurement standard deviation is the same at every maturity."""

    lam: float
    mu: np.ndarray
    A: np.ndarray
    Q: np.ndarray
    h: np.ndarray            # measurement standard deviations, percent
    maturities: np.ndarray
    loglik: float = float("nan")
    converged: bool = False
    common_h: bool = False

    def pack(self) -> np.ndarray:
        C = np.linalg.cholesky(self.Q)
        tril = np.tril_indices(3)
        chol = C[tril].copy()
        chol[[0, 2, 5]] = np.log(np.diag(C))
        log_h = [math.log(float(np.mean(self.h)))] if self.common_h else np.log(self.h)
        return np.concatenate([[math.log(self.lam)], self.mu, self.A.ravel(), chol, log_h])

    @classmethod
    def unpack(cls, theta, maturities) -> "DNSModel":
        """Inverse of `pack`; a parameter vector with a single log h means a common measurement error."""
        theta = np.asarray(theta, dtype=float)
        n = len(maturities)
        lam = math.exp(theta[0])
        mu, A = theta[1:4], theta[4:13].reshape(3, 3)
        chol = theta[13:19].copy()
        chol[[0, 2, 5]] = np.exp(chol[[0, 2, 5]])
        C = np.zeros((3, 3))
        C[np.tril_indices(3)] = chol
        common = theta.size == 20 and n > 1
        h = np.full(n, math.exp(theta[19])) if common else np.exp(theta[19:19 + n])
        return cls(lam, mu, A, C @ C.T, h, np.asarray(maturities, dtype=float), common_h=common)


def _initial_state_covariance(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Unconditional covariance of the factors when their dynamics are stationary, otherwise a diffuse prior."""
    if np.all(np.abs(np.linalg.eigvals(A)) < 0.999):
        P = np.linalg.solve(np.eye(9) - np.kron(A, A), Q.ravel()).reshape(3, 3)
        return 0.5 * (P + P.T)
    return np.eye(3) * 1e4


def kalman_filter(model: DNSModel, yields, engine: str = "cpp", keep_states: bool = True) -> dict:
    """Kalman filter of the DNS model. The state starts from its unconditional distribution when
    the factor dynamics are stationary (otherwise from a diffuse prior). Missing values are skipped.

    `engine="cpp"` uses the C++ filter of the extension (Woodbury update, 3 x 3 factorisations only);
    `engine="numpy"` is the reference implementation with the textbook update.
    """
    Y = np.asarray(yields, dtype=float)
    P0 = _initial_state_covariance(model.A, model.Q)
    if engine == "cpp":
        from . import _core
        out = _core.dns_kalman_filter(Y, model.maturities, model.lam, model.mu, model.A, model.Q, model.h, P0,
                                      keep_states=keep_states)
        out["loglik"] = float(out["loglik"])
        return out
    if engine != "numpy":
        raise ValueError("engine must be 'cpp' or 'numpy'")
    T, n = Y.shape
    L = ns_loadings(model.maturities, model.lam)
    A, Q, mu = model.A, model.Q, model.mu
    H = np.diag(model.h ** 2)
    x = mu.copy()
    P = P0
    filtered = np.zeros((T, 3))
    predicted = np.zeros((T, 3))
    loglik = 0.0
    for t in range(T):
        if t > 0:
            x = mu + A @ (x - mu)
            P = A @ P @ A.T + Q
        predicted[t] = x
        ok = np.isfinite(Y[t])
        if ok.any():
            Lt = L[ok]
            v = Y[t, ok] - Lt @ x
            S = Lt @ P @ Lt.T + H[np.ix_(ok, ok)]
            try:
                cho = np.linalg.cholesky(S)
            except np.linalg.LinAlgError:
                return {"loglik": -np.inf, "filtered": filtered, "predicted": predicted}
            w = np.linalg.solve(cho, v)
            loglik -= 0.5 * (ok.sum() * math.log(2 * math.pi) + 2 * np.log(np.diag(cho)).sum() + w @ w)
            K = P @ Lt.T @ np.linalg.inv(S)
            x = x + K @ v
            P = (np.eye(3) - K @ Lt) @ P
            P = 0.5 * (P + P.T)
        filtered[t] = x
    return {"loglik": float(loglik), "filtered": filtered, "predicted": predicted}


def fit_dns_kalman(yields: pd.DataFrame, lam0: float = DIEBOLD_LI_LAMBDA, common_h: bool = False,
                   max_rounds: int = 10, tol: float = 1e-6, grad_tol: float = 1e-2) -> DNSModel:
    """Maximum-likelihood estimation of the state-space DNS model, started from the two-step estimates.

    BFGS with numerical gradients is restarted from its own solution until the log-likelihood improves by
    less than `tol`; `converged` reports whether the largest central-difference gradient component is below
    `grad_tol`. Non-stationary factor dynamics, decays outside (0.02, 5) per year and measurement standard
    deviations below 0.01 bp are excluded. With `common_h` one measurement variance is shared by all
    maturities, which rules out the degenerate optimum where some maturities are observed without error.
    """
    maturities = yields.columns.to_numpy(dtype=float)
    Y = yields.to_numpy(dtype=float)
    factors = fit_ns_cross_section(yields, lam0)
    var = fit_var1(factors.to_numpy())
    resid = Y - factors.to_numpy() @ ns_loadings(maturities, lam0).T
    h0 = np.maximum(np.nanstd(resid, axis=0), 1e-3)
    start = DNSModel(lam0, factors.mean().to_numpy(), var["A"], var["Sigma"] + 1e-8 * np.eye(3),
                     np.full(maturities.size, float(np.sqrt(np.mean(h0 ** 2)))) if common_h else h0, maturities,
                     common_h=common_h)

    def objective(theta):
        model = DNSModel.unpack(theta, maturities)
        if (not np.all(np.isfinite(model.A)) or np.max(np.abs(np.linalg.eigvals(model.A))) >= 1.0
                or not 0.02 < model.lam < 5.0 or np.min(model.h) < 1e-4):
            return 1e12
        ll = kalman_filter(model, Y, keep_states=False)["loglik"]
        return -ll if np.isfinite(ll) else 1e12

    theta = start.pack()
    best = objective(theta)
    for _ in range(max_rounds):
        res = optimize.minimize(objective, theta, method="BFGS", options={"gtol": 1e-6, "maxiter": 5000})
        improvement = best - res.fun
        if res.fun <= best:
            theta, best = res.x, res.fun
        if improvement < tol:
            break
    step = 1e-5 * np.eye(theta.size)
    grad = np.array([(objective(theta + e) - objective(theta - e)) / 2e-5 for e in step])   # central differences
    model = DNSModel.unpack(theta, maturities)
    model.loglik = -float(best)
    model.converged = bool(np.max(np.abs(grad)) < grad_tol)
    return model


def parameter_names(maturities, common_h: bool = False) -> list[str]:
    """Names of the packed DNS parameters (see DNSModel.pack)."""
    f = ["level", "slope", "curvature"]
    h = ["log h"] if common_h else [f"log h {m:g}y" for m in maturities]
    return (["log lambda"] + [f"mu {a}" for a in f] + [f"A {a}<-{b}" for a in f for b in f]
            + [f"chol Q {i}{j}" for i, j in zip(*np.tril_indices(3))] + h)


def dns_standard_errors(model: DNSModel, yields, step: float = 1e-4) -> pd.Series:
    """Asymptotic standard errors of the packed parameters from the inverse of the numerical Hessian of the
    log-likelihood (central differences). The standard error of lambda itself is lambda x se(log lambda)."""
    Y = np.asarray(yields, dtype=float)
    theta = model.pack()
    k = theta.size

    def f(th):
        return kalman_filter(DNSModel.unpack(th, model.maturities), Y, keep_states=False)["loglik"]

    H = np.zeros((k, k))
    eye = np.eye(k) * step
    for i in range(k):
        for j in range(i, k):
            H[i, j] = H[j, i] = (f(theta + eye[i] + eye[j]) - f(theta + eye[i] - eye[j])
                                 - f(theta - eye[i] + eye[j]) + f(theta - eye[i] - eye[j])) / (4 * step ** 2)
    cov = np.linalg.pinv(-H)
    se = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))
    return pd.Series(se, index=parameter_names(model.maturities, model.common_h), name="standard error")


def dns_forecast(model: DNSModel, last_state, horizon: int) -> np.ndarray:
    """h-step-ahead yield forecast from a filtered state (percent)."""
    x = np.asarray(last_state, dtype=float)
    for _ in range(horizon):
        x = model.mu + model.A @ (x - model.mu)
    return ns_loadings(model.maturities, model.lam) @ x


# --------------------------------------------------------------------------- forecasting


def recursive_forecasts(yields: pd.DataFrame, first_origin, horizons=(1, 6, 12), lam: float = DIEBOLD_LI_LAMBDA,
                        min_obs: int = 60) -> dict:
    """Out-of-sample forecast errors (actual - forecast, percent) with an expanding estimation window.

    Models: random walk; Diebold-Li two-step DNS with a direct AR(1) regression of each factor at horizon
    h; two-step DNS with an iterated VAR(1). Each forecast made at origin t uses data up to t only.
    """
    Y = yields.to_numpy(dtype=float)
    dates = yields.index
    L = ns_loadings(yields.columns.to_numpy(dtype=float), lam)
    F_all = fit_ns_cross_section(yields, lam).to_numpy()   # cross-sectional fits use each date only
    start = int(np.searchsorted(dates, pd.Timestamp(first_origin)))
    out = {}
    for h in horizons:
        errors = {"random walk": [], "DNS AR(1), direct": [], "DNS VAR(1)": []}
        target_dates = []
        for t in range(max(start, min_obs), len(dates) - h):
            F = F_all[:t + 1]
            actual = Y[t + h]
            ar = np.empty(3)
            for j in range(3):
                Z = np.column_stack([np.ones(t + 1 - h), F[:-h, j]])
                c, b = np.linalg.lstsq(Z, F[h:, j], rcond=None)[0]
                ar[j] = c + b * F[-1, j]
            var = fit_var1(F)
            x = F[-1]
            for _ in range(h):
                x = var["c"] + var["A"] @ x
            errors["random walk"].append(actual - Y[t])
            errors["DNS AR(1), direct"].append(actual - L @ ar)
            errors["DNS VAR(1)"].append(actual - L @ x)
            target_dates.append(dates[t + h])
        out[h] = {k: pd.DataFrame(v, index=pd.DatetimeIndex(target_dates), columns=yields.columns) for k, v in errors.items()}
    return out


def recursive_kalman_forecasts(yields: pd.DataFrame, first_origin, horizons=(1, 6, 12), refit_every: int = 12,
                               min_obs: int = 60, lam0: float = DIEBOLD_LI_LAMBDA, common_h: bool = False) -> dict:
    """Out-of-sample forecast errors of the state-space DNS model (actual - forecast, percent).

    Parameters are re-estimated by maximum likelihood every `refit_every` months on data up to the origin;
    between re-estimations the state is filtered with the latest parameters, again on data up to the origin.
    """
    Y = yields.to_numpy(dtype=float)
    dates = yields.index
    start = max(int(np.searchsorted(dates, pd.Timestamp(first_origin))), min_obs)
    errors = {h: [] for h in horizons}
    targets = {h: [] for h in horizons}
    fits = []
    model = None
    for t in range(start, len(dates) - min(horizons)):
        if model is None or (t - start) % refit_every == 0:
            model = fit_dns_kalman(yields.iloc[:t + 1], lam0=lam0 if model is None else model.lam, common_h=common_h)
            fits.append({"origin": dates[t], "lambda": model.lam, "loglik": model.loglik, "converged": model.converged})
        state = kalman_filter(model, Y[:t + 1])["filtered"][-1]
        for h in horizons:
            if t + h < len(dates):
                errors[h].append(Y[t + h] - dns_forecast(model, state, h))
                targets[h].append(dates[t + h])
    out = {h: pd.DataFrame(errors[h], index=pd.DatetimeIndex(targets[h]), columns=yields.columns) for h in horizons}
    out["fits"] = pd.DataFrame(fits).set_index("origin")
    return out


def diebold_mariano(e1, e2, horizon: int = 1, power: int = 2) -> dict:
    """Diebold-Mariano test of equal accuracy of two forecasts (loss |e|^power) with the Harvey,
    Leybourne and Newbold (1997) correction; negative statistics favour the first forecast."""
    e1, e2 = np.asarray(e1, dtype=float), np.asarray(e2, dtype=float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = d.size
    dbar = d.mean()
    dc = d - dbar
    gamma = [dc @ dc / n] + [dc[k:] @ dc[:-k] / n for k in range(1, horizon)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return {"statistic": float("nan"), "p_value": float("nan"), "mean_loss_difference": float(dbar)}
    correction = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = correction * dbar / math.sqrt(var)
    return {"statistic": float(stat), "p_value": float(2 * stats.t.sf(abs(stat), n - 1)), "mean_loss_difference": float(dbar)}


# --------------------------------------------------------------------------- IRRBB


# Basel Committee (2016), Interest rate risk in the banking book, Annex 2: shock sizes for the euro.
EUR_SHOCKS = {"parallel": 0.020, "short": 0.025, "long": 0.010}


def bcbs_scenarios(maturities, parallel=EUR_SHOCKS["parallel"], short=EUR_SHOCKS["short"], long=EUR_SHOCKS["long"]) -> dict:
    """Rate changes (decimals) of the six standardised scenarios at the given maturities (years).

    Short-rate shock S e^{-t/4}, long-rate shock L (1 - e^{-t/4}); steepener -0.65|S(t)| + 0.9|L(t)|,
    flattener +0.8|S(t)| - 0.6|L(t)|.
    """
    t = np.asarray(maturities, dtype=float)
    s = short * np.exp(-t / 4.0)
    lg = long * (1.0 - np.exp(-t / 4.0))
    return {"parallel up": np.full_like(t, parallel), "parallel down": np.full_like(t, -parallel),
            "steepener": -0.65 * np.abs(s) + 0.9 * np.abs(lg), "flattener": 0.8 * np.abs(s) - 0.6 * np.abs(lg),
            "short rates up": s, "short rates down": -s}


def eba_floor(maturities, base_rates=None) -> np.ndarray:
    """Post-shock floor of EBA/GL/2022/14: -150 bp at the shortest maturity, rising by 3 bp a year to 0%
    at 50 years; where observed rates are already lower, the observed rate is the floor."""
    t = np.asarray(maturities, dtype=float)
    floor = np.minimum(-0.015 + 0.0003 * t, 0.0)
    if base_rates is not None:
        floor = np.minimum(floor, np.asarray(base_rates, dtype=float))
    return floor


def present_value(times, amounts, zero_rates) -> float:
    """Present value of cash flows discounted at continuously compounded zero rates (decimals)."""
    times, amounts, zero_rates = (np.asarray(a, dtype=float) for a in (times, amounts, zero_rates))
    return float(np.sum(amounts * np.exp(-zero_rates * times)))


def delta_eve(times, net_amounts, base_rates, scenarios: dict, floor=True) -> pd.Series:
    """Change in the economic value of equity (present value of net cash flows) under each scenario."""
    base_rates = np.asarray(base_rates, dtype=float)
    pv0 = present_value(times, net_amounts, base_rates)
    out = {}
    for name, shock in scenarios.items():
        shocked = base_rates + np.asarray(shock, dtype=float)
        if floor:
            shocked = np.maximum(shocked, eba_floor(times, base_rates))
        out[name] = present_value(times, net_amounts, shocked) - pv0
    return pd.Series(out, name="change in EVE")


def bullet_cashflows(notional: float, coupon: float, years: float, freq: int = 1):
    """Times and amounts of a fixed-rate bullet instrument (coupon as a decimal)."""
    n = int(round(years * freq))
    times = np.arange(1, n + 1) / freq
    amounts = np.full(n, notional * coupon / freq)
    amounts[-1] += notional
    return times, amounts


def annuity_cashflows(notional: float, rate: float, years: float, freq: int = 12):
    """Times and amounts of a fully amortising fixed-rate loan with constant instalments."""
    n = int(round(years * freq))
    r = rate / freq
    payment = notional * r / (1 - (1 + r) ** -n) if r != 0 else notional / n
    return np.arange(1, n + 1) / freq, np.full(n, payment)


def mortgage_cashflows(notional: float, rate: float, years: float, cpr: float = 0.0, freq: int = 12):
    """Level-payment mortgage pool with a constant prepayment rate (CPR, annual decimal).

    Each period the scheduled instalment is recomputed on the outstanding balance and a fraction
    SMM = 1 - (1 - CPR)^(1/freq) of the balance after scheduled amortisation is prepaid.
    Returns times, total cash flows and principal repaid (scheduled + prepaid).
    """
    n = int(round(years * freq))
    r = rate / freq
    smm = 1.0 - (1.0 - cpr) ** (1.0 / freq)
    balance = notional
    amounts, principal = np.zeros(n), np.zeros(n)
    for k in range(n):
        remaining = n - k
        payment = balance * r / (1 - (1 + r) ** -remaining) if r != 0 else balance / remaining
        scheduled = payment - balance * r
        prepaid = (balance - scheduled) * smm if k < n - 1 else 0.0
        amounts[k] = payment + prepaid
        principal[k] = scheduled + prepaid
        balance -= scheduled + prepaid
    return np.arange(1, n + 1) / freq, amounts, principal


def non_maturity_deposit_cashflows(balance: float, core_share: float, core_years: float, freq: int = 12):
    """Behavioural slotting of non-maturity deposits: the non-core part runs off overnight, the core
    part evenly over `core_years` (average maturity core_years / 2 plus half a period)."""
    n = int(round(core_years * freq))
    times = np.concatenate([[1.0 / 365.0], np.arange(1, n + 1) / freq])
    amounts = np.concatenate([[balance * (1.0 - core_share)], np.full(n, balance * core_share / n)])
    return times, amounts


def par_swap_rate(zero_rate, years: float, freq: int = 1) -> float:
    """Par rate of a fixed-for-floating swap; `zero_rate` maps maturities (years) to continuously
    compounded zero rates (decimals)."""
    times = np.arange(1, int(round(years * freq)) + 1) / freq
    discount = np.exp(-np.asarray(zero_rate(times)) * times)
    return float((1.0 - discount[-1]) / (discount.sum() / freq))


def payer_swap_cashflows(notional: float, fixed_rate: float, years: float, zero_rate, reset: float = 0.25,
                         freq: int = 1):
    """Pay-fixed swap as cash flows: the floating leg is worth par at the next reset (its current coupon
    is set so that it is worth `notional` today); the fixed leg is a short fixed-rate bullet."""
    t_fix, a_fix = bullet_cashflows(notional, fixed_rate, years, freq)
    float_amount = notional * math.exp(float(zero_rate(np.array([reset]))[0]) * reset)
    return np.concatenate([[reset], t_fix]), np.concatenate([[float_amount], -a_fix])


def delta_nii(repricing_times, notionals, betas, shock, horizon: float = 1.0) -> float:
    """Change in net interest income over `horizon` years, constant balance sheet, repricing-gap method.

    Positions repricing at time t < horizon earn (assets, positive notionals) or pay (liabilities, negative)
    beta x shock(t) for the rest of the horizon; `shock` maps times to rate changes (decimals).
    """
    t = np.asarray(repricing_times, dtype=float)
    n = np.asarray(notionals, dtype=float)
    b = np.broadcast_to(np.asarray(betas, dtype=float), t.shape)
    inside = t < horizon
    return float(np.sum(n[inside] * b[inside] * np.asarray(shock(t[inside])) * (horizon - t[inside])))


def historical_curve_shocks(yields: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    """Overlapping changes of zero rates over `months` (same units as `yields`)."""
    return (yields - yields.shift(months)).dropna()


def delta_eve_book(cashflows, zero_rate, scenario_names=None, floor: bool = True) -> pd.Series:
    """Change in EVE when cash flows depend on the scenario (for example through prepayments).

    `cashflows(name)` returns (times, net amounts) for scenario `name` (None for the base case);
    `zero_rate` maps times to continuously compounded base zero rates (decimals).
    """
    t0, a0 = cashflows(None)
    pv0 = present_value(t0, a0, zero_rate(np.asarray(t0, dtype=float)))
    names = scenario_names or list(bcbs_scenarios([1.0]))
    out = {}
    for name in names:
        t, a = cashflows(name)
        t = np.asarray(t, dtype=float)
        base = np.asarray(zero_rate(t), dtype=float)
        shocked = base + bcbs_scenarios(t)[name]
        if floor:
            shocked = np.maximum(shocked, eba_floor(t, base))
        out[name] = present_value(t, a, shocked) - pv0
    return pd.Series(out, name="change in EVE")
