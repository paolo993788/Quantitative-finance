"""Heston model: Python interface to the C++ engine, NumPy/SciPy reference
implementations and calibration.

Model (risk-neutral measure):

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW^v_t,   corr(dW^S, dW^v) = rho

Rates r and q are continuously compounded, maturities are in years.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np
from scipy import integrate, optimize

from . import require_cpp


@dataclass(frozen=True)
class HestonParams:
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    @property
    def feller_ratio(self) -> float:
        """2 kappa theta / sigma^2; the variance stays strictly positive if > 1."""
        return 2.0 * self.kappa * self.theta / self.sigma**2

    def as_tuple(self):
        return astuple(self)


# Fang and Oosterlee (2008), Table 4: S0 = K = 100, T = 1, r = q = 0.
FANG_OOSTERLEE_PARAMS = HestonParams(v0=0.0175, kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711)
FANG_OOSTERLEE_REFERENCE_CALL = 5.785155450


def _as_arrays(strikes, maturities, rates):
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    maturities = np.broadcast_to(np.asarray(maturities, dtype=float), strikes.shape).copy()
    rates = np.broadcast_to(np.asarray(rates, dtype=float), strikes.shape).copy()
    return strikes, maturities, rates


def price(S, strikes, maturities, rates, q, params: HestonParams, option_type="call"):
    """Fourier prices from the C++ engine, vectorised over strikes/maturities."""
    core = require_cpp()
    K, T, r = _as_arrays(strikes, maturities, rates)
    return core.heston_price(S, K, T, r, q, *params.as_tuple(), option_type)


def mc_price(S, strikes, T, r, q, params: HestonParams, n_steps=100, n_paths=200_000, seed=12345, n_threads=0):
    """QE Monte Carlo prices from the C++ engine (dict with prices and standard errors)."""
    core = require_cpp()
    K = np.atleast_1d(np.asarray(strikes, dtype=float))
    return core.heston_mc(S, K, T, r, q, *params.as_tuple(), n_steps, n_paths, seed, n_threads)


def implied_vols(prices, S, strikes, maturities, rates, q, option_type="call"):
    """Black-Scholes implied volatilities computed by the C++ engine."""
    core = require_cpp()
    K, T, r = _as_arrays(strikes, maturities, rates)
    return core.bs_implied_vol(np.asarray(prices, dtype=float), S, K, T, r, q, option_type)


# --------------------------------------------------------------------------- reference


def cf_reference(u, T, r, q, params: HestonParams):
    """Characteristic function of ln(S_T/S_0) ("little Heston trap" form), NumPy."""
    v0, kappa, theta, sigma, rho = params.as_tuple()
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    a = kappa - rho * sigma * iu
    d = np.sqrt(a * a + sigma**2 * (iu + u * u))
    g = (a - d) / (a + d)
    e = np.exp(-d * T)
    C = (r - q) * iu * T + kappa * theta / sigma**2 * ((a - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = (a - d) / sigma**2 * (1.0 - e) / (1.0 - g * e)
    return np.exp(C + D * v0)


def call_price_reference(S, K, T, r, q, params: HestonParams):
    """Call price by adaptive quadrature (scipy.integrate.quad) of the same
    Fourier representation used in C++; slow but independent."""
    k = np.log(S / K)

    def integrand(u):
        value = np.exp(1j * u * k) * (S * cf_reference(u - 1j, T, r, q, params) - K * cf_reference(u, T, r, q, params))
        return (value / (1j * u)).real

    total = 0.0
    width = 5.0
    lower = 0.0
    while True:
        piece, _ = integrate.quad(integrand, lower, lower + width, epsabs=1e-15, epsrel=1e-13, limit=200)
        total += piece
        lower += width
        if abs(piece) < 1e-15 * max(S, K) or lower > 1e5:
            break
    return 0.5 * (S * np.exp(-q * T) - K * np.exp(-r * T)) + np.exp(-r * T) / np.pi * total


def mc_price_numpy(S, strikes, T, r, q, params: HestonParams, n_steps=100, n_paths=100_000, seed=12345):
    """Vectorised NumPy implementation of the same QE scheme (reference and
    speed benchmark). Uses NumPy's PCG64 generator, so its random numbers
    differ from the C++ engine; the two agree within Monte Carlo error."""
    v0, kappa, theta, sigma, rho = params.as_tuple()
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    E = np.exp(-kappa * dt)
    K0 = -rho * kappa * theta * dt / sigma
    K1 = 0.5 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = 0.5 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = 0.5 * dt * (1.0 - rho**2)
    K4 = K3
    ln_s = np.full(n_paths, np.log(S))
    v = np.full(n_paths, v0)
    for _ in range(n_steps):
        m = theta + (v - theta) * E
        s2 = v * sigma**2 * E * (1.0 - E) / kappa + theta * sigma**2 * (1.0 - E) ** 2 / (2.0 * kappa)
        psi = s2 / m**2
        quad = psi <= 1.5
        v_next = np.empty_like(v)
        inv = 2.0 / psi[quad]
        b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
        a = m[quad] / (1.0 + b2)
        v_next[quad] = a * (np.sqrt(b2) + rng.standard_normal(quad.sum())) ** 2
        exp_branch = ~quad
        p = (psi[exp_branch] - 1.0) / (psi[exp_branch] + 1.0)
        beta = (1.0 - p) / m[exp_branch]
        u = rng.random(exp_branch.sum())
        v_next[exp_branch] = np.where(u <= p, 0.0, np.log((1.0 - p) / np.maximum(1.0 - u, 1e-300)) / beta)
        ln_s += (r - q) * dt + K0 + K1 * v + K2 * v_next + np.sqrt(np.maximum(K3 * v + K4 * v_next, 0.0)) * rng.standard_normal(n_paths)
        v = v_next
    ST = np.exp(ln_s)
    K = np.atleast_1d(np.asarray(strikes, dtype=float))
    payoff = np.maximum(ST[:, None] - K[None, :], 0.0)
    disc = np.exp(-r * T)
    return {
        "call": disc * payoff.mean(axis=0),
        "call_se": disc * payoff.std(axis=0, ddof=1) / np.sqrt(n_paths),
    }


# --------------------------------------------------------------------------- calibration


@dataclass
class CalibrationResult:
    params: HestonParams
    rmse_vol: float          # root-mean-square implied volatility error (decimal)
    max_abs_vol_error: float
    n_evaluations: int
    success: bool
    message: str


def calibrate(S, strikes, maturities, rates, q, market_vols, initial: HestonParams,
              option_type="call", bounds=None) -> CalibrationResult:
    """Calibrate Heston parameters to implied volatilities.

    The objective is the vector of price errors divided by the Black-Scholes
    vega at the market volatility, a first-order approximation of implied
    volatility errors that avoids inverting prices at every iteration. The
    fit is reported in true implied-volatility terms.
    """
    from .black_scholes import bs_greeks, bs_price

    K, T, r = _as_arrays(strikes, maturities, rates)
    market_vols = np.asarray(market_vols, dtype=float)
    market_prices = bs_price(S, K, T, r, q, market_vols, option_type)
    vega = np.maximum(bs_greeks(S, K, T, r, q, market_vols, option_type)["vega"], 1e-8)
    if bounds is None:
        bounds = ([1e-4, 1e-2, 1e-4, 1e-2, -0.999], [1.0, 20.0, 1.0, 3.0, 0.999])

    def residuals(x):
        model = price(S, K, T, r, q, HestonParams(*x), option_type)
        return (model - market_prices) / vega

    fit = optimize.least_squares(residuals, np.array(initial.as_tuple()), bounds=bounds,
                                 x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
    params = HestonParams(*fit.x)
    model_vols = implied_vols(price(S, K, T, r, q, params, option_type), S, K, T, r, q, option_type)
    errors = model_vols - market_vols
    return CalibrationResult(params, float(np.sqrt(np.mean(errors**2))), float(np.max(np.abs(errors))),
                             int(fit.nfev), bool(fit.success), str(fit.message))
