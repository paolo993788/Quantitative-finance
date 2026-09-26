"""Black-Scholes-Merton reference implementations in NumPy/SciPy.

These functions are independent of the C++ extension and serve as the
benchmark against which the compiled engines are validated. Rates and the
dividend yield are continuously compounded; maturities are in years.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def _d1_d2(S, K, T, r, q, sigma):
    sd = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sd
    return d1, d1 - sd


def bs_price(S, K, T, r, q, sigma, option_type="call"):
    """European option price under Black-Scholes-Merton (vectorised)."""
    S, K, T, r, q, sigma = np.broadcast_arrays(*map(np.asarray, (S, K, T, r, q, sigma)))
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    if option_type == "put":
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def bs_greeks(S, K, T, r, q, sigma, option_type="call"):
    """Delta, gamma, vega, theta (per year) and rho of a European option."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    pdf = stats.norm.pdf(d1)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    gamma = dq * pdf / (S * sigma * np.sqrt(T))
    vega = S * dq * pdf * np.sqrt(T)
    theta_common = -S * dq * pdf * sigma / (2.0 * np.sqrt(T))
    if option_type == "call":
        delta = dq * stats.norm.cdf(d1)
        theta = theta_common - r * K * dr * stats.norm.cdf(d2) + q * S * dq * stats.norm.cdf(d1)
        rho = K * T * dr * stats.norm.cdf(d2)
    else:
        delta = -dq * stats.norm.cdf(-d1)
        theta = theta_common + r * K * dr * stats.norm.cdf(-d2) - q * S * dq * stats.norm.cdf(-d1)
        rho = -K * T * dr * stats.norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(price, S, K, T, r, q, option_type="call"):
    """Implied volatility of a single price by Brent's method (reference)."""

    def objective(sigma):
        return float(bs_price(S, K, T, r, q, sigma, option_type)) - price

    return optimize.brentq(objective, 1e-9, 10.0, xtol=1e-14, rtol=1e-14, maxiter=500)


def crr_price(S, K, T, r, q, sigma, n_steps=5000, option_type="put", american=True):
    """Cox-Ross-Rubinstein binomial tree price, European or American.

    The tree converges to the Black-Scholes value at rate O(1/n); it is used
    as an independent benchmark for the finite-difference solver.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(n_steps + 1)
    spot = S * u ** (n_steps - j) * d**j
    sign = 1.0 if option_type == "call" else -1.0
    values = np.maximum(sign * (spot - K), 0.0)
    for step in range(n_steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        if american:
            spot = spot[:-1] * d
            values = np.maximum(values, sign * (spot - K))
    return float(values[0])
