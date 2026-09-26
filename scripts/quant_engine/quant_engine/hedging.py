"""Dealer's view of an option sale: discrete delta hedging, costs and pricing.

The C++ engine simulates the profit and loss of a short European option that
is delta-hedged in discrete time (see cpp/hedging.hpp). This module wraps it
and adds the risk statistics used to set a price: a desk that sells an option
needs a premium that covers the expected hedging cost plus a buffer for the
tail of the P&L distribution.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import require_cpp


def garman_kohlhagen(S, K, T, r_dom, r_for, sigma, option_type="call"):
    """Price and Greeks of a currency option (Black-Scholes with the foreign rate as yield)."""
    return require_cpp().bs_price_greeks(S, K, T, r_dom, r_for, sigma, option_type)


def simulate(S0, K, T, r_dom, r_for, sigma_price, sigma_hedge, option_type="call", n_steps=126, rebalance_every=1,
             cost_rate=0.0, dynamics="gbm", mu=0.0, sigma=0.1, heston=None, garch=None, residuals=None,
             n_paths=100_000, seed=12345, n_threads=0) -> dict:
    """P&L of one short option, per unit of underlying notional, discounted to today.

    dynamics: "gbm" (mu, sigma), "heston" (mu, heston=(v0, kappa, theta, sigma, rho)) or
    "garch_fhs" (garch=(mu, omega, alpha, gamma, beta, var0) in percent units and the
    standardised residuals to resample).
    """
    core = require_cpp()
    return core.hedge_simulation(S0, K, T, r_dom, r_for, option_type, sigma_price, sigma_hedge, n_steps, rebalance_every,
                                 cost_rate, dynamics, mu, sigma, np.asarray(heston if heston is not None else [], float),
                                 np.asarray(garch if garch is not None else [], float),
                                 np.asarray(residuals if residuals is not None else [], float), n_paths, seed, n_threads)


def pnl_summary(pnl, premium=None, scale=1.0) -> dict:
    """Mean, dispersion and tail of a P&L sample (losses are negative P&L)."""
    x = np.asarray(pnl, dtype=float) * scale
    q01, q05 = np.quantile(x, [0.01, 0.05])
    es975 = -x[x <= np.quantile(x, 0.025)].mean()
    out = {"mean": x.mean(), "std. dev.": x.std(ddof=1), "MC s.e.": x.std(ddof=1) / math.sqrt(x.size),
           "P(loss)": float(np.mean(x < 0)), "VaR 95%": -q05, "VaR 99%": -q01, "ES 97.5%": es975}
    if premium is not None:
        out["premium"] = premium * scale
    return out


def required_markup(pnl, quantile=0.05):
    """Extra premium (same units as pnl, at time 0) so that P(P&L < 0) equals `quantile`."""
    return float(max(-np.quantile(np.asarray(pnl, dtype=float), quantile), 0.0))


def derman_kamal_std(S, K, T, r_dom, r_for, sigma, n_rebalances):
    """Approximate standard deviation of the hedging error with n rebalances
    (Kamal and Derman, 1999): sqrt(pi / 4) * vega * sigma / sqrt(n)."""
    vega = garman_kohlhagen(S, K, T, r_dom, r_for, sigma)["vega"]
    return math.sqrt(math.pi / 4.0) * vega * sigma / math.sqrt(n_rebalances)


def comparison_table(results: dict, scale=1.0) -> pd.DataFrame:
    """Summary table for a dict {label: simulate(...) output}."""
    rows = {label: {**pnl_summary(out["pnl"], out["premium"], scale), "mean costs": float(np.mean(out["costs"])) * scale}
            for label, out in results.items()}
    return pd.DataFrame(rows).T
