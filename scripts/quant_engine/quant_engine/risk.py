"""Value at Risk and Expected Shortfall: parametric formulas and backtests.

Conventions: returns are profits (positive = gain); VaR and ES are reported
as positive losses for a tail probability `alpha` (for example 0.01 for a
99% VaR). An exception ("hit") occurs on day t when the loss -r_t exceeds
the VaR forecast made at t - 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def normal_var_es(mu, sigma, alpha):
    """VaR and ES of a Gaussian return with mean mu and volatility sigma."""
    z = stats.norm.ppf(alpha)
    var = -(mu + sigma * z)
    es = -mu + sigma * stats.norm.pdf(z) / alpha
    return var, es


def student_t_var_es(mu, sigma, nu, alpha):
    """VaR and ES when (r - mu) / sigma is a unit-variance Student-t with nu
    degrees of freedom (McNeil, Frey and Embrechts, 2015, Example 2.15)."""
    nu = np.asarray(nu, dtype=float)
    t = stats.t.ppf(alpha, nu)
    scale = np.sqrt((nu - 2.0) / nu)
    var = -(mu + sigma * scale * t)
    es_std = stats.t.pdf(t, nu) / alpha * (nu + t**2) / (nu - 1.0)
    es = -mu + sigma * scale * es_std
    return var, es


def historical_var_es(returns: pd.Series, window: int, alpha: float) -> pd.DataFrame:
    """Rolling historical-simulation VaR/ES; the row for t uses t-window..t-1."""
    r = returns.to_numpy(dtype=float)
    var = np.full(len(r), np.nan)
    es = np.full(len(r), np.nan)
    for t in range(window, len(r)):
        sample = r[t - window:t]
        q = np.quantile(sample, alpha)
        var[t] = -q
        es[t] = -sample[sample <= q].mean()
    return pd.DataFrame({"var": var, "es": es}, index=returns.index).iloc[window:]


def kupiec_pof(hits, alpha):
    """Kupiec (1995) proportion-of-failures likelihood-ratio test."""
    hits = np.asarray(hits, dtype=bool)
    n, x = hits.size, int(hits.sum())
    pi_hat = x / n
    log_l0 = (n - x) * np.log1p(-alpha) + x * np.log(alpha)
    log_l1 = (n - x) * np.log1p(-pi_hat) + x * np.log(pi_hat) if 0 < x < n else 0.0
    lr = -2.0 * (log_l0 - log_l1)
    return {"exceptions": x, "expected": alpha * n, "rate": pi_hat, "lr": lr, "p_value": stats.chi2.sf(lr, 1)}


def christoffersen_independence(hits):
    """Christoffersen (1998) test of independence of exceptions (first-order Markov)."""
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))

    def xlogy(x, y):
        return x * np.log(y) if x > 0 else 0.0

    pi0 = n01 / (n00 + n01) if n00 + n01 > 0 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    log_l0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_l1 = xlogy(n00, 1 - pi0) + xlogy(n01, pi0) + xlogy(n10, 1 - pi1) + xlogy(n11, pi1)
    lr = -2.0 * (log_l0 - log_l1)
    return {"pi01": pi0, "pi11": pi1, "lr": lr, "p_value": stats.chi2.sf(lr, 1)}


def conditional_coverage(hits, alpha):
    """Christoffersen (1998) joint test of coverage and independence (2 d.o.f.)."""
    pof = kupiec_pof(hits, alpha)
    ind = christoffersen_independence(hits)
    lr = pof["lr"] + ind["lr"]
    return {"lr": lr, "p_value": stats.chi2.sf(lr, 2)}


def basel_traffic_light(exceptions_250: int) -> str:
    """Basel Committee (1996) zones for 250 daily 99% VaR observations."""
    if exceptions_250 <= 4:
        return "green"
    if exceptions_250 <= 9:
        return "yellow"
    return "red"


def acerbi_szekely_z2(losses, var, es, alpha):
    """Acerbi and Szekely (2014) Z2 statistic for Expected Shortfall.

    Z2 = sum_t L_t 1{L_t > VaR_t} / (T alpha ES_t) - 1. Its expectation is 0
    when ES is correctly forecast; positive values signal underestimated
    risk. The authors report approximate 5% and 0.01% critical values of 0.70
    and 1.80 that hold across a wide range of distributions.
    """
    losses, var, es = map(lambda x: np.asarray(x, dtype=float), (losses, var, es))
    hits = losses > var
    return float(np.sum(losses * hits / es) / (losses.size * alpha) - 1.0)


def backtest_table(returns: pd.Series, forecasts: dict, alpha: float, es_forecasts: dict | None = None) -> pd.DataFrame:
    """Summary of VaR backtests for several models.

    `forecasts` maps model names to VaR series aligned with `returns`;
    `es_forecasts` optionally maps the same names to ES series.
    """
    rows = {}
    losses = -returns
    for name, var in forecasts.items():
        aligned = pd.concat([losses, var], axis=1, join="inner").dropna()
        hits = aligned.iloc[:, 0] > aligned.iloc[:, 1]
        pof = kupiec_pof(hits, alpha)
        ind = christoffersen_independence(hits)
        cc = conditional_coverage(hits, alpha)
        row = {"observations": len(hits), "exceptions": pof["exceptions"], "expected": pof["expected"],
               "exception rate": pof["rate"], "Kupiec p": pof["p_value"],
               "independence p": ind["p_value"], "cond. coverage p": cc["p_value"]}
        if es_forecasts is not None and name in es_forecasts:
            es = es_forecasts[name].reindex(aligned.index)
            row["ES Z2"] = acerbi_szekely_z2(aligned.iloc[:, 0], aligned.iloc[:, 1], es, alpha)
        rows[name] = row
    return pd.DataFrame(rows).T
