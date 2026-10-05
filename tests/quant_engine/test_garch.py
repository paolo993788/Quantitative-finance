"""GARCH engine: likelihood, estimation, parameter recovery and rolling forecasts."""

import numpy as np
import pandas as pd
import pytest

from quant_engine import garch

TRUE = {"mu": 0.02, "omega": 0.02, "alpha": 0.04, "gamma": 0.08, "beta": 0.90, "nu": 6.0}


@pytest.fixture(scope="module")
def simulated():
    return garch.simulate(TRUE, 20_000, seed=11)


@pytest.mark.parametrize("model,dist", [("gjr", "t"), ("garch", "normal"), ("gjr", "normal"), ("garch", "t")])
def test_cpp_likelihood_matches_python(simulated, model, dist):
    r = simulated[:3000]
    assert garch.nll(r, TRUE, model, dist) == pytest.approx(garch.nll_reference(r, TRUE, model, dist), rel=1e-11)


@pytest.mark.parametrize("model,dist", [("gjr", "t"), ("garch", "normal")])
def test_nelder_mead_matches_scipy_reference(simulated, model, dist):
    r = simulated[:2000]
    cpp = garch.fit(r, model, dist, std_errors=False)
    assert cpp.converged
    ref = garch.fit_reference(r, model, dist)
    # Both reach the same maximum: log-likelihoods within 1e-4 (the
    # reference stops at ftol = 1e-12 on a 2000-term sum).
    assert cpp.loglik >= ref.loglik - 1e-4
    for name in ("mu", "omega", "alpha", "beta"):
        assert cpp.params[name] == pytest.approx(ref.params[name], rel=2e-2, abs=2e-3)


def test_parameter_recovery(simulated):
    fit = garch.fit(simulated, "gjr", "t")
    assert fit.converged
    for name, value in TRUE.items():
        # Within four asymptotic standard errors of the true value.
        assert abs(fit.params[name] - value) < 4 * fit.std_errors[name], name
    assert fit.persistence < 1


def test_rolling_forecast_uses_only_past_data(simulated):
    r = pd.Series(simulated[:1400], index=pd.bdate_range("2010-01-01", periods=1400))
    out = garch.rolling_forecast(r, window=1000, refit_every=100, alphas=(0.01,))
    assert out.attrs["converged"].all() and len(out.attrs["converged"]) == 4   # one fit per refit
    # First forecast: parameters fitted on the first window, variance filtered to its end.
    first_fit = garch.fit(r.iloc[:1000], std_errors=False)
    s2 = garch.conditional_variance(r.iloc[:1000], first_fit.params)
    assert out["sigma"].iloc[0] == pytest.approx(np.sqrt(s2[-1]), rel=1e-10)
    # Changing a future return must not change earlier forecasts.
    shocked = r.copy()
    shocked.iloc[1300] += 25.0
    out2 = garch.rolling_forecast(shocked, window=1000, refit_every=100, alphas=(0.01,))
    pd.testing.assert_series_equal(out["sigma"].iloc[:301], out2["sigma"].iloc[:301])
    assert out["sigma"].iloc[301] != out2["sigma"].iloc[301]


def test_half_life_and_news_impact_curve(simulated):
    fit = garch.fit(simulated[:3000], "gjr", "t", std_errors=False)
    assert fit.half_life == pytest.approx(np.log(0.5) / np.log(fit.persistence))
    assert fit.persistence ** fit.half_life == pytest.approx(0.5)
    p = fit.params
    uncond = p["omega"] / (1 - fit.persistence)
    shocks = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    curve = garch.news_impact_curve(fit, shocks)
    manual = [p["omega"] + (p["alpha"] + (p["gamma"] if e < 0 else 0.0)) * e ** 2 + p["beta"] * uncond for e in shocks]
    np.testing.assert_allclose(curve, manual, rtol=1e-12)
    assert curve[0] > curve[-1] and curve[1] > curve[3]   # leverage: negative shocks raise the variance more
    assert curve[2] == pytest.approx(p["omega"] + p["beta"] * uncond)
    symmetric = garch.fit(simulated[:3000], "garch", "normal", std_errors=False)
    np.testing.assert_allclose(garch.news_impact_curve(symmetric, shocks), garch.news_impact_curve(symmetric, -shocks))
