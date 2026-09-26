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
    ref = garch.fit_reference(r, model, dist)
    # Both reach the same maximum: log-likelihoods within 1e-4 (the
    # reference stops at ftol = 1e-12 on a 2000-term sum).
    assert cpp.loglik >= ref.loglik - 1e-4
    for name in ("mu", "omega", "alpha", "beta"):
        assert cpp.params[name] == pytest.approx(ref.params[name], rel=2e-2, abs=2e-3)


def test_parameter_recovery(simulated):
    fit = garch.fit(simulated, "gjr", "t")
    for name, value in TRUE.items():
        # Within four asymptotic standard errors of the true value.
        assert abs(fit.params[name] - value) < 4 * fit.std_errors[name], name
    assert fit.persistence < 1


def test_rolling_forecast_uses_only_past_data(simulated):
    r = pd.Series(simulated[:1400], index=pd.bdate_range("2010-01-01", periods=1400))
    out = garch.rolling_forecast(r, window=1000, refit_every=100, alphas=(0.01,))
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
