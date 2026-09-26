"""VaR/ES formulas and backtests against known values and simulated data."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from quant_engine import risk


def test_normal_var_es_known_values():
    var, es = risk.normal_var_es(0.0, 1.0, 0.01)
    assert var == pytest.approx(2.326347874, abs=1e-9)
    _, es975 = risk.normal_var_es(0.0, 1.0, 0.025)
    assert es975 == pytest.approx(stats.norm.pdf(stats.norm.ppf(0.975)) / 0.025, rel=1e-12)
    assert es975 == pytest.approx(2.337802, abs=1e-6)


def test_student_t_es_by_integration():
    nu, alpha = 5.0, 0.025
    var, es = risk.student_t_var_es(0.0, 1.0, nu, alpha)
    scale = np.sqrt((nu - 2) / nu)
    tail, _ = integrate.quad(lambda x: -x * scale * stats.t.pdf(x, nu), -np.inf, stats.t.ppf(alpha, nu))
    assert es == pytest.approx(tail / alpha, rel=1e-8)
    assert var == pytest.approx(-scale * stats.t.ppf(alpha, nu), rel=1e-12)


def test_student_t_tends_to_normal():
    var_t, es_t = risk.student_t_var_es(0.1, 2.0, 1e7, 0.01)
    var_n, es_n = risk.normal_var_es(0.1, 2.0, 0.01)
    assert var_t == pytest.approx(var_n, rel=1e-5)
    assert es_t == pytest.approx(es_n, rel=1e-5)


def test_kupiec_known_case():
    hits = np.zeros(250, dtype=bool)
    hits[:5] = True
    res = risk.kupiec_pof(hits, 0.01)
    expected_lr = -2 * (245 * np.log(0.99) + 5 * np.log(0.01) - 245 * np.log(0.98) - 5 * np.log(0.02))
    assert res["lr"] == pytest.approx(expected_lr, rel=1e-12)
    assert risk.kupiec_pof(np.zeros(250, dtype=bool), 0.01)["lr"] == pytest.approx(-2 * 250 * np.log(0.99))


def test_christoffersen_detects_clustering():
    clustered = np.zeros(1000, dtype=bool)
    clustered[100:110] = True
    assert risk.christoffersen_independence(clustered)["p_value"] < 1e-6
    rng = np.random.default_rng(0)
    independent = rng.random(5000) < 0.02
    assert risk.christoffersen_independence(independent)["p_value"] > 0.01


def test_traffic_light_zones():
    assert [risk.basel_traffic_light(k) for k in (0, 4, 5, 9, 10)] == ["green", "green", "yellow", "yellow", "red"]


def test_z2_is_centred_under_correct_model():
    rng = np.random.default_rng(1)
    losses = rng.standard_normal(400_000)
    var, es = risk.normal_var_es(0.0, 1.0, 0.025)
    z2 = risk.acerbi_szekely_z2(losses, np.full_like(losses, var), np.full_like(losses, es), 0.025)
    assert abs(z2) < 0.03  # standard deviation of Z2 is about 0.012 here
    too_low = risk.acerbi_szekely_z2(1.5 * losses, np.full_like(losses, var), np.full_like(losses, es), 0.025)
    assert too_low > 0.7


def test_historical_var_uses_past_window():
    r = pd.Series(np.arange(10.0) - 5.0)
    out = risk.historical_var_es(r, window=5, alpha=0.2)
    assert out.index[0] == 5
    assert out["var"].iloc[0] == pytest.approx(-np.quantile(r.iloc[:5], 0.2))
