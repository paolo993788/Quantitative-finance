"""Yield-curve models and IRRBB: limiting cases, exact likelihood, recovery of known parameters, no look-ahead."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quant_engine import term_structure as ts
from quant_engine.curves import SvenssonCurve

MATURITIES = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])


def test_nelson_siegel_loadings_limits_and_curvature_peak():
    L = ts.ns_loadings([1e-8, 1e4], ts.DIEBOLD_LI_LAMBDA)
    np.testing.assert_allclose(L[0], [1.0, 1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(L[1], [1.0, 0.0, 0.0], atol=1e-3)
    grid = np.linspace(0.1, 10, 100_000)
    peak = grid[np.argmax(ts.ns_loadings(grid)[:, 2])]
    assert peak * 12 == pytest.approx(29.4, abs=0.1)   # about 30 months (Diebold and Li, 2006)


def test_cross_sectional_fit_recovers_exact_factors():
    rng = np.random.default_rng(1)
    beta = rng.normal([4, -2, 1], 1.0, size=(50, 3))
    Y = pd.DataFrame(beta @ ts.ns_loadings(MATURITIES).T, columns=MATURITIES)
    np.testing.assert_allclose(ts.fit_ns_cross_section(Y).to_numpy(), beta, atol=1e-10)


def test_principal_components_are_orthonormal_with_sign_conventions():
    rng = np.random.default_rng(2)
    changes = pd.DataFrame(rng.standard_normal((300, 3)) @ (ts.ns_loadings(MATURITIES) * [0.3, 0.15, 0.08]).T
                           + 0.01 * rng.standard_normal((300, MATURITIES.size)), columns=MATURITIES)
    pca = ts.principal_components(changes)
    V = pca["loadings"].to_numpy()
    np.testing.assert_allclose(V.T @ V, np.eye(3), atol=1e-12)
    assert pca["explained"].sum() == pytest.approx(1.0)
    assert pca["explained"].iloc[:3].sum() > 0.99
    assert V[:, 0].mean() > 0 and V[-1, 1] > V[0, 1] and V[MATURITIES.size // 2, 2] > 0


def _small_model():
    A = np.array([[0.95, 0.02, 0.0], [0.0, 0.9, 0.05], [0.0, 0.0, 0.8]])
    Q = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
    return ts.DNSModel(0.6, np.array([3.0, -1.0, 0.5]), A, Q, np.array([0.05, 0.03, 0.04]), np.array([1.0, 5.0, 10.0]))


def test_kalman_loglikelihood_equals_the_joint_gaussian_density():
    model = _small_model()
    T = 6
    rng = np.random.default_rng(3)
    Y = rng.normal(2.0, 1.0, size=(T, 3))
    L = ts.ns_loadings(model.maturities, model.lam)
    P = np.linalg.solve(np.eye(9) - np.kron(model.A, model.A), model.Q.ravel()).reshape(3, 3)
    cov_f = np.zeros((3 * T, 3 * T))
    for s in range(T):
        for t in range(T):
            block = np.linalg.matrix_power(model.A, t - s) @ P if t >= s else P @ np.linalg.matrix_power(model.A, s - t).T
            cov_f[3 * t:3 * t + 3, 3 * s:3 * s + 3] = block
    big_L = np.kron(np.eye(T), L)
    cov_y = big_L @ cov_f @ big_L.T + np.kron(np.eye(T), np.diag(model.h ** 2))
    mean_y = np.tile(L @ model.mu, T)
    exact = stats.multivariate_normal(mean_y, cov_y).logpdf(Y.ravel())
    assert ts.kalman_filter(model, Y, engine="numpy")["loglik"] == pytest.approx(exact, rel=1e-10)
    assert ts.kalman_filter(model, Y, engine="cpp")["loglik"] == pytest.approx(exact, rel=1e-10)


def test_cpp_kalman_filter_matches_numpy_with_missing_values():
    model = _small_model()
    rng = np.random.default_rng(7)
    Y = rng.normal(2.0, 1.0, size=(40, 3))
    Y[5, 1] = np.nan
    Y[12, :] = np.nan
    ref = ts.kalman_filter(model, Y, engine="numpy")
    fast = ts.kalman_filter(model, Y, engine="cpp")
    assert fast["loglik"] == pytest.approx(ref["loglik"], rel=1e-11)
    np.testing.assert_allclose(fast["filtered"], ref["filtered"], atol=1e-10)
    np.testing.assert_allclose(fast["predicted"], ref["predicted"], atol=1e-10)
    np.testing.assert_allclose(fast["filtered"][12], fast["predicted"][12])   # no update without data


def test_parameter_packing_round_trip():
    model = _small_model()
    back = ts.DNSModel.unpack(model.pack(), model.maturities)
    for name in ("mu", "A", "Q", "h"):
        np.testing.assert_allclose(getattr(back, name), getattr(model, name), rtol=1e-12)
    assert back.lam == pytest.approx(model.lam)
    model.h = np.full(3, 0.04)
    model.common_h = True
    theta = model.pack()
    assert theta.size == 20
    back = ts.DNSModel.unpack(theta, model.maturities)
    assert back.common_h and np.allclose(back.h, 0.04)
    assert len(ts.parameter_names(model.maturities, True)) == 20


def test_maximum_likelihood_recovers_simulated_parameters():
    rng = np.random.default_rng(4)
    T, lam = 500, 0.7
    A = np.diag([0.98, 0.95, 0.9])
    mu = np.array([3.0, -1.5, 0.5])
    C = np.diag([0.15, 0.25, 0.4])
    L = ts.ns_loadings(MATURITIES, lam)
    f = np.zeros((T, 3))
    f[0] = mu
    for t in range(1, T):
        f[t] = mu + A @ (f[t - 1] - mu) + C @ rng.standard_normal(3)
    Y = pd.DataFrame(f @ L.T + 0.03 * rng.standard_normal((T, MATURITIES.size)), columns=MATURITIES)
    model = ts.fit_dns_kalman(Y, lam0=0.5)
    assert model.lam == pytest.approx(lam, rel=0.05)
    np.testing.assert_allclose(np.diag(model.A), np.diag(A), atol=0.05)   # sampling error with 500 months
    np.testing.assert_allclose(model.h, 0.03, rtol=0.2)
    se = ts.dns_standard_errors(model, Y)
    assert np.all(np.isfinite(se.to_numpy())) and np.all(se.to_numpy() > 0)
    assert abs(math.log(model.lam) - math.log(lam)) < 4 * se["log lambda"]
    for i in range(3):
        name = ["level", "slope", "curvature"][i]
        assert abs(model.A[i, i] - A[i, i]) < 4 * se[f"A {name}<-{name}"]
    assert model.converged
    common = ts.fit_dns_kalman(Y, lam0=0.5, common_h=True)   # true measurement errors are equal across maturities
    assert common.converged and common.common_h
    assert common.h[0] == pytest.approx(0.03, rel=0.1)
    assert common.lam == pytest.approx(lam, rel=0.05)


def test_recursive_forecasts_use_only_past_data():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2000-01-31", periods=160, freq="ME")
    base = rng.normal([3, -1, 0.5], 0.3, size=(160, 3)).cumsum(axis=0) * 0.1 + [3, -1, 0.5]
    Y = pd.DataFrame(base @ ts.ns_loadings(MATURITIES).T, index=idx, columns=MATURITIES)
    out = ts.recursive_forecasts(Y, "2006-01-31", horizons=(1, 6))
    shocked = Y.copy()
    shocked.loc["2010-01-31":] += 2.0
    out2 = ts.recursive_forecasts(shocked, "2006-01-31", horizons=(1, 6))
    for h in (1, 6):
        for model in out[h]:
            before = out[h][model].index < pd.Timestamp("2010-01-31")
            pd.testing.assert_frame_equal(out[h][model].loc[before], out2[h][model].loc[before])


def test_diebold_mariano_statistic_matches_the_formula():
    rng = np.random.default_rng(6)
    e1, e2 = rng.standard_normal(200), 1.3 * rng.standard_normal(200)
    d = e1 ** 2 - e2 ** 2
    n = d.size
    manual = math.sqrt((n - 1) / n) * d.mean() / math.sqrt(d.var() / n)   # horizon 1: no autocovariance terms
    res = ts.diebold_mariano(e1, e2, horizon=1)
    assert res["statistic"] == pytest.approx(manual, rel=1e-12)
    assert res["statistic"] < 0
    assert math.isnan(ts.diebold_mariano(e1, e1)["statistic"])


def test_bcbs_scenarios_and_floor():
    t = np.array([0.0, 4.0, 1000.0])
    sc = ts.bcbs_scenarios(t)
    assert sc["parallel up"][0] == pytest.approx(0.02) and sc["parallel down"][2] == pytest.approx(-0.02)
    assert sc["short rates up"][0] == pytest.approx(0.025) and sc["short rates up"][1] == pytest.approx(0.025 / math.e)
    assert sc["steepener"][0] == pytest.approx(-0.65 * 0.025) and sc["steepener"][2] == pytest.approx(0.9 * 0.01)
    assert sc["flattener"][0] == pytest.approx(0.8 * 0.025) and sc["flattener"][2] == pytest.approx(-0.6 * 0.01)
    np.testing.assert_allclose(ts.eba_floor([0.0, 25.0, 50.0, 60.0]), [-0.015, -0.0075, 0.0, 0.0])
    np.testing.assert_allclose(ts.eba_floor([0.0], base_rates=[-0.02]), [-0.02])


def test_delta_eve_of_a_zero_coupon_and_floor():
    base = np.array([0.0])
    res = ts.delta_eve([10.0], [100.0], base, {"up": [0.02], "down": [-0.02]})
    assert res["up"] == pytest.approx(100 * (math.exp(-0.02 * 10) - 1), rel=1e-12)
    floored = -0.015 + 0.0003 * 10    # -1.2% at 10 years: the -2% shock is floored
    assert res["down"] == pytest.approx(100 * (math.exp(-floored * 10) - 1), rel=1e-12)
    unfloored = ts.delta_eve([10.0], [100.0], base, {"down": [-0.02]}, floor=False)
    assert unfloored["down"] == pytest.approx(100 * (math.exp(0.02 * 10) - 1), rel=1e-12)


def test_cash_flow_generators():
    times, amounts = ts.annuity_cashflows(1000.0, 0.03, 20, freq=12)
    assert np.sum(amounts / (1 + 0.03 / 12) ** np.arange(1, 241)) == pytest.approx(1000.0, rel=1e-12)
    times, amounts = ts.bullet_cashflows(100.0, 0.04, 5, freq=1)
    assert list(times) == [1, 2, 3, 4, 5] and amounts[-1] == pytest.approx(104.0)


def test_mortgage_cash_flows_with_prepayment():
    t0, a0 = ts.annuity_cashflows(100.0, 0.036, 25, freq=12)
    t, a, principal = ts.mortgage_cashflows(100.0, 0.036, 25, cpr=0.0)
    np.testing.assert_allclose(a, a0, rtol=1e-12)
    for cpr in (0.05, 0.2):
        t, a, principal = ts.mortgage_cashflows(100.0, 0.036, 25, cpr=cpr)
        assert principal.sum() == pytest.approx(100.0, rel=1e-12)
        # prepayment at par: discounting at the loan rate returns the notional whatever the CPR
        assert np.sum(a / (1 + 0.036 / 12) ** np.arange(1, a.size + 1)) == pytest.approx(100.0, rel=1e-12)
    faster = ts.mortgage_cashflows(100.0, 0.036, 25, cpr=0.2)
    assert np.sum(faster[0] * faster[2]) < np.sum(t0 * ts.mortgage_cashflows(100.0, 0.036, 25)[2])


def test_deposit_slotting_swap_and_nii():
    t, a = ts.non_maturity_deposit_cashflows(100.0, 0.7, 5)
    assert a.sum() == pytest.approx(100.0) and a[0] == pytest.approx(30.0)
    assert np.sum(t[1:] * a[1:]) / 70.0 == pytest.approx(2.5 + 1 / 24)
    curve = SvenssonCurve(2.6, -0.6, -1.2, 1.5, 1.8, 12.0)
    k = ts.par_swap_rate(curve.zero_rate, 10)
    times, amounts = ts.payer_swap_cashflows(50.0, k, 10, curve.zero_rate)
    assert ts.present_value(times, amounts, curve.zero_rate(times)) == pytest.approx(0.0, abs=1e-10)
    up = ts.present_value(times, amounts, curve.zero_rate(times) + 0.01)
    assert up > 0    # a payer swap gains when rates rise
    shock = lambda x: np.full_like(np.asarray(x, dtype=float), 0.02)
    assert ts.delta_nii([0.25, 0.0, 2.0], [100.0, -50.0, 30.0], [1.0, 0.5, 1.0], shock) == pytest.approx(
        100 * 0.02 * 0.75 - 50 * 0.5 * 0.02 * 1.0)


def test_delta_eve_book_matches_fixed_cash_flows():
    curve = SvenssonCurve(2.6, -0.6, -1.2, 1.5, 1.8, 12.0)
    times, amounts = ts.bullet_cashflows(100.0, 0.03, 7)
    book = ts.delta_eve_book(lambda name: (times, amounts), curve.zero_rate)
    fixed = ts.delta_eve(times, amounts, curve.zero_rate(times), ts.bcbs_scenarios(times))
    pd.testing.assert_series_equal(book, fixed)


def test_recursive_kalman_forecasts_use_only_past_data():
    rng = np.random.default_rng(8)
    idx = pd.date_range("2000-01-31", periods=110, freq="ME")
    f = np.zeros((110, 3))
    f[0] = [3.0, -1.0, 0.5]
    for t in range(1, 110):
        f[t] = [3.0, -1.0, 0.5] + 0.95 * (f[t - 1] - [3.0, -1.0, 0.5]) + rng.normal(0, [0.15, 0.2, 0.3])
    Y = pd.DataFrame(f @ ts.ns_loadings(MATURITIES).T + 0.03 * rng.standard_normal((110, 11)), index=idx,
                     columns=MATURITIES)
    out = ts.recursive_kalman_forecasts(Y, "2005-01-31", horizons=(1, 3), refit_every=24)
    shocked = Y.copy()
    shocked.loc["2007-06-30":] += 1.5
    out2 = ts.recursive_kalman_forecasts(shocked, "2005-01-31", horizons=(1, 3), refit_every=24)
    for h in (1, 3):
        before = out[h].index < pd.Timestamp("2007-06-30")
        pd.testing.assert_frame_equal(out[h].loc[before], out2[h].loc[before])
        assert (out[h].index > out[h].index.min()).sum() == len(out[h]) - 1
    assert out["fits"]["converged"].all()
