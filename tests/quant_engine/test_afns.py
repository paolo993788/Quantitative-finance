"""Arbitrage-free Nelson-Siegel model: the yield adjustment against numerical integration and Monte Carlo under
the risk-neutral dynamics, the exact discretisation, the likelihood against the joint Gaussian density, parameter
recovery and no look-ahead in the recursive forecasts."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from quant_engine import afns, term_structure as ts

MATURITIES = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])


def _numerical_adjustment(tau, lam, sigma):
    s1, s2, s3 = np.asarray(sigma) / 100

    def B2(s):
        return -np.expm1(-lam * s) / lam

    def integrand(s):
        return s1**2 * s**2 + s2**2 * B2(s) ** 2 + s3**2 * (B2(s) - s * np.exp(-lam * s)) ** 2

    value, _ = integrate.quad(integrand, 0, tau, epsabs=1e-16, epsrel=1e-13, limit=200)
    return -100 * value / (2 * tau)


@pytest.mark.parametrize("lam,sigma", [(0.5, (0.6, 1.0, 1.5)), (0.2, (0.3, 2.0, 0.5)), (1.5, (1.0, 0.1, 3.0))])
def test_closed_form_adjustment_equals_the_convexity_integral(lam, sigma):
    closed = afns.yield_adjustment(MATURITIES, lam, sigma)
    numerical = np.array([_numerical_adjustment(t, lam, sigma) for t in MATURITIES])
    np.testing.assert_allclose(closed, numerical, rtol=1e-10)
    assert np.all(closed < 0) and np.all(np.diff(closed) < 0)          # convexity lowers yields, more at long maturities
    np.testing.assert_allclose(afns.yield_adjustment(MATURITIES, lam, (0, 0, 0)), 0.0, atol=0)
    assert abs(afns.yield_adjustment([1e-8], lam, sigma)[0]) < 1e-12        # vanishes at zero maturity


def test_adjustment_is_the_risk_neutral_convexity_by_monte_carlo():
    # Simulate the risk-neutral dynamics dX = -K^Q X dt + Sigma dW (theta^Q = 0 without loss of generality) with
    # r = X1 + X2 and compare -log E[exp(-int r)] / tau - E[int r] / tau with the adjustment. Euler steps of
    # 0.005 years; 20,000 paths give a standard error of about 1 bp at 10 years, against an adjustment of 23 bp.
    lam, sigma, tau, dt = 0.4, np.array([0.8, 1.2, 1.6]), 10.0, 0.005
    rng = np.random.default_rng(11)
    n = 20_000
    x = np.zeros((n, 3))
    integral = np.zeros(n)
    s = sigma / 100
    for _ in range(int(tau / dt)):
        integral += (x[:, 0] + x[:, 1]) * dt
        drift = np.column_stack([np.zeros(n), -lam * x[:, 1] + lam * x[:, 2], -lam * x[:, 2]])
        x = x + drift * dt + rng.standard_normal((n, 3)) * s * np.sqrt(dt)
    centred = integral - integral.mean()
    estimate = -100 * np.log(np.mean(np.exp(-centred))) / tau
    se = 100 * np.std(np.exp(-centred)) / np.mean(np.exp(-centred)) / np.sqrt(n) / tau
    exact = afns.yield_adjustment([tau], lam, sigma)[0]
    assert abs(estimate - exact) < 4 * se + 0.002          # plus the O(dt) bias of the Euler scheme (about 0.1 bp)
    assert exact == pytest.approx(-100 * np.var(integral) / (2 * tau), rel=0.05)


def _model(**kw):
    args = dict(lam=0.5, theta=np.array([3.0, -1.5, -1.0]), kappa=np.array([0.1, 0.4, 1.2]),
                sigma=np.array([0.5, 0.8, 1.5]), h=np.full(MATURITIES.size, 0.05), maturities=MATURITIES)
    args.update(kw)
    return afns.AFNSModel(**args)


def test_exact_discretisation_and_round_trip():
    model = _model()
    A, Q = model.transition()
    for i in range(3):
        k, s = model.kappa[i], model.sigma[i]
        variance, _ = integrate.quad(lambda u: s**2 * np.exp(-2 * k * u), 0, model.dt, epsabs=1e-16)
        assert Q[i, i] == pytest.approx(variance, rel=1e-12) and A[i, i] == pytest.approx(np.exp(-k / 12), rel=1e-15)
    back = afns.AFNSModel.unpack(model.pack(), MATURITIES)
    for name in ("theta", "kappa", "sigma", "h"):
        np.testing.assert_allclose(getattr(back, name), getattr(model, name), rtol=1e-12)
    assert back.lam == pytest.approx(model.lam) and back.common_h
    # The unconditional state covariance of the filter is sigma^2 / (2 kappa).
    P0 = ts._initial_state_covariance(A, Q)
    np.testing.assert_allclose(np.diag(P0), model.sigma**2 / (2 * model.kappa), rtol=1e-12)


def _simulate(model, T, seed):
    rng = np.random.default_rng(seed)
    A, Q = model.transition()
    L = ts.ns_loadings(MATURITIES, model.lam)
    x = model.theta + rng.standard_normal(3) * np.sqrt(model.sigma**2 / (2 * model.kappa))
    rows = []
    for _ in range(T):
        x = model.theta + A @ (x - model.theta) + rng.standard_normal(3) * np.sqrt(np.diag(Q))
        rows.append(L @ x + model.adjustment() + model.h * rng.standard_normal(MATURITIES.size))
    return np.array(rows)


def test_likelihood_equals_the_joint_gaussian_density():
    model = _model()
    T = 6
    Y = _simulate(model, T, seed=2)
    A, Q = model.transition()
    P = ts._initial_state_covariance(A, Q)
    L = ts.ns_loadings(MATURITIES, model.lam)
    cov_f = np.zeros((3 * T, 3 * T))
    for t in range(T):
        for s_ in range(t + 1):
            block = np.linalg.matrix_power(A, t - s_) @ P
            cov_f[3 * t:3 * t + 3, 3 * s_:3 * s_ + 3] = block
            cov_f[3 * s_:3 * s_ + 3, 3 * t:3 * t + 3] = block.T
    big_L = np.kron(np.eye(T), L)
    cov_y = big_L @ cov_f @ big_L.T + np.kron(np.eye(T), np.diag(model.h**2))
    mean_y = np.tile(L @ model.theta + model.adjustment(), T)
    exact = stats.multivariate_normal(mean_y, cov_y).logpdf(Y.ravel())
    for engine in ("cpp", "numpy"):
        assert afns.kalman_filter(model, Y, engine=engine)["loglik"] == pytest.approx(exact, rel=1e-10)
    # Without the adjustment the same data are less likely (the adjustment is part of the mean).
    plain = _model(arbitrage_free=False)
    assert afns.kalman_filter(plain, Y)["loglik"] < afns.kalman_filter(model, Y)["loglik"]


def test_maximum_likelihood_recovers_simulated_parameters():
    model = _model()
    Y = pd.DataFrame(_simulate(model, 400, seed=3), columns=MATURITIES)
    fit = afns.fit_afns(Y, lam0=0.4)
    assert fit.converged
    assert fit.lam == pytest.approx(model.lam, rel=0.05)
    assert fit.h[0] == pytest.approx(0.05, rel=0.05)              # 4,400 residuals: relative s.e. about 1%
    # Volatilities are estimated within about 5% on three simulated samples; 20% leaves room for other seeds.
    np.testing.assert_allclose(fit.sigma, model.sigma, rtol=0.2)
    # Mean reversions are weakly identified in 33 years (the slowest one, true 0.1 per year, was estimated between
    # 0.09 and 0.41 on three samples), so they are not asserted; the maximum is at least the likelihood of the truth.
    assert np.all(fit.kappa > 0)
    assert fit.loglik >= afns.kalman_filter(model, Y.to_numpy(), keep_states=False)["loglik"] - 1e-6


def test_forecasts_converge_to_the_long_run_curve_and_use_only_past_data():
    model = _model()
    far = afns.forecast(model, model.theta + 1.0, 10_000)
    np.testing.assert_allclose(far, ts.ns_loadings(MATURITIES, model.lam) @ model.theta + model.adjustment(), atol=1e-8)
    np.testing.assert_allclose(afns.forecast(model, model.theta, 7), far, atol=1e-12)
    idx = pd.date_range("2000-01-31", periods=96, freq="ME")
    Y = pd.DataFrame(_simulate(model, 96, seed=4), index=idx, columns=MATURITIES)
    out = afns.recursive_forecasts(Y, "2005-01-31", horizons=(1, 3), refit_every=24)
    shift = pd.Timestamp("2006-06-30")
    shocked = Y.copy()
    shocked.loc[shift:] += 1.5
    out2 = afns.recursive_forecasts(shocked, "2005-01-31", horizons=(1, 3), refit_every=24)
    for h in (1, 3):
        before = out[h].index < shift
        pd.testing.assert_frame_equal(out[h].loc[before], out2[h].loc[before])
        straddling = (out[h].index >= shift) & (out[h].index < shift + pd.DateOffset(months=h))
        assert straddling.sum() == h
        np.testing.assert_allclose(out2[h].loc[straddling] - out[h].loc[straddling], 1.5, rtol=0, atol=1e-9)
    assert out["fits"]["converged"].all()
