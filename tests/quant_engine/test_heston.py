"""Heston engine: published benchmark, independent quadrature, limiting case
and Monte Carlo consistency."""

import numpy as np
import pytest

from quant_engine import heston
from quant_engine import require_cpp

core = require_cpp()
FO = heston.FANG_OOSTERLEE_PARAMS


def test_fang_oosterlee_benchmark():
    # The published reference value (10 significant digits) is itself a
    # numerical result; our Fourier price agrees to 2e-8, i.e. 3e-9 relative.
    price = heston.price(100.0, [100.0], 1.0, 0.0, 0.0, FO)[0]
    assert price == pytest.approx(heston.FANG_OOSTERLEE_REFERENCE_CALL, abs=2e-8)


@pytest.mark.parametrize("K,T,r,q", [(80, 0.5, 0.02, 0.0), (100, 1.0, 0.0, 0.0), (130, 2.0, 0.03, 0.01), (100, 0.1, 0.01, 0.0)])
def test_matches_scipy_quadrature(K, T, r, q):
    params = heston.HestonParams(v0=0.04, kappa=2.0, theta=0.05, sigma=0.6, rho=-0.7)
    cpp = heston.price(100.0, [K], T, r, q, params)[0]
    reference = heston.call_price_reference(100.0, K, T, r, q, params)
    assert cpp == pytest.approx(reference, abs=1e-9)


def test_characteristic_function_matches_reference():
    u = np.linspace(-20, 20, 81) + 0.3j
    np.testing.assert_allclose(core.heston_cf(u, 1.3, 0.02, 0.01, *FO.as_tuple()),
                               heston.cf_reference(u, 1.3, 0.02, 0.01, FO), rtol=1e-12, atol=1e-14)


def test_small_vol_of_vol_limit_is_black_scholes():
    # With sigma -> 0 and rho = 0 the variance is deterministic,
    # v(t) = theta + (v0 - theta) e^{-kappa t}, so the price is Black-Scholes
    # with the average variance. The first correction is O(sigma^2).
    v0, kappa, theta, T = 0.09, 1.5, 0.04, 1.5
    params = heston.HestonParams(v0, kappa, theta, 1e-3, 0.0)
    avg_var = theta + (v0 - theta) * (1 - np.exp(-kappa * T)) / (kappa * T)
    for K in (70.0, 100.0, 140.0):
        bs = core.bs_price_greeks(100.0, K, T, 0.02, 0.0, np.sqrt(avg_var), "call")["price"]
        assert heston.price(100.0, [K], T, 0.02, 0.0, params)[0] == pytest.approx(bs, abs=1e-5)


def test_put_call_parity():
    params = heston.HestonParams(0.04, 1.2, 0.06, 0.8, -0.5)
    K = np.array([80.0, 100.0, 120.0])
    call = heston.price(100.0, K, 0.7, 0.03, 0.01, params, "call")
    put = heston.price(100.0, K, 0.7, 0.03, 0.01, params, "put")
    np.testing.assert_allclose(call - put, 100 * np.exp(-0.01 * 0.7) - K * np.exp(-0.03 * 0.7), atol=1e-10)


def test_monte_carlo_agrees_with_fourier():
    strikes = np.array([80.0, 100.0, 120.0])
    mc = heston.mc_price(100.0, strikes, 1.0, 0.0, 0.0, FO, n_steps=50, n_paths=200_000, seed=7)
    exact = heston.price(100.0, strikes, 1.0, 0.0, 0.0, FO)
    # 4 standard errors (sampling) plus 0.01 for the QE discretisation bias
    # at 50 steps per year, which Andersen (2008) reports to be of this order or smaller.
    assert np.all(np.abs(mc["call"] - exact) < 4 * mc["call_se"] + 0.01)
    assert mc["forward"] == pytest.approx(100.0, abs=4 * mc["forward_se"] + 0.01)


def test_monte_carlo_is_reproducible_across_thread_counts():
    args = (100.0, np.array([100.0]), 1.0, 0.01, 0.0, *FO.as_tuple(), 20, 20_000, 99)
    one = core.heston_mc(*args, 1)
    many = core.heston_mc(*args, 4)
    assert one["call"][0] == many["call"][0]
    assert one["put"][0] == many["put"][0]


def test_numpy_reference_agrees_with_cpp_monte_carlo():
    strikes = np.array([90.0, 110.0])
    cpp = heston.mc_price(100.0, strikes, 1.0, 0.01, 0.0, FO, n_steps=25, n_paths=100_000, seed=3)
    py = heston.mc_price_numpy(100.0, strikes, 1.0, 0.01, 0.0, FO, n_steps=25, n_paths=100_000, seed=4)
    # Independent random numbers: the difference has standard deviation sqrt(se1^2 + se2^2).
    se = np.sqrt(cpp["call_se"] ** 2 + py["call_se"] ** 2)
    assert np.all(np.abs(cpp["call"] - py["call"]) < 4 * se)


def test_calibration_recovers_parameters():
    true = heston.HestonParams(0.03, 1.8, 0.045, 0.55, -0.65)
    strikes = np.tile(np.linspace(80, 120, 9), 3)
    maturities = np.repeat([0.25, 0.75, 1.5], 9)
    prices = heston.price(100.0, strikes, maturities, 0.02, 0.0, true)
    vols = heston.implied_vols(prices, 100.0, strikes, maturities, 0.02, 0.0)
    fit = heston.calibrate(100.0, strikes, maturities, 0.02, 0.0, vols,
                           heston.HestonParams(0.05, 1.0, 0.06, 0.4, -0.3))
    assert fit.rmse_vol < 1e-6
    np.testing.assert_allclose(fit.params.as_tuple(), true.as_tuple(), rtol=1e-3, atol=1e-4)
