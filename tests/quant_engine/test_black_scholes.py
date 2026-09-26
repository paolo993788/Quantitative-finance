"""Black-Scholes engine: C++ against the NumPy reference and analytical identities."""

import numpy as np
import pytest

from quant_engine import black_scholes as ref
from quant_engine import require_cpp

core = require_cpp()
CASES = [(100, 100, 1.0, 0.05, 0.0, 0.2), (100, 80, 0.25, 0.03, 0.01, 0.35), (50, 70, 2.0, -0.005, 0.02, 0.15)]


@pytest.mark.parametrize("S,K,T,r,q,sigma", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_price_and_greeks_match_reference(S, K, T, r, q, sigma, kind):
    # Same closed-form formulas in two languages: agreement to rounding error.
    cpp = core.bs_price_greeks(S, K, T, r, q, sigma, kind)
    assert cpp["price"] == pytest.approx(float(ref.bs_price(S, K, T, r, q, sigma, kind)), rel=1e-12, abs=1e-12)
    greeks = ref.bs_greeks(S, K, T, r, q, sigma, kind)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert cpp[name] == pytest.approx(float(greeks[name]), rel=1e-10, abs=1e-12)


def test_textbook_value():
    # Hull, Options, Futures and Other Derivatives: S=K=100, T=1, r=5%, sigma=20% -> 10.4506.
    assert core.bs_price_greeks(100, 100, 1, 0.05, 0, 0.2, "call")["price"] == pytest.approx(10.4506, abs=5e-5)


@pytest.mark.parametrize("S,K,T,r,q,sigma", CASES)
def test_put_call_parity(S, K, T, r, q, sigma):
    call = core.bs_price_greeks(S, K, T, r, q, sigma, "call")["price"]
    put = core.bs_price_greeks(S, K, T, r, q, sigma, "put")["price"]
    assert call - put == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-11)


def test_greeks_against_finite_differences():
    S, K, T, r, q, sigma = 100, 95, 0.5, 0.02, 0.01, 0.25
    f = lambda **kw: core.bs_price_greeks(**{**dict(S=S, K=K, T=T, r=r, q=q, sigma=sigma), **kw}, option_type="call")["price"]
    g = core.bs_price_greeks(S, K, T, r, q, sigma, "call")
    h = 1e-4  # central differences: truncation O(h^2) ~ 1e-8, well inside the tolerance
    assert g["delta"] == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((f(S=S + h) - 2 * f() + f(S=S - h)) / h**2, rel=1e-4)
    assert g["vega"] == pytest.approx((f(sigma=sigma + h) - f(sigma=sigma - h)) / (2 * h), rel=1e-6)
    assert g["rho"] == pytest.approx((f(r=r + h) - f(r=r - h)) / (2 * h), rel=1e-6)
    assert g["theta"] == pytest.approx(-(f(T=T + h) - f(T=T - h)) / (2 * h), rel=1e-6)


def test_implied_vol_round_trip():
    strikes = np.linspace(60, 160, 21)
    T, r, q, sigma = 0.75, 0.03, 0.01, 0.27
    prices = np.array([core.bs_price_greeks(100, k, T, r, q, sigma, "call")["price"] for k in strikes])
    iv = core.bs_implied_vol(prices, 100, strikes, np.full_like(strikes, T), np.full_like(strikes, r), q, "call")
    np.testing.assert_allclose(iv, sigma, atol=1e-9)
    assert ref.implied_vol(prices[10], 100, strikes[10], T, r, q) == pytest.approx(sigma, abs=1e-10)


def test_implied_vol_rejects_arbitrage_prices():
    iv = core.bs_implied_vol(np.array([0.0, 150.0]), 100, np.array([100.0, 100.0]), np.array([1.0, 1.0]),
                             np.array([0.0, 0.0]), 0.0, "call")
    assert np.isnan(iv).all()
