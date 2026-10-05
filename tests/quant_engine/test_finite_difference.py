"""Crank-Nicolson solver against closed-form prices and a binomial tree."""

import numpy as np
import pytest

from quant_engine import black_scholes as ref
from quant_engine import require_cpp

core = require_cpp()


@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_matches_closed_form(kind):
    fd = core.bs_finite_difference(100, 100, 1, 0.05, 0.0, 0.2, kind, False, 1600, 1600)
    exact = core.bs_price_greeks(100, 100, 1, 0.05, 0.0, 0.2, kind)
    # Observed spatial error at 1600 nodes is about 4e-5 (second order, see below).
    assert fd["price"] == pytest.approx(exact["price"], abs=1e-4)
    assert fd["delta"] == pytest.approx(exact["delta"], abs=1e-4)
    assert fd["gamma"] == pytest.approx(exact["gamma"], abs=1e-4)


def test_second_order_convergence():
    exact = core.bs_price_greeks(100, 100, 1, 0.05, 0.0, 0.2, "put")["price"]
    errors = [abs(core.bs_finite_difference(100, 100, 1, 0.05, 0.0, 0.2, "put", False, m, m)["price"] - exact)
              for m in (200, 400, 800)]
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all((ratios > 3.5) & (ratios < 4.5))  # error ~ C h^2 -> ratio 4


def test_american_put_matches_binomial_tree():
    fd = core.bs_finite_difference(100, 100, 1, 0.05, 0.0, 0.2, "put", True, 1600, 1600)
    tree = ref.crr_price(100, 100, 1, 0.05, 0.0, 0.2, n_steps=20000, option_type="put", american=True)
    # Tree error O(1/n) ~ 1e-4 at n = 20000, grid error ~ 5e-5.
    assert fd["price"] == pytest.approx(tree, abs=5e-4)
    european = core.bs_price_greeks(100, 100, 1, 0.05, 0.0, 0.2, "put")["price"]
    assert fd["price"] > european + 0.4  # material early-exercise premium
    boundary = fd["exercise_boundary"]
    assert np.all(np.diff(boundary) <= 1e-9)  # critical price falls as maturity lengthens
    assert np.all(boundary < 100)


def test_american_call_without_dividends_equals_european():
    fd = core.bs_finite_difference(100, 110, 1, 0.04, 0.0, 0.25, "call", True, 1600, 1600)
    exact = core.bs_price_greeks(100, 110, 1, 0.04, 0.0, 0.25, "call")["price"]
    assert fd["price"] == pytest.approx(exact, abs=1e-4)


@pytest.mark.parametrize("kind,r,q", [("call", 0.03, 0.06), ("call", 0.03, 0.08), ("put", -0.01, 0.0), ("put", 0.02, 0.06)])
def test_american_options_with_yields_and_negative_rates_match_binomial_tree(kind, r, q):
    # Calls with a dividend yield above the rate are exercised early; a put with a negative rate and no yield
    # never is (its value equals the European one). Same tolerance as the put benchmark: tree error of order
    # 1e-4 at 20,000 steps plus the grid error.
    fd = core.bs_finite_difference(100, 95, 1.5, r, q, 0.25, kind, True, 1600, 1600)
    tree = ref.crr_price(100, 95, 1.5, r, q, 0.25, n_steps=20000, option_type=kind, american=True)
    assert fd["price"] == pytest.approx(tree, abs=5e-4)
    european = core.bs_price_greeks(100, 95, 1.5, r, q, 0.25, kind)["price"]
    assert fd["price"] >= european - 1e-4


def test_american_greeks_match_bumped_prices():
    args = (100, 1, 0.05, 0.0, 0.2, "put", True, 800, 800)
    fd = core.bs_finite_difference(100, *args)
    h = 0.5
    up = core.bs_finite_difference(100 + h, *args)["price"]
    down = core.bs_finite_difference(100 - h, *args)["price"]
    # Central differences of the price over re-centred grids: truncation O(h^2) ~ 1e-5 for delta and grid errors
    # of about 5e-5 in each price, i.e. 5e-5 on delta and 4e-4 on gamma.
    assert fd["delta"] == pytest.approx((up - down) / (2 * h), abs=1e-4)
    assert fd["gamma"] == pytest.approx((up - 2 * fd["price"] + down) / h ** 2, abs=1e-3)


@pytest.mark.parametrize("kind,r,q", [("call", 0.05, 0.0), ("put", -0.01, 0.0)])
def test_exercise_boundary_is_nan_when_early_exercise_is_never_optimal(kind, r, q):
    # The Dirichlet nodes equal the payoff by construction and used to be reported as a critical price at the
    # edge of the grid (271.8 for the call, 36.8 for the put), a spurious constant line that moved with the
    # grid width.
    fd = core.bs_finite_difference(100, 100, 1, r, q, 0.3, kind, True, 1600, 1600)
    assert len(fd["exercise_boundary"]) == len(fd["tau_grid"]) > 0
    assert np.all(np.isnan(fd["exercise_boundary"]))
    assert fd["price"] == pytest.approx(core.bs_price_greeks(100, 100, 1, r, q, 0.3, kind)["price"], abs=1e-4)
