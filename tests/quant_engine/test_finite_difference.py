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
