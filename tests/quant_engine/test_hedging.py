"""Delta-hedging simulator: exact identities, continuous-hedging limits and the Derman-Kamal approximation."""

import numpy as np
import pytest

from quant_engine import hedging, require_cpp

core = require_cpp()
S0, K, T, R_DOM, R_FOR, SIGMA = 1.10, 1.10, 0.5, 0.04, 0.02, 0.08


def run(**kw):
    args = dict(S0=S0, K=K, T=T, r_dom=R_DOM, r_for=R_FOR, sigma_price=SIGMA, sigma_hedge=SIGMA, n_steps=126,
                rebalance_every=1, cost_rate=0.0, dynamics="gbm", mu=0.03, sigma=SIGMA, n_paths=60_000, seed=7)
    args.update(kw)
    return hedging.simulate(**args)


def test_unhedged_pnl_is_premium_minus_payoff():
    out = run(rebalance_every=0, n_paths=1000)
    expected = out["premium"] - np.exp(-R_DOM * T) * np.maximum(out["final_spot"] - K, 0.0)
    np.testing.assert_allclose(out["pnl"], expected, rtol=1e-12, atol=1e-15)
    assert out["premium"] == pytest.approx(core.bs_price_greeks(S0, K, T, R_DOM, R_FOR, SIGMA, "call")["price"], rel=1e-14)


def test_daily_hedge_is_unbiased_and_matches_derman_kamal():
    out = run()
    pnl = out["pnl"]
    assert abs(pnl.mean()) < 4 * pnl.std() / np.sqrt(pnl.size)
    # sqrt(pi/4) vega sigma / sqrt(n) is an asymptotic approximation for near-the-money options.
    assert pnl.std() == pytest.approx(hedging.derman_kamal_std(S0, K, T, R_DOM, R_FOR, SIGMA, 126), rel=0.10)


def test_error_scales_with_square_root_of_rebalancing_interval():
    daily, weekly = run()["pnl"].std(), run(rebalance_every=4)["pnl"].std()
    assert weekly / daily == pytest.approx(2.0, rel=0.10)


def test_hedging_at_true_volatility_locks_in_the_premium_difference():
    # Sell at 10% implied, hedge at the true 8%: the hedge replicates the payoff at the 8% price.
    out = run(sigma_price=0.10)
    edge = out["premium"] - core.bs_price_greeks(S0, K, T, R_DOM, R_FOR, SIGMA, "call")["price"]
    se = out["pnl"].std() / np.sqrt(out["pnl"].size)
    assert out["pnl"].mean() == pytest.approx(edge, abs=4 * se + 2e-5)


def test_transaction_costs_are_accounted_exactly():
    free, costly = run(n_paths=5000), run(n_paths=5000, cost_rate=0.0002)
    np.testing.assert_allclose(free["pnl"] - costly["pnl"], costly["costs"], rtol=1e-9, atol=1e-14)
    assert costly["costs"].min() > 0


def test_garch_fhs_constant_variance_limit():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(20_000)
    z = (z - z.mean()) / z.std()
    omega = 0.25  # daily variance in percent^2 (0.5% daily volatility)
    out = run(dynamics="garch_fhs", garch=(0.0, omega, 0.0, 0.0, 0.0, omega), residuals=z, rebalance_every=0, n_paths=40_000)
    log_ret = np.log(out["final_spot"] / S0)
    assert log_ret.var() == pytest.approx(126 * omega / 1e4, rel=0.03)


@pytest.mark.parametrize("dynamics", ["heston", "garch_fhs"])
def test_reproducible_across_threads(dynamics):
    z = np.random.default_rng(1).standard_normal(500)
    kw = dict(dynamics=dynamics, heston=(0.0064, 2.0, 0.0064, 0.3, -0.3), garch=(0.0, 0.01, 0.05, 0.0, 0.9, 0.2),
              residuals=z, n_paths=3000)
    a, b = run(n_threads=1, **kw), run(n_threads=4, **kw)
    np.testing.assert_array_equal(a["pnl"], b["pnl"])
