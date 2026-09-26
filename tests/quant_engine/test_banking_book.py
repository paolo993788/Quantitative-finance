"""IRRBB case study book: consistency of the balance sheet, EVE decomposition, hedge optimisation."""

import numpy as np
import pandas as pd
import pytest

from quant_engine import banking_book as bb
from quant_engine import synthetic
from quant_engine import term_structure as ts
from quant_engine.curves import SvenssonCurve

CURVE = SvenssonCurve(2.6, -0.6, -1.2, 1.5, 1.8, 12.0)


def test_balance_sheet_balances_and_floating_positions_are_at_par():
    book = bb.BankBook()
    sheet = book.balance_sheet().groupby("side")["amount"].sum()
    assert sheet["assets"] == pytest.approx(sheet["liabilities"])
    pos = book.positions(CURVE.zero_rate)
    for name in ("floating corporate loans", "central bank funding"):
        t, a = pos[name]
        assert abs(ts.present_value(t, a, CURVE.zero_rate(t))) == pytest.approx(
            book.floating_loans if name.startswith("floating") else book.central_bank_funding, rel=1e-12)


def test_eve_by_position_adds_up_and_swaps_are_linear():
    book = bb.BankBook()
    total = book.delta_eve(CURVE.zero_rate)
    parts = book.delta_eve(CURVE.zero_rate, by_position=True)
    pd.testing.assert_series_equal(parts.sum().rename("change in EVE"), total, rtol=1e-10)
    one = book.delta_eve(CURVE.zero_rate, [(1.0, 10)]) - total
    hundred = book.delta_eve(CURVE.zero_rate, [(100.0, 10)]) - total
    np.testing.assert_allclose(hundred, 100 * one, rtol=1e-9)
    assert one["parallel up"] > 0 > one["parallel down"]


def test_prepayment_multipliers_change_mortgage_cash_flows():
    book = bb.BankBook()
    base = book.positions(CURVE.zero_rate, None)["fixed-rate mortgages"][1]
    down = book.positions(CURVE.zero_rate, "parallel down")["fixed-rate mortgages"][1]
    assert down[:12].sum() > base[:12].sum() and down.sum() < base.sum()


def test_minimax_hedge_is_consistent_and_respects_the_nii_limit():
    book = bb.BankBook()
    free = bb.minimax_hedge(book, CURVE.zero_rate)
    assert all(n >= 0 for n, _ in free["swaps"])
    assert free["worst_eve"] > book.delta_eve(CURVE.zero_rate).min()
    signed = bb.minimax_hedge(book, CURVE.zero_rate, allow_receive=True)
    assert signed["worst_eve"] >= free["worst_eve"] - 1e-9     # a larger feasible set cannot do worse
    nii0 = book.delta_nii(CURVE.zero_rate)
    limit = -nii0.min() + 0.5
    tight = bb.minimax_hedge(book, CURVE.zero_rate, max_nii_loss=limit)
    assert tight["nii"].min() >= -limit - 1e-6
    assert tight["worst_eve"] <= free["worst_eve"] + 1e-9
    lean = bb.minimax_hedge(book, CURVE.zero_rate, allow_receive=True, penalty=0.01)
    assert lean["gross_notional"] <= signed["gross_notional"] + 1e-9


def test_minimax_hedge_with_historical_scenarios():
    book = bb.BankBook()
    frame = synthetic.svensson_history("2015-01-01", "2020-12-31").resample("ME").last()
    res = bb.minimax_hedge(book, CURVE.zero_rate, svensson_monthly=frame, allow_receive=True, penalty=1e-3)
    worst = min(res["eve"].min(), res["historical"].min())
    unhedged = min(book.delta_eve(CURVE.zero_rate).min(), book.historical_eve_changes(CURVE.zero_rate, frame).min())
    assert worst > unhedged
    receiver = book.positions(CURVE.zero_rate, None, [(-10.0, 5)])
    assert "receiver swap 5y" in receiver


def test_historical_eve_changes_zero_for_a_constant_curve():
    book = bb.BankBook()
    frame = synthetic.svensson_parameters(pd.date_range("2020-01-31", periods=15, freq="ME"))
    out = book.historical_eve_changes(CURVE.zero_rate, frame, months=12)
    assert len(out) == 3 and np.allclose(out, 0.0)
