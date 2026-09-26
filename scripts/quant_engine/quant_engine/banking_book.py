"""Stylised balance sheet of a euro area retail bank for the IRRBB case study.

Amounts in EUR million; rates as decimals, continuously compounded zero rates for discounting.
The book follows the Basel Committee standardised framework (BCBS, 2016) and the EBA supervisory
outlier tests (EBA/RTS/2022/10): economic value of equity (EVE) under six scenarios with the
post-shock floor of EBA/GL/2022/14, and net interest income (NII) over one year with a constant
balance sheet under the two parallel scenarios.

Behavioural assumptions (stated, not estimated):

* fixed-rate mortgages prepay at a constant rate (CPR 5% a year), scaled in each scenario by the
  BCBS multipliers: 0.8 when rates rise in the parallel and steepener scenarios, 1.2 in the parallel
  down and flattener scenarios, 1.0 in the short-rate scenarios;
* non-maturity deposits: 70% core, slotted evenly over 7 years (average maturity about 3.5 years,
  within the BCBS cap of 5 years for retail transactional deposits), 30% overnight;
* floating-rate loans and central bank funding reprice every three months; equity and other assets
  carry no interest-rate cash flows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from . import term_structure as ts

SCENARIOS = list(ts.bcbs_scenarios([1.0]))
PREPAYMENT_MULTIPLIERS = {None: 1.0, "parallel up": 0.8, "parallel down": 1.2, "steepener": 0.8,
                          "flattener": 1.2, "short rates up": 1.0, "short rates down": 1.0}
OVERNIGHT = 1.0 / 365.0
EVE_OUTLIER_THRESHOLD = 0.15   # decline in EVE above 15% of Tier 1 capital
NII_OUTLIER_THRESHOLD = 0.05   # decline in NII above 5% of Tier 1 capital


def floored_shock(zero_rate, scenario: str):
    """Rate change of a BCBS scenario after the EBA floor, as a function of time."""
    def shock(t):
        t = np.asarray(t, dtype=float)
        base = np.asarray(zero_rate(np.maximum(t, OVERNIGHT)), dtype=float)
        return np.maximum(base + ts.bcbs_scenarios(t)[scenario], ts.eba_floor(t, base)) - base
    return shock


@dataclass
class BankBook:
    cash: float = 60.0
    floating_loans: float = 300.0
    mortgages: float = 350.0
    mortgage_rate: float = 0.034
    mortgage_years: float = 25.0
    cpr: float = 0.05
    bonds: float = 200.0
    bond_coupon: float = 0.028
    bond_years: float = 8.0
    other_assets: float = 90.0
    deposits: float = 550.0
    core_share: float = 0.7
    core_years: float = 7.0
    term_deposits: float = 150.0
    term_rate: float = 0.022
    term_years: float = 2.0
    senior_bonds: float = 150.0
    senior_coupon: float = 0.031
    senior_years: float = 5.0
    central_bank_funding: float = 70.0
    tier1: float = 80.0
    reset: float = 0.25

    def balance_sheet(self) -> pd.DataFrame:
        rows = [
            ("assets", "cash and reserves", self.cash, "overnight"),
            ("assets", "floating corporate loans", self.floating_loans, "reprice every 3 months"),
            ("assets", "fixed-rate mortgages", self.mortgages,
             f"{self.mortgage_years:.0f}y annuities at {100 * self.mortgage_rate:.2f}%, CPR {100 * self.cpr:.0f}%"),
            ("assets", "government bonds", self.bonds, f"{self.bond_years:.0f}y bullet, coupon {100 * self.bond_coupon:.2f}%"),
            ("assets", "other assets", self.other_assets, "no rate sensitivity"),
            ("liabilities", "non-maturity deposits", self.deposits,
             f"{100 * self.core_share:.0f}% core over {self.core_years:.0f}y, rest overnight"),
            ("liabilities", "term deposits", self.term_deposits, f"{self.term_years:.0f}y at {100 * self.term_rate:.2f}%"),
            ("liabilities", "senior bonds", self.senior_bonds,
             f"{self.senior_years:.0f}y bullet, coupon {100 * self.senior_coupon:.2f}%"),
            ("liabilities", "central bank funding", self.central_bank_funding, "reprice every 3 months"),
            ("liabilities", "equity (Tier 1)", self.tier1, "excluded from EVE"),
        ]
        return pd.DataFrame(rows, columns=["side", "item", "amount", "cash-flow assumption"])

    def positions(self, zero_rate, scenario=None, swaps=()) -> dict:
        """Cash flows (times, amounts; assets positive) of each position in a scenario (None = base)."""
        grow = float(np.exp(zero_rate(np.array([self.reset]))[0] * self.reset))   # floating notes worth par today
        cpr = min(self.cpr * PREPAYMENT_MULTIPLIERS[scenario], 1.0)
        mortgage_t, mortgage_a, _ = ts.mortgage_cashflows(self.mortgages, self.mortgage_rate, self.mortgage_years, cpr)
        deposit_t, deposit_a = ts.non_maturity_deposit_cashflows(self.deposits, self.core_share, self.core_years)
        term_t, term_a = ts.bullet_cashflows(self.term_deposits, self.term_rate, self.term_years)
        senior_t, senior_a = ts.bullet_cashflows(self.senior_bonds, self.senior_coupon, self.senior_years)
        out = {
            "cash and reserves": (np.array([OVERNIGHT]), np.array([self.cash])),
            "floating corporate loans": (np.array([self.reset]), np.array([self.floating_loans * grow])),
            "fixed-rate mortgages": (mortgage_t, mortgage_a),
            "government bonds": ts.bullet_cashflows(self.bonds, self.bond_coupon, self.bond_years),
            "non-maturity deposits": (deposit_t, -deposit_a),
            "term deposits": (term_t, -term_a),
            "senior bonds": (senior_t, -senior_a),
            "central bank funding": (np.array([self.reset]), np.array([-self.central_bank_funding * grow])),
        }
        for notional, years in swaps:   # positive notional: pay fixed; negative: receive fixed
            if notional:
                rate = ts.par_swap_rate(zero_rate, years)
                side = "payer" if notional > 0 else "receiver"
                out[f"{side} swap {years:g}y"] = ts.payer_swap_cashflows(notional, rate, years, zero_rate, self.reset)
        return out

    def cashflows(self, zero_rate, swaps=(), keys=None):
        def cf(scenario):
            pos = self.positions(zero_rate, scenario, swaps)
            items = [pos[k] for k in (keys or pos)]
            return np.concatenate([i[0] for i in items]), np.concatenate([i[1] for i in items])
        return cf

    def delta_eve(self, zero_rate, swaps=(), by_position: bool = False, floor: bool = True):
        if not by_position:
            return ts.delta_eve_book(self.cashflows(zero_rate, swaps), zero_rate, SCENARIOS, floor)
        keys = list(self.positions(zero_rate, None, swaps))
        return pd.DataFrame({k: ts.delta_eve_book(self.cashflows(zero_rate, swaps, [k]), zero_rate, SCENARIOS, floor)
                             for k in keys}).T

    def repricing(self, scenario=None, swaps=()):
        """Repricing times and notionals (assets positive) for the NII gap: overnight and floating positions,
        principal repaid by mortgages and core deposits (reinvested or refinanced at new rates), floating legs."""
        cpr = min(self.cpr * PREPAYMENT_MULTIPLIERS[scenario], 1.0)
        mortgage_t, _, mortgage_p = ts.mortgage_cashflows(self.mortgages, self.mortgage_rate, self.mortgage_years, cpr)
        deposit_t, deposit_a = ts.non_maturity_deposit_cashflows(self.deposits, self.core_share, self.core_years)
        times = [np.array([OVERNIGHT, self.reset, self.reset]), mortgage_t, deposit_t]
        amounts = [np.array([self.cash, self.floating_loans, -self.central_bank_funding]), mortgage_p, -deposit_a]
        for notional, years in swaps:
            times.append(np.array([self.reset]))
            amounts.append(np.array([notional]))   # the floating leg reprices (received by a payer, paid by a receiver)
        return np.concatenate(times), np.concatenate(amounts)

    def delta_nii(self, zero_rate, swaps=(), scenarios=("parallel up", "parallel down"), horizon: float = 1.0) -> pd.Series:
        return pd.Series({s: ts.delta_nii(*self.repricing(s, swaps), 1.0, floored_shock(zero_rate, s), horizon)
                          for s in scenarios}, name="change in NII")

    def historical_eve_changes(self, zero_rate, svensson_monthly: pd.DataFrame, months: int = 12, swaps=()) -> pd.Series:
        """EVE change when the base curve moves by each historical `months` change of the Svensson curve
        (base cash flows, EBA floor applied)."""
        from .curves import SvenssonCurve
        t, a = self.cashflows(zero_rate, swaps)(None)
        base = np.asarray(zero_rate(t), dtype=float)
        pv0 = ts.present_value(t, a, base)
        Z = np.array([SvenssonCurve.from_series(row).zero_rate(t) for _, row in svensson_monthly.iterrows()])
        dz = Z[months:] - Z[:-months]
        values = [ts.present_value(t, a, np.maximum(base + s, ts.eba_floor(t, base))) - pv0 for s in dz]
        return pd.Series(values, index=svensson_monthly.index[months:], name="change in EVE")


def minimax_hedge(book: BankBook, zero_rate, tenors=(2, 5, 10, 20), svensson_monthly: pd.DataFrame | None = None,
                  months: int = 12, allow_receive: bool = False, penalty: float = 0.0,
                  max_nii_loss: float | None = None, max_notional: float = 600.0) -> dict:
    """Par swaps that maximise the worst EVE change across a scenario set, as a linear programme.

    EVE changes are linear in the swap notionals, so maximising the minimum over scenarios is a linear
    programme. The scenario set is the six BCBS scenarios, plus every historical `months` change of the
    Svensson curve when `svensson_monthly` is given. With `allow_receive` notionals may be negative
    (receive-fixed swaps). `penalty` is subtracted from the objective per unit of gross notional, trading
    residual risk for simplicity; `max_nii_loss` (EUR million) bounds the NII decline in the parallel scenarios.
    """
    tenors = tuple(tenors)
    d = book.delta_eve(zero_rate).reindex(SCENARIOS).to_numpy()
    G = np.column_stack([book.delta_eve(zero_rate, [(1.0, y)]).reindex(SCENARIOS).to_numpy() - d for y in tenors])
    if svensson_monthly is not None:
        h0 = book.historical_eve_changes(zero_rate, svensson_monthly, months).to_numpy()
        Gh = np.column_stack([book.historical_eve_changes(zero_rate, svensson_monthly, months, [(1.0, y)]).to_numpy() - h0
                              for y in tenors])
        d, G = np.r_[d, h0], np.r_[G, Gh]
    nii0 = book.delta_nii(zero_rate).to_numpy()
    Gn = np.column_stack([book.delta_nii(zero_rate, [(1.0, y)]).to_numpy() - nii0 for y in tenors])
    signs = (1.0, -1.0) if allow_receive else (1.0,)
    k, m = len(tenors), d.size
    GG = np.hstack([s * G for s in signs])
    c = np.r_[penalty * np.ones(k * len(signs)), -1.0]
    A_ub, b_ub = np.c_[-GG, np.ones(m)], d.copy()
    if max_nii_loss is not None:
        A_ub = np.r_[A_ub, np.c_[-np.hstack([s * Gn for s in signs]), np.zeros(Gn.shape[0])]]
        b_ub = np.r_[b_ub, nii0 + max_nii_loss]
    bounds = [(0.0, max_notional)] * (k * len(signs)) + [(None, None)]
    res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"hedge optimisation failed: {res.message}")
    x = res.x[:-1].reshape(len(signs), k)
    notionals = x[0] - (x[1] if allow_receive else 0.0)
    notionals = np.where(np.abs(notionals) < 1e-6, 0.0, notionals)
    swaps = [(float(n), y) for n, y in zip(notionals, tenors)]
    eve = book.delta_eve(zero_rate, swaps)
    out = {"swaps": swaps, "notionals": pd.Series(notionals, index=[f"{y}y" for y in tenors], name="notional"),
           "gross_notional": float(np.abs(notionals).sum()), "eve": eve, "worst_eve": float(eve.min()),
           "nii": book.delta_nii(zero_rate, swaps)}
    if svensson_monthly is not None:
        out["historical"] = book.historical_eve_changes(zero_rate, svensson_monthly, months, swaps)
    return out
