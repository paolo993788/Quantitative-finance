"""Figures shown in the README, computed with the same data, models and seeds as the notebooks.

Run from the repository root (official data are downloaded and cached on first use):

    python -m quant_engine.readme_figures              # writes docs/figures/*-light.png and *-dark.png
    python -m quant_engine.readme_figures --synthetic  # offline check on simulated rates

Each figure is saved in a light and a dark variant; the README selects one with <picture>.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import curves, data, garch, hedging, risk, synthetic
from .figstyle import header, new_figure, point_label, render

# Settings shared with the notebooks (FX option desk case study and FX risk notebook).
VALUATION_DATE, NOTIONAL_EUR, TENOR_DAYS, COST_RATE = "2025-12-31", 10_000_000, 126, 0.5e-4
ASSUMED_RATES, SEED = {"EUR": 0.020, "USD": 0.040}, 20251231
START, END, CURRENCIES, WINDOW, REFIT_EVERY, VAR_ALPHA = "1999-01-04", "2025-12-31", ("USD", "GBP", "JPY", "CHF"), 1000, 20, 0.01


def _rates(official, T):
    if not official:
        return ASSUMED_RATES["EUR"], ASSUMED_RATES["USD"]
    try:
        sv = data.load_ecb_svensson_parameters("2025-12-01", VALUATION_DATE)
        r_eur = float(curves.SvenssonCurve.from_series(sv.iloc[-1]).zero_rate(T))
    except Exception:
        r_eur = ASSUMED_RATES["EUR"]
    try:
        r_usd = float(np.log1p(data.load_fred("DTB3", end=VALUATION_DATE).iloc[-1] / 100.0))
    except Exception:
        r_usd = ASSUMED_RATES["USD"]
    return r_eur, r_usd


def load(official=True, n_paths=200_000) -> dict:
    T = TENOR_DAYS / 252
    fx = (data.load_ecb_fx_rates(("USD",), end=VALUATION_DATE)["USD"].dropna() if official
          else synthetic.fx_rates(end=VALUATION_DATE)["USD"])
    r_eur, r_usd = _rates(official, T)
    spot = float(fx.iloc[-1])
    strike = spot * np.exp((r_usd - r_eur) * T)
    returns = 100 * np.log(fx).diff().dropna()
    fit = garch.fit(returns, "gjr", "t")
    p = fit.params
    s2 = garch.conditional_variance(returns, p, "gjr", "t")
    var_next, phi = float(s2[-1]), fit.persistence
    uncond = p["omega"] / (1 - phi)
    h = np.arange(1, TENOR_DAYS + 1)
    fair_vol = float(np.sqrt((uncond + phi ** (h - 1) * (var_next - uncond)).mean() * 252) / 100)
    z = (returns.to_numpy() - p["mu"]) / np.sqrt(s2[:-1])
    kappa = np.log(2) / (fit.half_life / 252)
    theta = uncond * 252 / 1e4
    vol_of_var = float(np.sqrt(2 * kappa * (s2 * 252 / 1e4).var() / theta))
    heston_params = (var_next * 252 / 1e4, kappa, theta, vol_of_var, 0.0)
    garch_params = (100 * (r_usd - r_eur) / 252, p["omega"], p["alpha"], p["gamma"], p["beta"], var_next)
    common = dict(S0=spot, K=strike, T=T, r_dom=r_usd, r_for=r_eur, sigma_price=fair_vol, sigma_hedge=fair_vol,
                  n_steps=TENOR_DAYS, rebalance_every=1, cost_rate=COST_RATE, mu=r_usd - r_eur, n_paths=n_paths, seed=SEED)
    scale = NOTIONAL_EUR / spot
    runs = {"Constant volatility (GBM)": hedging.simulate(**common, dynamics="gbm", sigma=fair_vol),
            "Stochastic volatility (Heston)": hedging.simulate(**common, dynamics="heston", heston=heston_params),
            "GARCH with historical shocks (FHS)": hedging.simulate(**common, dynamics="garch_fhs", garch=garch_params, residuals=z)}
    out = {"pnl": {k: v["pnl"] * scale for k, v in runs.items()}}
    frontier = {}
    for label, k in {"daily": 1, "every 2 days": 2, "weekly": 5, "every 2 weeks": 10, "monthly": 21}.items():
        o = hedging.simulate(**{**common, "rebalance_every": k}, dynamics="garch_fhs", garch=garch_params, residuals=z)
        frontier[label] = {"std": o["pnl"].std() * scale, "cost": o["costs"].mean() * scale}
    out["frontier"] = pd.DataFrame(frontier).T

    rates = (data.load_ecb_fx_rates(CURRENCIES, start=START, end=END) if official else synthetic.fx_rates(START, END)).dropna()
    port = 100.0 * np.log1p((rates.shift(1) / rates - 1.0).mean(axis=1)).dropna()
    gjr = garch.rolling_forecast(port, WINDOW, REFIT_EVERY, "gjr", "t", alphas=(VAR_ALPHA, 0.025))
    idx = gjr.index
    var = pd.DataFrame({"Historical simulation (500 days)": risk.historical_var_es(port, 500, VAR_ALPHA).reindex(idx)["var"],
                        "Filtered historical simulation (GJR-GARCH)": gjr[f"fhs_var_{VAR_ALPHA:g}"]}, index=idx)
    hits = (-port.reindex(idx).to_numpy()[:, None] > var.to_numpy()).astype(int)
    out["exceptions"] = pd.DataFrame(hits, index=idx, columns=var.columns).groupby(idx.year).sum()
    out["source"] = ("Source: ECB euro reference rates and AAA yield curve; US Treasury bill (FRED DTB3)" if official
                     else "Simulated rates, not market data")
    return out


def hedging_pnl(t, d):
    fig, ax = new_figure(t)
    fig.subplots_adjust(right=0.97, bottom=0.17)
    edges = np.linspace(-250, 100, 141)
    centres = 0.5 * (edges[1:] + edges[:-1])
    es = {}
    for k, (name, pnl) in enumerate(d["pnl"].items()):
        dens, _ = np.histogram(pnl / 1e3, bins=edges, density=True)
        ax.plot(centres, dens, color=t["series"][k], lw=1.8, label=name)
        es[name] = hedging.pnl_summary(pnl)["ES 97.5%"] / 1e3
    ax.axvline(0, color=t["axis"], lw=1)
    top = ax.get_ylim()[1]
    for k, (name, v) in enumerate(es.items()):
        ax.plot([-v, -v], [0, top * (0.18 + 0.1 * k)], color=t["series"][k], lw=1)
        ax.annotate(f"ES 97.5%: EUR {v:,.0f}k", (-v, top * (0.18 + 0.1 * k)), xytext=(-4, 0), textcoords="offset points",
                    fontsize=8, color=t["ink2"], va="center", ha="right")
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("P&L of the hedged short option, EUR thousand")
    ax.legend(loc="upper left")
    ratio = es["GARCH with historical shocks (FHS)"] / es["Constant volatility (GBM)"]
    header(fig, t, f"A constant-volatility model understates the tail risk {ratio:.1f} times",
           "Six-month EUR call sold to an exporter, EUR 10 million, delta-hedged daily at the fair volatility: P&L distribution",
           d["source"] + "; 200,000 paths per model, seed 20251231")
    fig.texts[-1].set_y(0.015)
    return fig


def hedging_frontier(t, d):
    f = d["frontier"] / 1e3
    fig, ax = new_figure(t)
    fig.subplots_adjust(right=0.95, bottom=0.17)
    ax.grid(axis="x", visible=True)
    ax.plot(f["std"], f["cost"], color=t["series"][0], lw=1.6)
    for label, row in f.iterrows():
        point_label(ax, t, row["std"], row["cost"], label, t["series"][0], dx=6, dy=4, ha="left")
    ax.set(xlabel="standard deviation of the hedged P&L, EUR thousand", ylabel="expected transaction costs, EUR thousand")
    ax.set_xlim(f["std"].min() * 0.9, f["std"].max() * 1.12)
    ax.set_ylim(0, f["cost"].max() * 1.2)
    header(fig, t, "Hedging less often saves costs but multiplies the risk",
           "Cost and risk of the delta hedge by rebalancing frequency (GARCH dynamics with historical shocks, same random paths)",
           d["source"])
    fig.texts[-1].set_y(0.015)
    return fig


def var_exceptions(t, d):
    ex = d["exceptions"]
    fig, ax = new_figure(t)
    fig.subplots_adjust(right=0.86)
    x = np.arange(len(ex))
    width = 0.36
    for k, col in enumerate(ex.columns):
        ax.bar(x + (k - 0.5) * (width + 0.04), ex[col].to_numpy(), width, color=t["series"][k], label=col)
    for level, label in ((5, "yellow zone from 5"), (10, "red zone from 10")):
        ax.axhline(level, color=t["axis"], lw=1)
        ax.annotate(label, (x[-1] + 0.6, level), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                    color=t["ink2"], annotation_clip=False)
    worst = ex.iloc[:, 0].idxmax()
    ax.annotate(f"{worst}: {int(ex.loc[worst].iloc[0])} exceptions for historical simulation", (list(ex.index).index(worst) - 0.2, ex.loc[worst].iloc[0]),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, color=t["ink2"])
    ax.set_xticks(x[::2], [str(y) for y in ex.index[::2]])
    ax.set_ylim(0, max(ex.to_numpy().max() * 1.2, 12))
    ax.yaxis.set_major_locator(__import__("matplotlib").ticker.MaxNLocator(integer=True))
    ax.legend(loc="upper right")
    fhs, hs = ex.columns[1], ex.columns[0]
    title = ("Filtered historical simulation stays out of the red zone" if not (ex[fhs] >= 10).any()
             else "Exceptions of the one-day 99% VaR by year")
    header(fig, t, title,
           "One-day 99% VaR exceptions per year, book of USD, GBP, JPY and CHF against the euro (2.5 expected per year)",
           d["source"].split(";")[0])
    return fig


FIGURES = {"hedging_pnl": hedging_pnl, "hedging_frontier": hedging_frontier, "var_exceptions": var_exceptions}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draw the README figures (light and dark variants).")
    parser.add_argument("--out", default=str(data.repository_root() / "docs" / "figures"))
    parser.add_argument("--synthetic", action="store_true", help="use simulated rates (offline)")
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    d = load(official=not args.synthetic)
    for name, builder in FIGURES.items():
        for path in render(builder, name, Path(args.out), d):
            print(path)


if __name__ == "__main__":
    main()
