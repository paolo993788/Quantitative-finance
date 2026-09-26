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

from . import banking_book, curves, data, garch, hedging, risk, synthetic
from . import term_structure as ts
from .figstyle import end_label, header, label_offsets, new_figure, point_label, render

# Settings shared with the notebooks (FX option desk case study and FX risk notebook).
VALUATION_DATE, NOTIONAL_EUR, TENOR_DAYS, COST_RATE = "2025-12-31", 10_000_000, 126, 0.5e-4
ASSUMED_RATES, SEED = {"EUR": 0.020, "USD": 0.040}, 20251231
START, END, CURRENCIES, WINDOW, REFIT_EVERY, VAR_ALPHA = "1999-01-04", "2025-12-31", ("USD", "GBP", "JPY", "CHF"), 1000, 20, 0.01
# Settings shared with the yield-curve and IRRBB notebook.
YC_START, YC_END = "2004-09-01", "2025-12-31"
MATURITIES = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
FIRST_ORIGIN, HORIZON = "2014-12-31", 12
REGIMES = {"2015-2021: negative rates": ("2015-01-01", "2021-12-31"), "2022-2025: tightening and easing": ("2022-01-01", "2025-12-31")}


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


def load_rates(official=True) -> dict:
    svensson = (data.load_ecb_svensson_parameters(YC_START, YC_END) if official
                else synthetic.svensson_history(YC_START, YC_END))
    monthly = svensson.resample("ME").last().dropna()
    Y = ts.zero_curves(svensson, MATURITIES)
    fc = ts.recursive_forecasts(Y, FIRST_ORIGIN, (HORIZON,))[HORIZON]
    kfc = ts.recursive_kalman_forecasts(Y, FIRST_ORIGIN, (HORIZON,), refit_every=12, common_h=True)[HORIZON]
    errors = {"Nelson-Siegel, direct AR(1)": fc["DNS AR(1), direct"], "Nelson-Siegel, VAR(1)": fc["DNS VAR(1)"],
              "Nelson-Siegel, state space": kfc.loc[fc["random walk"].index]}
    ratios = {}
    for regime, (a, b) in REGIMES.items():
        rw = np.sqrt((fc["random walk"].loc[a:b] ** 2).mean())
        ratios[regime] = pd.DataFrame({k: np.sqrt((e.loc[a:b] ** 2).mean()) / rw for k, e in errors.items()})

    zero = curves.SvenssonCurve.from_series(svensson.iloc[-1]).zero_rate
    book = banking_book.BankBook()
    designs = {"Pay fixed, fitted to the six scenarios": dict(),
               "Pay or receive, fitted to the six scenarios": dict(allow_receive=True, penalty=1e-4),
               "Pay or receive, fitted to scenarios and history": dict(allow_receive=True, penalty=1e-2,
                                                                       svensson_monthly=monthly)}
    eve = {"Unhedged": book.delta_eve(zero)}
    hist = {"Unhedged": book.historical_eve_changes(zero, monthly)}
    for name, kw in designs.items():
        res = banking_book.minimax_hedge(book, zero, (2, 5, 10, 20), **kw)
        eve[name] = res["eve"]
        hist[name] = res.get("historical", book.historical_eve_changes(zero, monthly, swaps=res["swaps"]))
    table = pd.DataFrame(eve)
    table.loc["worst historical year"] = pd.Series({k: v.min() for k, v in hist.items()})
    source = ("Source: ECB euro area yield curve (AAA), Svensson parameters" if official
              else "Simulated yield curves, not market data")
    return {"ratios": ratios, "eve": 100 * table / book.tier1, "source": source,
            "base_date": svensson.index[-1]}


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


def yield_forecasts(t, d):
    fig, axes = new_figure(t, ncols=2, sharey=True)
    fig.subplots_adjust(right=0.97, bottom=0.2, top=0.74, wspace=0.08)
    for ax, (regime, table) in zip(axes, d["ratios"].items()):
        x = np.log(table.index.to_numpy(dtype=float))
        for k, col in enumerate(table.columns):
            ax.plot(x, table[col].to_numpy(), color=t["series"][k], lw=2, marker="o", ms=4, label=col)
        ax.axhline(1.0, color=t["ink2"], lw=1)
        ax.set_yscale("log")
        ax.set_yticks([0.5, 0.75, 1, 1.5, 2, 3, 4, 6], ["0.5", "0.75", "1", "1.5", "2", "3", "4", "6"])
        ax.minorticks_off()
        shown = [0.25, 1, 2, 5, 10, 30]
        ax.set_xticks(np.log(shown), ["3m", "1y", "2y", "5y", "10y", "30y"])
        ax.set_title(regime, fontsize=10, color=t["ink2"], loc="left")
        ax.set_xlabel("maturity")
    axes[0].set_ylabel("RMSE relative to random walk")
    axes[0].annotate("random walk = 1", (np.log(30), 1.0), xytext=(0, -4), textcoords="offset points", ha="right", va="top",
                     fontsize=8, color=t["ink2"])
    axes[1].legend(loc="upper right")
    header(fig, t, "Nelson-Siegel forecasts beat no change only in the 2022-2025 rate cycle",
           "12-month-ahead forecast error of three dynamic Nelson-Siegel models relative to no change, out of sample",
           d["source"] + "; month-end zero rates, recursive estimation")
    fig.texts[-1].set_y(0.015)
    return fig


def irrbb_hedging(t, d):
    table = d["eve"]
    fig, ax = new_figure(t, height=5.0)
    fig.subplots_adjust(right=0.97, bottom=0.14, top=0.74)
    groups = list(table.index)
    x = np.arange(len(groups)) + np.r_[np.zeros(len(groups) - 1), 0.5]
    width = 0.19
    for k, col in enumerate(table.columns):
        ax.bar(x + (k - 1.5) * (width + 0.01), table[col].to_numpy(), width, color=t["series"][k], label=col)
    ax.axhline(0, color=t["axis"], lw=1)
    ax.axhline(-100 * banking_book.EVE_OUTLIER_THRESHOLD, color=t["ink2"], lw=1, ls=(0, (4, 3)))
    ax.annotate("supervisory outlier threshold:\nEVE loss of 15% of Tier 1", (x[3] - 0.4, -100 * banking_book.EVE_OUTLIER_THRESHOLD),
                xytext=(0, 4), textcoords="offset points", fontsize=8, color=t["ink2"], va="bottom")
    ax.axvline(x[-1] - 0.75, color=t["grid"], lw=1)
    ax.set_xticks(x, [g.replace("short rates", "short") for g in groups])
    ax.set_ylabel("change in EVE, % of Tier 1")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
    ax.set_ylim(min(table.to_numpy().min() * 1.1, -60), table.to_numpy().max() * 1.35)
    robust = table.columns[-1]
    header(fig, t, "A hedge fitted only to the six supervisory scenarios fails on history",
           f"Stylised euro area bank, curve of {d['base_date']:%d %B %Y}: change in economic value of equity by BCBS scenario\n"
           "and in the worst historical 12-month move of the curve since 2004",
           d["source"] + f"; swap hedges from minimax linear programmes (robust hedge: worst loss {-table[robust].min():.1f}% of Tier 1)")
    fig.texts[-1].set_y(0.015)
    return fig


FIGURES = {"hedging_pnl": (hedging_pnl, "fx"), "hedging_frontier": (hedging_frontier, "fx"),
           "var_exceptions": (var_exceptions, "fx"), "yield_forecasts": (yield_forecasts, "rates"),
           "irrbb_hedging": (irrbb_hedging, "rates")}
LOADERS = {"fx": load, "rates": load_rates}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draw the README figures (light and dark variants).")
    parser.add_argument("--out", default=str(data.repository_root() / "docs" / "figures"))
    parser.add_argument("--synthetic", action="store_true", help="use simulated rates (offline)")
    parser.add_argument("--only", nargs="*", choices=list(FIGURES), help="draw only these figures")
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    names = args.only or list(FIGURES)
    inputs = {key: LOADERS[key](official=not args.synthetic) for key in {FIGURES[n][1] for n in names}}
    for name in names:
        builder, key = FIGURES[name]
        for path in render(builder, name, Path(args.out), inputs[key]):
            print(path)


if __name__ == "__main__":
    main()
