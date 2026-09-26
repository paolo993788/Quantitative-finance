"""Synthetic stand-ins for the official data, used by the tests and by the
notebooks' offline mode (QUANT_ENGINE_DATA_MODE=synthetic).

The values are illustrative only and must not be read as market data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import garch

# Illustrative Svensson parameters of the same order of magnitude as a euro
# area AAA curve (not an official observation).
ILLUSTRATIVE_SVENSSON = {"BETA0": 2.6, "BETA1": -0.6, "BETA2": -1.2, "BETA3": 1.5, "TAU1": 1.8, "TAU2": 12.0}


def svensson_parameters(dates=("2025-12-31",)) -> pd.DataFrame:
    frame = pd.DataFrame([ILLUSTRATIVE_SVENSSON] * len(dates), index=pd.to_datetime(list(dates)))
    frame.attrs["source"] = "synthetic (illustrative parameters, not official data)"
    return frame


def fx_rates(start="2000-01-03", end="2025-12-31", seed=2024) -> pd.DataFrame:
    """Four synthetic exchange rates driven by GJR-GARCH-t shocks with a common
    factor, on a business-day calendar."""
    dates = pd.bdate_range(start, end)
    n = len(dates)
    base = {"mu": 0.0, "omega": 0.01, "alpha": 0.03, "gamma": 0.03, "beta": 0.94, "nu": 7.0}
    factor = garch.simulate(base, n, seed)
    names = ["USD", "GBP", "JPY", "CHF"]
    loadings = [0.5, 0.6, 0.4, 0.3]
    levels = [1.10, 0.85, 150.0, 1.05]
    columns = {}
    for k, (name, load, level) in enumerate(zip(names, loadings, levels)):
        own = garch.simulate(base, n, seed + 17 * (k + 1))
        log_returns = (load * factor + np.sqrt(1 - load**2) * own) / 100.0
        columns[name] = level * np.exp(np.cumsum(log_returns))
    frame = pd.DataFrame(columns, index=dates)
    frame.attrs["source"] = f"synthetic GJR-GARCH-t rates (seed {seed}), not official data"
    return frame
