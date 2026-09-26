"""Nelson-Siegel-Svensson zero-coupon curves, as published by the ECB.

The ECB estimates its euro area yield curves with the Svensson (1994) model
and publishes the daily parameters BETA0-BETA3 (in percent) and TAU1, TAU2
(in years). The resulting spot rates are continuously compounded; maturities
are in years.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SvenssonCurve:
    beta0: float  # percent
    beta1: float  # percent
    beta2: float  # percent
    beta3: float  # percent
    tau1: float   # years
    tau2: float   # years

    @classmethod
    def from_series(cls, row) -> "SvenssonCurve":
        """Build a curve from a row with columns BETA0..BETA3, TAU1, TAU2."""
        return cls(*(float(row[k]) for k in ("BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2")))

    def zero_rate(self, maturity):
        """Continuously compounded spot rate (decimal) for maturities in years."""
        m = np.maximum(np.asarray(maturity, dtype=float), 1e-10)
        x1, x2 = m / self.tau1, m / self.tau2
        f1 = (1.0 - np.exp(-x1)) / x1
        f2 = f1 - np.exp(-x1)
        f3 = (1.0 - np.exp(-x2)) / x2 - np.exp(-x2)
        return (self.beta0 + self.beta1 * f1 + self.beta2 * f2 + self.beta3 * f3) / 100.0

    def forward_rate(self, maturity):
        """Instantaneous forward rate (decimal, continuous compounding)."""
        m = np.asarray(maturity, dtype=float)
        x1, x2 = m / self.tau1, m / self.tau2
        return (self.beta0 + self.beta1 * np.exp(-x1) + self.beta2 * x1 * np.exp(-x1)
                + self.beta3 * x2 * np.exp(-x2)) / 100.0

    def discount(self, maturity):
        m = np.asarray(maturity, dtype=float)
        return np.exp(-self.zero_rate(m) * m)
