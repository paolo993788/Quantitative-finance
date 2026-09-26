"""Parsers for the official ECB formats and the Svensson curve."""

import numpy as np
import pytest
from scipy import integrate

from quant_engine import curves, data

FX_SAMPLE = """Date,USD,JPY,BGN,GBP,CHF,
2024-01-03,1.0919,155.13,1.9558,0.86265,0.9291,
2024-01-02,1.0956,155.34,N/A,0.86760,0.9272,
"""

YC_SAMPLE = """KEY,FREQ,REF_AREA,CURRENCY,PROVIDER_FM,INSTRUMENT_FM,PROVIDER_FM_ID,DATA_TYPE_FM,TIME_PERIOD,OBS_VALUE
YC.B.U2.EUR.4F.G_N_A.SV_C_YM.BETA0,B,U2,EUR,4F,G_N_A,SV_C_YM,BETA0,2024-01-02,2.1
YC.B.U2.EUR.4F.G_N_A.SV_C_YM.BETA1,B,U2,EUR,4F,G_N_A,SV_C_YM,BETA1,2024-01-02,1.6
YC.B.U2.EUR.4F.G_N_A.SV_C_YM.BETA2,B,U2,EUR,4F,G_N_A,SV_C_YM,BETA2,2024-01-02,-2.5
YC.B.U2.EUR.4F.G_N_A.SV_C_YM.BETA3,B,U2,EUR,4F,G_N_A,SV_C_YM,BETA3,2024-01-02,3.0
YC.B.U2.EUR.4F.G_N_A.SV_C_YM.TAU1,B,U2,EUR,4F,G_N_A,SV_C_YM,TAU1,2024-01-02,1.2
YC.B.U2.EUR.4F.G_N_A.SV_C_YM.TAU2,B,U2,EUR,4F,G_N_A,SV_C_YM,TAU2,2024-01-02,9.5
"""


def test_parse_fx_file():
    frame = data.parse_ecb_fx_csv(FX_SAMPLE)
    assert list(frame.columns) == ["USD", "JPY", "BGN", "GBP", "CHF"]
    assert frame.index.is_monotonic_increasing
    assert frame.loc["2024-01-03", "USD"] == pytest.approx(1.0919)
    assert np.isnan(frame.loc["2024-01-02", "BGN"])


def test_parse_yield_curve_file():
    table = data.parse_ecb_yield_curve_csv(YC_SAMPLE)
    assert list(table.columns) == data.SVENSSON_COLUMNS
    curve = curves.SvenssonCurve.from_series(table.iloc[0])
    assert curve.beta2 == pytest.approx(-2.5)


def test_svensson_limits_and_forward_consistency():
    curve = curves.SvenssonCurve(2.1, 1.6, -2.5, 3.0, 1.2, 9.5)
    assert curve.zero_rate(1e-8) == pytest.approx((2.1 + 1.6) / 100, abs=1e-8)
    assert curve.zero_rate(1e4) == pytest.approx(2.1 / 100, abs=1e-4)
    for m in (0.5, 5.0, 30.0):
        avg_forward, _ = integrate.quad(curve.forward_rate, 0, m, epsabs=1e-14)
        assert curve.zero_rate(m) == pytest.approx(avg_forward / m, abs=1e-12)
    assert curve.discount(10.0) == pytest.approx(np.exp(-10 * curve.zero_rate(10.0)))
