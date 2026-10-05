"""The README figures render in both themes on simulated rates."""

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from quant_engine import data, readme_figures  # noqa: E402
from quant_engine.figstyle import render  # noqa: E402


def test_all_readme_figures_render_in_both_themes(tmp_path):
    inputs = {"fx": readme_figures.load(official=False, n_paths=5_000), "rates": readme_figures.load_rates(official=False)}
    assert set(inputs) == set(readme_figures.LOADERS)
    for name, (builder, key) in readme_figures.FIGURES.items():
        paths = render(builder, name, tmp_path, inputs[key])
        assert [p.name for p in paths] == [f"{name}-light.png", f"{name}-dark.png"]
        assert all(p.stat().st_size > 10_000 for p in paths)


def test_rate_fallbacks_are_announced_in_the_source_line(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(data, "load_ecb_svensson_parameters", unavailable)
    monkeypatch.setattr(data, "load_fred", unavailable)
    with pytest.warns(UserWarning, match="assumed USD rate"):
        r_eur, r_usd, assumed = readme_figures._rates(True, 0.5)
    assert (r_eur, r_usd) == (readme_figures.ASSUMED_RATES["EUR"], readme_figures.ASSUMED_RATES["USD"])
    assert assumed == ["EUR", "USD"]
    line = readme_figures.source_line(True, assumed)
    assert line.startswith("Source: ECB") and "assumed EUR rate 2.0%, USD rate 4.0% (download failed)" in line
    assert "assumed" not in readme_figures.source_line(True, [])
    assert readme_figures._rates(False, 0.5)[2] == [] and "Simulated" in readme_figures.source_line(False)
