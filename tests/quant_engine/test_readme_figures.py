"""The README figures render in both themes on simulated rates."""

import matplotlib

matplotlib.use("Agg")

from quant_engine import readme_figures  # noqa: E402
from quant_engine.figstyle import render  # noqa: E402


def test_all_readme_figures_render_in_both_themes(tmp_path):
    d = readme_figures.load(official=False, n_paths=5_000)
    for name, builder in readme_figures.FIGURES.items():
        paths = render(builder, name, tmp_path, d)
        assert [p.name for p in paths] == [f"{name}-light.png", f"{name}-dark.png"]
        assert all(p.stat().st_size > 10_000 for p in paths)
