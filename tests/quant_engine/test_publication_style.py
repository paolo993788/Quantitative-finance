"""Check export compatibility and figure-level text fitting."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from quant_engine.figstyle import header, new_figure, render, THEMES, _fit_figure_text


def builder(t):
    fig, ax = new_figure(t)
    ax.plot([0, 1, 2], [1, 2, 1], label="Observed")
    header(fig, t, "Observed and fitted values", "Illustrative series", "Source: synthetic test data")
    return fig


def test_publication_export(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_CHART_WIDTH_PX", "1200")
    monkeypatch.setenv("RESEARCH_CHART_SVG", "1")
    paths = render(builder, "publication", tmp_path)
    assert [p.name for p in paths] == ["publication-light.png", "publication-dark.png"]
    for p in paths:
        with Image.open(p) as image:
            assert image.width == 1200
        assert p.with_suffix(".svg").exists()


def test_figure_text_fits():
    fig = builder(THEMES["light"])
    try:
        _fit_figure_text(fig)
        renderer = fig.canvas.get_renderer()
        for artist in fig.texts:
            box = artist.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0
            assert box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1
    finally:
        plt.close(fig)
