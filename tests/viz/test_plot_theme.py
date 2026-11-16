"""The plot theme reaches every axis, legend and title of a serialized figure."""

from bokeh.embed import json_item
from bokeh.plotting import figure

from metabis import PLOT_THEME, apply_academic_style
from metabis.visual_config import theme


def _attributes(item, name):
    """Every serialized model of type ``name`` in a json_item, by its attributes."""
    found = []

    def visit(value):
        if isinstance(value, dict):
            if value.get("type") == "object" and value.get("name") == name:
                found.append(value.get("attributes", {}))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(item)
    return found


def _plot():
    plot = figure(title="A title", width=300, height=200)
    plot.scatter([1, 2], [3, 4], legend_label="Group")
    plot.xaxis.axis_label = "x"
    return apply_academic_style(plot)


def test_the_theme_reaches_axes_legends_and_titles_through_the_document():
    item = json_item(_plot(), theme=PLOT_THEME)

    axes = _attributes(item, "LinearAxis")
    assert axes and all(axis["major_label_text_font_size"] == "12px" for axis in axes)
    assert all(axis["axis_label_text_font_style"] == "normal" for axis in axes)
    assert all(axis["major_label_text_color"] == theme.INK_MUTED for axis in axes)
    (title,) = _attributes(item, "Title")
    assert title["text_font_size"] == "14px" and title["text_color"] == theme.INK
    (legend,) = _attributes(item, "Legend")
    assert legend["click_policy"] == "hide"
    (plot,) = _attributes(item, "Figure")
    assert plot["outline_line_color"] is None


def test_an_explicit_setting_still_wins_over_the_theme():
    plot = _plot()
    plot.xaxis.major_label_text_font_size = "9px"
    item = json_item(plot, theme=PLOT_THEME)

    sizes = sorted(axis["major_label_text_font_size"] for axis in _attributes(item, "LinearAxis"))
    assert sizes == ["12px", "9px"]


def test_the_toolbar_hides_until_pointed_at_and_shows_no_logo():
    plot = _plot()
    assert plot.toolbar.autohide is True
    assert plot.toolbar.logo is None
