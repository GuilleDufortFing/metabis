"""Tap behavior for figures embedded with and without a Bokeh server.

Inside a Bokeh server document, taps drive the server's own ``Select`` and
``Toggle``. A figure delivered as ``json_item`` would carry detached copies of
those widgets, so it hands subsystem taps to the page as a window event and
reads the hover-download setting from a page flag instead.
"""

import json

import pytest
from bokeh.embed import json_item
from bokeh.models import ColumnDataSource, CustomJS, Select, TapTool, Toggle
from bokeh.plotting import figure

from metabis.viz.click_behavior import CLICK_BEHAVIOR_CODE, add_click_behavior


def _plot():
    plot = figure()
    subsystems = ColumnDataSource(
        {
            "x": [0, 1],
            "y": [0, 1],
            "element_id": ["glycolysis", "tca"],
            "display_label": ["Glycolysis", "TCA cycle"],
            "is_subsystem": [True, True],
        }
    )
    hovers = ColumnDataSource({"x": [2], "y": [2], "hover": ["<b>MAR04370</b>"]})
    plot.scatter("x", "y", source=subsystems)
    plot.scatter("x", "y", source=hovers)
    return plot, subsystems, hovers


def _callbacks(source):
    return [
        callback
        for callback in source.selected.js_property_callbacks.get("change:indices", [])
        if isinstance(callback, CustomJS)
    ]


def test_a_server_free_figure_hands_subsystem_taps_to_the_page():
    plot, subsystems, hovers = _plot()

    add_click_behavior(plot, navigation_event="metabis:navigate-subsystem")

    (callback,) = _callbacks(subsystems)
    assert callback.args["navigation_event"] == "metabis:navigate-subsystem"
    assert "subsystem_select" not in callback.args
    assert callback.code == CLICK_BEHAVIOR_CODE
    assert "window.dispatchEvent(new CustomEvent(navigation_event" in callback.code
    # Without a download setting the hover-only source has nothing to do on tap.
    assert _callbacks(hovers) == []
    assert plot.select(type=TapTool)


def test_the_tap_selects_only_what_it_acts_on():
    """A tap on the network's category circles must not dim them as unselected."""
    plot, subsystems, _hovers = _plot()
    circles = ColumnDataSource({"x": [0], "y": [0], "radius": [5]})
    plot.circle(x="x", y="y", radius="radius", source=circles)

    add_click_behavior(plot, navigation_event="metabis:navigate-subsystem")

    (tap,) = plot.select(type=TapTool)
    assert [renderer.data_source for renderer in tap.renderers] == [subsystems]


def test_a_server_free_figure_reads_hover_downloads_from_a_page_flag():
    plot, _subsystems, hovers = _plot()

    add_click_behavior(plot, download_flag="metabisDownloadHoverOnClick")

    (callback,) = _callbacks(hovers)
    assert callback.args["download_flag"] == "metabisDownloadHoverOnClick"
    assert "enabled_toggle" not in callback.args
    assert "Boolean(window[download_flag])" in callback.code


def test_a_server_free_figure_serializes_without_widget_copies():
    plot, _subsystems, _hovers = _plot()

    add_click_behavior(
        plot,
        navigation_event="metabis:navigate-subsystem",
        download_flag="metabisDownloadHoverOnClick",
    )

    payload = json.dumps(json_item(plot))
    assert '"Select"' not in payload
    assert '"Toggle"' not in payload


def test_server_documents_keep_driving_their_own_widgets():
    plot, subsystems, hovers = _plot()
    select = Select(options=["Glycolysis", "TCA cycle"])
    toggle = Toggle()

    add_click_behavior(plot, enabled_toggle=toggle, subsystem_select=select)

    (subsystem_callback,) = _callbacks(subsystems)
    (hover_callback,) = _callbacks(hovers)
    assert subsystem_callback.args["subsystem_select"] is select
    assert hover_callback.args["enabled_toggle"] is toggle
    assert "navigation_event" not in subsystem_callback.args


@pytest.mark.parametrize(
    ("widgets", "message"),
    [
        ({"subsystem_select": Select(), "navigation_event": "e"}, "subsystem_select or navigation_event"),
        ({"enabled_toggle": Toggle(), "download_flag": "f"}, "enabled_toggle or download_flag"),
    ],
)
def test_each_alternative_replaces_its_widget(widgets, message):
    plot, _subsystems, _hovers = _plot()

    with pytest.raises(ValueError, match=message):
        add_click_behavior(plot, **widgets)
