"""Utilities for creating Bokeh visualizations."""

import math
import textwrap

from bokeh.core.properties import value
from bokeh.models import ColumnDataSource, CustomJS, HoverTool, LabelSet
from bokeh.plotting import figure

from ....managers.hover import panel as hover_panel
from ....visual_config import theme
from ...brand import SELECTED_HEXAGON_SIZE
from ...subsystem_link import (
    CATEGORY_BADGE,
    CATEGORY_CIRCLES,
    CATEGORY_HEXES,
    attach,
    focus_source,
)

# Import visualization constants
from ..enrichment.utilities import (
    calculate_arrowhead_vertices,
    get_connection_coordinates,
)
from ..generation.data_structures.constants import VisualizationConstants
from .js_callbacks import (
    NETWORK_LABEL_CHARACTER_WIDTH,
    NETWORK_LABEL_LINE_HEIGHT,
    add_fitting_label_callback,
    add_zoom_based_alpha_callback,
)

# The network's subsystem names, which the page's shared selection also names.
NETWORK_LABELS = "metabis-network-labels"
# An element's hover where the graph has none of its own: its identifier and name.
_FALLBACK_TOOLTIP = hover_panel.panel(hover_panel.row("ID", "@id"), hover_panel.row("Name", "@display_label"))
# A diamond's width over its height, as Bokeh's diamond marker draws it.
REACTION_DIAMOND_WIDTH = 2 / 3


def _include_point(bounds, x_value, y_value):
    """Expand bounds to include a single point."""
    bounds["min_x"] = min(bounds["min_x"], x_value)
    bounds["max_x"] = max(bounds["max_x"], x_value)
    bounds["min_y"] = min(bounds["min_y"], y_value)
    bounds["max_y"] = max(bounds["max_y"], y_value)


def _include_box(bounds, center_x, center_y, half_width, half_height):
    """Expand bounds to include a centered rectangle or circle extents."""
    _include_point(bounds, center_x - half_width, center_y - half_height)
    _include_point(bounds, center_x + half_width, center_y + half_height)


def _iter_coordinate_groups(xs, ys):
    """Yield flat coordinate groups from either flat or nested point sequences."""
    if xs is None or ys is None:
        return

    if len(xs) == 0 or len(ys) == 0:
        return

    first_x = xs[0]
    first_y = ys[0]
    if hasattr(first_x, "__iter__") or hasattr(first_y, "__iter__"):
        for x_group, y_group in zip(xs, ys):
            yield x_group, y_group
        return

    yield xs, ys


def _get_glyph_half_extents(glyph):
    """Return half-width and half-height for the glyph's visible footprint."""
    glyph_class = glyph.get("glyph_class")

    if glyph_class == "reaction":
        size = float(glyph.get("size", VisualizationConstants.REACTION_SIZE))
        angle = float(glyph.get("angle", VisualizationConstants.REACTION_ANGLE))
        extent = 0.5 * size * (abs(math.cos(angle)) + abs(math.sin(angle)))
        return extent, extent

    if glyph_class == "gene":
        return float(glyph.get("width", 60)) / 2, float(glyph.get("height", 30)) / 2

    if glyph_class == "metabolite":
        if glyph.get("subtype") == "intermediate":
            radius = float(
                glyph.get(
                    "radius", VisualizationConstants.INTERMEDIATE_METABOLITE_RADIUS
                )
            )
            return radius, radius

        return (
            float(glyph.get("width", VisualizationConstants.MAIN_METABOLITE_WIDTH)) / 2,
            float(glyph.get("height", VisualizationConstants.MAIN_METABOLITE_HEIGHT))
            / 2,
        )

    if glyph_class in {"compartment", "enzyme_group"}:
        return float(glyph.get("width", 0)) / 2, float(glyph.get("height", 0)) / 2

    if glyph_class in {"subsystem", "subsystem_category"}:
        radius = float(glyph.get("radius", 0))
        if radius > 0:
            return radius, radius

        return float(glyph.get("width", 0)) / 2, float(glyph.get("height", 0)) / 2

    return 0.0, 0.0


def _calculate_plot_bounds(glyphs, connections=None, auxiliary_data=None, zoom=1.0):
    """Calculate viewport bounds from glyph geometry and connection endpoints."""
    bounds = {
        "min_x": math.inf,
        "max_x": -math.inf,
        "min_y": math.inf,
        "max_y": -math.inf,
    }

    for glyph in glyphs or []:
        half_width, half_height = _get_glyph_half_extents(glyph)
        _include_box(
            bounds,
            float(glyph["x"]),
            float(glyph["y"]),
            half_width,
            half_height,
        )

    for connection in connections or []:
        for x_key, y_key in (("x0", "y0"), ("x1", "y1")):
            if x_key in connection and y_key in connection:
                _include_point(
                    bounds,
                    float(connection[x_key]),
                    float(connection[y_key]),
                )

        arrowhead_xs = connection.get("arrowhead_xs") or []
        arrowhead_ys = connection.get("arrowhead_ys") or []
        for xs, ys in _iter_coordinate_groups(arrowhead_xs, arrowhead_ys):
            for x_value, y_value in zip(xs, ys):
                _include_point(bounds, float(x_value), float(y_value))

    auxiliary_data = auxiliary_data or {}

    influence_circles = auxiliary_data.get("influence_circles") or {}
    if influence_circles:
        radii = influence_circles.get("radius", [])
        for x_value, y_value, radius in zip(
            influence_circles.get("x", []),
            influence_circles.get("y", []),
            radii,
        ):
            radius_value = float(radius)
            _include_box(
                bounds,
                float(x_value),
                float(y_value),
                radius_value,
                radius_value,
            )

    for line_key in ("influence_lines", "enzyme_reaction_connections"):
        line_data = auxiliary_data.get(line_key) or {}
        for x0, y0, x1, y1 in zip(
            line_data.get("x0", []),
            line_data.get("y0", []),
            line_data.get("x1", []),
            line_data.get("y1", []),
        ):
            _include_point(bounds, float(x0), float(y0))
            _include_point(bounds, float(x1), float(y1))

    if math.isinf(bounds["min_x"]) or math.isinf(bounds["min_y"]):
        return {
            "min_x": -1.0,
            "max_x": 1.0,
            "min_y": -1.0,
            "max_y": 1.0,
            "aspect_ratio": 1.0,
        }

    zoom_factor = max(float(zoom), 0.01)
    x_span = max(bounds["max_x"] - bounds["min_x"], 1.0)
    y_span = max(bounds["max_y"] - bounds["min_y"], 1.0)
    padding_x = max(VisualizationConstants.REACTION_SIZE * 2, x_span * 0.03)
    padding_y = max(VisualizationConstants.REACTION_SIZE * 2, y_span * 0.05)

    center_x = (bounds["min_x"] + bounds["max_x"]) / 2
    center_y = (bounds["min_y"] + bounds["max_y"]) / 2
    half_width = (x_span / 2 + padding_x) * zoom_factor
    half_height = (y_span / 2 + padding_y) * zoom_factor

    return {
        "min_x": center_x - half_width,
        "max_x": center_x + half_width,
        "min_y": center_y - half_height,
        "max_y": center_y + half_height,
        "aspect_ratio": max((2 * half_width) / max(2 * half_height, 1.0), 0.25),
    }


def _apply_plot_bounds(plot, bounds):
    """Apply precomputed viewport bounds to a Bokeh plot."""
    plot.x_range.start = bounds["min_x"]
    plot.x_range.end = bounds["max_x"]
    plot.y_range.start = bounds["max_y"]
    plot.y_range.end = bounds["min_y"]


def create_bokeh_plot(
    display_title,
    width,
    height,
    aspect_ratio,
    sizing_mode,
    glyphs,
    connections,
    auxiliary_data,
    zoom,
    match_aspect=False,
    fit_view_to_bounds=False,
):
    """Create and configure the Bokeh plot figure."""
    resolved_aspect_ratio = aspect_ratio or 1.0
    use_bounds_viewport = fit_view_to_bounds or not glyphs

    if use_bounds_viewport:
        bounds = _calculate_plot_bounds(
            glyphs,
            connections=connections,
            auxiliary_data=auxiliary_data,
            zoom=zoom,
        )
        resolved_aspect_ratio = aspect_ratio or bounds["aspect_ratio"]
    else:
        bounds = None

    figure_kwargs = {
        "width": int(width or max(height * resolved_aspect_ratio, 1)),
        "height": height,
        "sizing_mode": sizing_mode,
        "match_aspect": match_aspect,
        "title": display_title,
        "outline_line_width": VisualizationConstants.PLOT_OUTLINE_WIDTH,
        "outline_line_alpha": VisualizationConstants.PLOT_OUTLINE_ALPHA,
        "outline_line_color": VisualizationConstants.PLOT_OUTLINE_COLOR,
        "tools": "pan,wheel_zoom,reset,save",
    }

    if sizing_mode and sizing_mode.startswith("scale"):
        figure_kwargs["aspect_ratio"] = resolved_aspect_ratio

    plot = figure(
        **figure_kwargs,
    )

    # Style the plot
    plot.axis.visible = False
    plot.grid.visible = False
    plot.min_width = 0
    # Every link and node stays drawn while the view is dragged or zoomed. Past
    # 2,000 rows a glyph would otherwise be thinned to every tenth row, in a faint
    # grey, until the pointer is released: a compartment map's links (up to 8,500
    # rows in Mitochondria) and a large subsystem's would all but vanish.
    plot.lod_threshold = None

    if use_bounds_viewport and bounds is not None:
        _apply_plot_bounds(plot, bounds)
        return plot

    # Legacy fixed-aspect viewport logic.
    y_values = [g["y"] for g in glyphs]
    y_length = (min(y_values) - max(y_values)) * zoom
    plot.y_range.start = max(y_values)
    plot.y_range.end = plot.y_range.start + y_length

    x_values = [g["x"] for g in glyphs]
    plot.x_range.start = min(x_values)
    plot.x_range.end = plot.x_range.start + resolved_aspect_ratio * abs(y_length)

    return plot


# ============================================================================
# Phase 2: Unified Drawing Functions for Enriched Data
# ============================================================================


def draw_graph_elements(
    plot,
    enriched_glyphs,
    enriched_connections,
    auxiliary_data,
):
    """Draw all graph elements using enriched glyphs and connections directly.

    This is the Phase 3 unified drawing function that works directly with
    enriched data structures, eliminating the need for per-type dictionaries.

    Args:
        plot: Bokeh plot figure
        enriched_glyphs: List of enriched glyph dictionaries
        enriched_connections: List of enriched connection dictionaries
        auxiliary_data: Dict with influence_lines, influence_circles, etc.
    """
    # A Human1 map draws its links as Metabolic Atlas does: along its paths, and to
    # a nearby copy of a far node rather than across the map.
    atlas = auxiliary_data.get("atlas")
    if atlas:
        enriched_glyphs, enriched_connections = _local_copies(
            enriched_glyphs, enriched_connections, atlas
        )

    # Group glyphs by type for efficient processing
    grouped_glyphs = _group_glyphs_by_type(enriched_glyphs)

    # Draw all glyph types
    _draw_enriched_nodes(plot, grouped_glyphs, auxiliary_data.get("influence_circles"))

    # Draw all connection types; where Metabolic Atlas routes a reaction's links,
    # along its paths.
    if atlas:
        enriched_connections = _draw_atlas_links(plot, atlas, enriched_connections)
    _draw_enriched_connections(plot, enriched_connections)

    # Draw auxiliary visualizations
    _draw_auxiliary_elements(plot, auxiliary_data)


def _group_glyphs_by_type(enriched_glyphs):
    """Group enriched glyphs by their type for efficient rendering."""
    grouped = {
        "main_metabolite": [],
        "intermediate_metabolite": [],
        "gene": [],
        "reaction": [],
        "compartment": [],
        "enzyme_group": [],
        "subsystem": [],
        "subsystem_category": [],
    }

    for glyph in enriched_glyphs:
        glyph_class = glyph["glyph_class"]

        if glyph_class == "metabolite":
            group_key = (
                "intermediate_metabolite"
                if glyph.get("subtype") == "intermediate"
                else "main_metabolite"
            )
        else:
            group_key = glyph_class

        if group_key in grouped:
            grouped[group_key].append(glyph)

    return grouped


def _draw_enriched_nodes(plot, grouped_glyphs, influence_circles):
    """Draw different types of nodes using enriched glyph data."""
    renderers = {}

    # Draw subsystem categories first (as background boundaries)
    if grouped_glyphs["subsystem_category"]:
        renderers["subsystem_categories"] = _draw_enriched_subsystem_categories(
            plot, grouped_glyphs["subsystem_category"]
        )

    # Draw main metabolites
    if grouped_glyphs["main_metabolite"]:
        renderers["main_metabolites"] = _draw_enriched_main_metabolites(
            plot, grouped_glyphs["main_metabolite"]
        )

    # Draw intermediate metabolites
    if grouped_glyphs["intermediate_metabolite"]:
        renderers["intermediate_metabolites"] = _draw_enriched_intermediate_metabolites(
            plot, grouped_glyphs["intermediate_metabolite"]
        )

    # Draw genes
    if grouped_glyphs["gene"]:
        renderers["genes"] = _draw_enriched_genes(plot, grouped_glyphs["gene"])

    # Draw reactions
    if grouped_glyphs["reaction"]:
        renderers["reactions"] = _draw_enriched_reactions(
            plot, grouped_glyphs["reaction"]
        )

    # Draw compartments
    if grouped_glyphs["compartment"]:
        renderers["compartments"] = _draw_enriched_compartments(
            plot, grouped_glyphs["compartment"]
        )

    # Draw reaction background circles for gene influence
    if influence_circles:
        renderers["influence_circles"] = _draw_enriched_influence_circles(
            plot, influence_circles
        )

    # Draw enzyme groups
    if grouped_glyphs["enzyme_group"]:
        renderers["enzyme_groups"] = _draw_enriched_enzyme_groups(
            plot, grouped_glyphs["enzyme_group"]
        )

    # Draw subsystems
    if grouped_glyphs["subsystem"]:
        renderers["subsystems"] = _draw_enriched_subsystems(
            plot, grouped_glyphs["subsystem"]
        )

    return renderers


def _label_only_data(data):
    """Project glyph data to fields labels use, excluding rich hover HTML."""
    return {key: value for key, value in data.items() if key != "hover"}


def _draw_enriched_main_metabolites(plot, metabolite_glyphs):
    """Draw main metabolites using enriched glyph data."""
    # Convert enriched glyphs to ColumnDataSource format
    data = _convert_glyphs_to_bokeh_data(metabolite_glyphs)

    source = ColumnDataSource(data)
    renderer = plot.ellipse(
        x="x",
        y="y",
        width=VisualizationConstants.MAIN_METABOLITE_WIDTH,
        height=VisualizationConstants.MAIN_METABOLITE_HEIGHT,
        fill_color="color",
        fill_alpha="alpha",
        line_color=VisualizationConstants.GLOBAL_BORDER_COLOR,
        line_alpha="alpha",
        line_dash="line_dash",
        level="glyph",
        source=source,
    )

    # Add hover tool with dynamic tooltips
    hover = HoverTool(
        renderers=[renderer],
        tooltips="@hover{safe}"
        if any(g.get("hover") for g in metabolite_glyphs)
        else _FALLBACK_TOOLTIP,
    )
    plot.add_tools(hover)

    # Add labels
    # Initialize label alpha based on current zoom level
    initial_alpha = VisualizationConstants.SUBSYSTEM_LABEL_ALPHA_MAX

    # Add alpha values to the data source for labels
    label_data = _label_only_data(data)
    # Respect element alpha in label visibility
    label_data["base_alpha"] = label_data.get("alpha", [1.0] * len(label_data["x"]))
    label_data["text_alpha"] = [a * initial_alpha for a in label_data["base_alpha"]]
    label_source = ColumnDataSource(label_data)

    labels = LabelSet(
        x="x",
        y="y",
        text="display_label",
        text_color="black",
        text_font_size=VisualizationConstants.FONT_SIZE_METABOLITE_MAIN,
        text_font_style="bold",
        text_align="center",
        level="annotation",
        y_offset=VisualizationConstants.LABEL_OFFSET_DOWN,
        text_alpha="text_alpha",  # Use dynamic alpha from data source
        source=label_source,
    )
    plot.add_layout(labels)

    # Add zoom-based alpha callback for metabolite labels
    add_zoom_based_alpha_callback(
        plot,
        label_source,
        "main_metabolite",
        alpha_min=VisualizationConstants.METABOLITE_LABEL_ALPHA_MIN,
        alpha_max=VisualizationConstants.METABOLITE_LABEL_ALPHA_MAX,
        zoom_min=VisualizationConstants.METABOLITE_LABEL_ZOOM_THRESHOLD_MIN,
        zoom_max=VisualizationConstants.METABOLITE_LABEL_ZOOM_THRESHOLD_MAX,
    )

    return renderer


def _draw_enriched_intermediate_metabolites(plot, metabolite_glyphs):
    """Draw intermediate metabolites using enriched glyph data."""
    data = _convert_glyphs_to_bokeh_data(metabolite_glyphs)

    source = ColumnDataSource(data)
    renderer = plot.circle(
        x="x",
        y="y",
        radius=VisualizationConstants.INTERMEDIATE_METABOLITE_RADIUS,
        fill_color="color",
        fill_alpha="alpha",
        line_color=VisualizationConstants.GLOBAL_BORDER_COLOR,
        line_alpha="alpha",
        line_dash="line_dash",
        level="glyph",
        source=source,
    )

    # Add hover tool with dynamic tooltips
    hover = HoverTool(
        renderers=[renderer],
        tooltips="@hover{safe}"
        if any(g.get("hover") for g in metabolite_glyphs)
        else _FALLBACK_TOOLTIP,
    )
    plot.add_tools(hover)

    # Add labels
    # Initialize label alpha based on current zoom level
    initial_alpha = VisualizationConstants.SUBSYSTEM_LABEL_ALPHA_MAX

    # Add alpha values to the data source for labels
    label_data = _label_only_data(data)
    label_data["base_alpha"] = label_data.get("alpha", [1.0] * len(label_data["x"]))
    label_data["text_alpha"] = [a * initial_alpha for a in label_data["base_alpha"]]
    label_source = ColumnDataSource(label_data)

    labels = LabelSet(
        x="x",
        y="y",
        text="display_label",
        text_color="black",
        text_font_size=VisualizationConstants.FONT_SIZE_METABOLITE_INTERMEDIATE,
        text_font_style="bold",
        text_align="center",
        y_offset=VisualizationConstants.LABEL_OFFSET_UP,
        text_alpha="text_alpha",  # Use dynamic alpha from data source
        level="annotation",
        source=label_source,
    )
    plot.add_layout(labels)

    # Add zoom-based alpha callback for metabolite labels
    add_zoom_based_alpha_callback(
        plot,
        label_source,
        "intermediate_metabolite",
        alpha_min=VisualizationConstants.METABOLITE_LABEL_ALPHA_MIN,
        alpha_max=VisualizationConstants.METABOLITE_LABEL_ALPHA_MAX,
        zoom_min=VisualizationConstants.METABOLITE_LABEL_ZOOM_THRESHOLD_MIN,
        zoom_max=VisualizationConstants.METABOLITE_LABEL_ZOOM_THRESHOLD_MAX,
    )

    return renderer


def _draw_enriched_genes(plot, gene_glyphs):
    """Draw genes using enriched glyph data."""
    data = _convert_glyphs_to_bokeh_data(gene_glyphs)

    source = ColumnDataSource(data)
    renderer = plot.rect(
        x="x",
        y="y",
        width="width",
        height="height",
        fill_color="color",
        fill_alpha="alpha",
        line_color=VisualizationConstants.GLOBAL_BORDER_COLOR,
        line_alpha="alpha",
        line_dash="line_dash",
        level="annotation",
        source=source,
    )

    # Add hover tool with dynamic tooltips
    hover = HoverTool(
        renderers=[renderer],
        tooltips="@hover{safe}"
        if any(g.get("hover") for g in gene_glyphs)
        else _FALLBACK_TOOLTIP,
    )
    plot.add_tools(hover)

    # Add labels
    # Initialize label alpha based on current zoom level
    initial_alpha = VisualizationConstants.SUBSYSTEM_LABEL_ALPHA_MAX

    # Add alpha values to the data source for labels
    label_data = _label_only_data(data)
    label_data["base_alpha"] = label_data.get("alpha", [1.0] * len(label_data["x"]))
    label_data["text_alpha"] = [a * initial_alpha for a in label_data["base_alpha"]]
    label_source = ColumnDataSource(label_data)

    labels = LabelSet(
        x="x",
        y="y",
        text="display_label",
        text_color="black",
        text_font_size=VisualizationConstants.FONT_SIZE_GENE,
        text_font_style="bold",
        text_align="center",
        level="annotation",
        y_offset=VisualizationConstants.LABEL_OFFSET_DOWN,
        text_alpha="text_alpha",  # Use dynamic alpha from data source
        source=label_source,
    )
    plot.add_layout(labels)

    # Add zoom-based alpha callback for gene labels
    add_zoom_based_alpha_callback(
        plot,
        label_source,
        "gene",
        alpha_min=VisualizationConstants.GENE_LABEL_ALPHA_MIN,
        alpha_max=VisualizationConstants.GENE_LABEL_ALPHA_MAX,
        zoom_min=VisualizationConstants.GENE_LABEL_ZOOM_THRESHOLD_MIN,
        zoom_max=VisualizationConstants.GENE_LABEL_ZOOM_THRESHOLD_MAX,
    )

    return renderer


def _draw_enriched_reactions(plot, reaction_glyphs):
    """Draw reactions using enriched glyph data.

    A reaction is a diamond, taller than wide as the key's and the rankings' are,
    as tall as the square on its point it was drawn as before.
    """
    data = _convert_glyphs_to_bokeh_data(reaction_glyphs)
    half_heights = [float(size) / math.sqrt(2) for size in data["size"]]
    data["xs"] = [
        [x, x + h * REACTION_DIAMOND_WIDTH, x, x - h * REACTION_DIAMOND_WIDTH]
        for x, h in zip(data["x"], half_heights)
    ]
    data["ys"] = [[y - h, y, y + h, y] for y, h in zip(data["y"], half_heights)]

    source = ColumnDataSource(data)
    renderer = plot.patches(
        xs="xs",
        ys="ys",
        fill_color="color",
        fill_alpha="alpha",
        line_color=VisualizationConstants.GLOBAL_BORDER_COLOR,
        line_alpha="alpha",
        line_dash="line_dash",
        level="glyph",
        source=source,
    )

    # Add hover tool with dynamic tooltips
    hover = HoverTool(
        renderers=[renderer],
        tooltips="@hover{safe}"
        if any(g.get("hover") for g in reaction_glyphs)
        else _FALLBACK_TOOLTIP,
    )
    plot.add_tools(hover)

    # Add labels
    # Initialize label alpha based on current zoom level
    initial_alpha = VisualizationConstants.SUBSYSTEM_LABEL_ALPHA_MAX

    # Add alpha values to the data source for labels
    label_data = _label_only_data(data)
    label_data["base_alpha"] = label_data.get("alpha", [1.0] * len(label_data["x"]))
    label_data["text_alpha"] = [a * initial_alpha for a in label_data["base_alpha"]]
    label_source = ColumnDataSource(label_data)

    labels = LabelSet(
        x="x",
        y="y",
        text="display_label",
        text_color="black",
        text_font_size=VisualizationConstants.FONT_SIZE_REACTION,
        text_font_style="bold",
        y_offset=VisualizationConstants.LABEL_OFFSET_DOWN,
        level="annotation",
        text_align="center",
        text_alpha="text_alpha",  # Use dynamic alpha from data source
        source=label_source,
    )
    plot.add_layout(labels)

    # Add zoom-based alpha callback for reaction labels
    add_zoom_based_alpha_callback(
        plot,
        label_source,
        "reaction",
        alpha_min=VisualizationConstants.REACTION_LABEL_ALPHA_MIN,
        alpha_max=VisualizationConstants.REACTION_LABEL_ALPHA_MAX,
        zoom_min=VisualizationConstants.REACTION_LABEL_ZOOM_THRESHOLD_MIN,
        zoom_max=VisualizationConstants.REACTION_LABEL_ZOOM_THRESHOLD_MAX,
    )

    return renderer


def _draw_enriched_compartments(plot, compartment_glyphs):
    """Draw compartments using enriched glyph data."""
    data = _convert_glyphs_to_bokeh_data(compartment_glyphs)

    source = ColumnDataSource(data)
    renderer = plot.rect(
        x="x",
        y="y",
        height="height",
        width="width",
        height_units="data",
        width_units="data",
        fill_color="color",
        line_color="line_color",
        line_width=VisualizationConstants.COMPARTMENT_LINE_WIDTH,
        line_dash="line_dash",
        border_radius=VisualizationConstants.COMPARTMENT_BORDER_RADIUS,
        level="image",
        line_alpha="alpha",
        source=source,
    )

    # Add labels using the pre-computed label positions
    # Initialize label alpha based on current zoom level
    initial_alpha = VisualizationConstants.SUBSYSTEM_LABEL_ALPHA_MAX

    # Add alpha values to the data source for labels
    label_data = _label_only_data(data)
    label_data["base_alpha"] = label_data.get("alpha", [1.0] * len(label_data["x"]))
    label_data["text_alpha"] = [a * initial_alpha for a in label_data["base_alpha"]]
    # A long name takes two lines, so a narrow compartment's name keeps near its width.
    label_data["display_label"] = [
        "\n".join(textwrap.wrap(str(name), 12, break_long_words=False))
        for name in label_data["display_label"]
    ]
    label_source = ColumnDataSource(label_data)

    # Each name inside its compartment's top-left corner, on a white backing.
    labels = LabelSet(
        x="x_label",
        y="y_label",
        text="display_label",
        text_color=VisualizationConstants.CATEGORY_LABEL_COLOR,
        text_font=value(VisualizationConstants.FONT_FAMILY),
        text_font_size=VisualizationConstants.FONT_SIZE_COMPARTMENT,
        text_font_style="bold",
        text_align="left",
        text_baseline="top",
        x_offset=8,
        y_offset=-6,
        background_fill_color="#ffffff",
        background_fill_alpha=0.85,
        level="underlay",
        text_alpha="text_alpha",  # Use dynamic alpha from data source
        source=label_source,
    )
    plot.add_layout(labels)

    # Add zoom-based alpha callback for compartment labels
    add_zoom_based_alpha_callback(
        plot,
        label_source,
        "compartment",
        alpha_min=VisualizationConstants.COMPARTMENT_LABEL_ALPHA_MIN,
        alpha_max=VisualizationConstants.COMPARTMENT_LABEL_ALPHA_MAX,
        zoom_min=VisualizationConstants.COMPARTMENT_LABEL_ZOOM_THRESHOLD_MIN,
        zoom_max=VisualizationConstants.COMPARTMENT_LABEL_ZOOM_THRESHOLD_MAX,
    )

    return renderer


def _draw_enriched_enzyme_groups(plot, enzyme_group_glyphs):
    """Draw enzyme groups using enriched glyph data."""
    if not enzyme_group_glyphs:
        return

    data = _convert_glyphs_to_bokeh_data(enzyme_group_glyphs)

    source = ColumnDataSource(data)
    plot.rect(
        x="x",
        y="y",
        width="width",
        height="height",
        fill_alpha=VisualizationConstants.TRANSPARENT_ALPHA,
        line_color=VisualizationConstants.ENZYME_GROUP_COLOR,
        line_alpha="alpha",
        level="underlay",
        line_width=VisualizationConstants.ENZYME_CONNECTION_WIDTH,
        line_dash="line_dash",
        source=source,
    )


def _draw_enriched_subsystems(plot, subsystem_glyphs):
    """Draw subsystem glyphs as rectangular nodes with labels."""
    if not subsystem_glyphs:
        return

    # Convert enriched glyphs to ColumnDataSource format
    data = _convert_glyphs_to_bokeh_data(subsystem_glyphs)
    # A node with no data is white with a dashed edge, drawn a little heavier
    # so the dashes read at the overview's size.
    data["line_width"] = [
        1.0 if dash == "solid" else VisualizationConstants.NO_DATA_LINE_WIDTH for dash in data["line_dash"]
    ]
    source = ColumnDataSource(data)
    renderer = plot.circle(
        x="x",
        y="y",
        radius="radius",  # Use dynamic radius from enriched glyph data
        fill_color="color",
        fill_alpha="alpha",
        line_color=VisualizationConstants.GLOBAL_BORDER_COLOR,
        line_alpha="alpha",
        line_width="line_width",
        line_dash="line_dash",
        level="glyph",
        source=source,
    )

    # Add hover tool with rich hover text
    hover = HoverTool(
        renderers=[renderer],
        tooltips="@hover{safe}"  # Use rich hover from enriched data
        if any(g.get("hover") for g in subsystem_glyphs)
        else hover_panel.panel(
            hover_panel.row("ID", "@id"),
            hover_panel.row("Subsystem", "@display_label"),
            hover_panel.row("Reactions", "@reaction_count"),
            hover_panel.row("Metabolites", "@metabolite_count"),
        ),
    )
    plot.add_tools(hover)

    # Each name is drawn only where it fits inside its node, or for the subsystem
    # the page points at (js_callbacks.add_fitting_label_callback). The labels'
    # source holds only what they draw, so a page looking for the subsystems'
    # own source by its fields never takes it for that.
    label_data = {key: data[key] for key in ("x", "y", "element_id", "radius", "category")}
    lines = [
        textwrap.wrap(
            str(name),
            VisualizationConstants.NETWORK_LABEL_LINE_LENGTH,
            break_long_words=False,
        )
        or [""]
        for name in data["name"]
    ]
    label_data["label_text"] = ["\n".join(name) for name in lines]
    label_data["label_width"] = [
        max(len(line) for line in name) * NETWORK_LABEL_CHARACTER_WIDTH for name in lines
    ]
    label_data["label_height"] = [len(name) * NETWORK_LABEL_LINE_HEIGHT for name in lines]
    label_data["text_alpha"] = [0.0] * len(lines)
    label_data["text_align"] = ["center"] * len(lines)
    label_data["x_offset"] = [0.0] * len(lines)
    label_source = ColumnDataSource(label_data, name=NETWORK_LABELS)

    # A name has no box behind it, which would hide its node's colour, but a thin
    # white outline round its letters: the same names drawn first in white, each
    # letter stroked a little wider, so the dark letters over them keep their
    # full weight (BokehJS strokes text after filling it).
    for color, outline in (("#ffffff", "#ffffff"), (VisualizationConstants.CATEGORY_LABEL_COLOR, None)):
        plot.add_layout(
            LabelSet(
                x="x",
                y="y",
                text="label_text",
                text_color=color,
                text_outline_color=outline,
                text_font=value(VisualizationConstants.FONT_FAMILY),
                text_font_size=VisualizationConstants.FONT_SIZE_SUBSYSTEM,
                text_font_style="bold",
                text_align="text_align",
                text_baseline="middle",
                x_offset="x_offset",
                text_alpha="text_alpha",
                level="glyph",
                source=label_source,
            )
        )
    focus = focus_source()
    attach(plot, focus)
    add_fitting_label_callback(plot, label_source, focus)

    return renderer


def _draw_enriched_influence_circles(plot, influence_circle_data):
    """Draw background circles for reactions to indicate gene influence."""
    if not influence_circle_data or not influence_circle_data.get("x"):
        return

    source = ColumnDataSource(influence_circle_data)
    plot.circle(
        x="x",
        y="y",
        radius="radius",
        fill_color="color",
        fill_alpha="alpha",
        line_color=None,
        level="underlay",
        source=source,
    )


def _draw_enriched_connections(plot, enriched_connections):
    """Draw connections using enriched connection data."""
    if not enriched_connections:
        return {}

    renderers = {}

    # Separate connections by type for specialized rendering
    metabolic_connections = []
    regulatory_connections = []
    subsystem_connections = []
    other_connections = []

    for conn in enriched_connections:
        conn_class = conn.get("connection_class")
        if conn_class == "metabolic" and conn.get("arrowhead_xs"):
            metabolic_connections.append(conn)
        elif conn_class == "regulatory":
            regulatory_connections.append(conn)
        elif conn_class == "subsystem":
            subsystem_connections.append(conn)
        else:
            other_connections.append(conn)

    # Draw metabolic connections with arrowheads
    if metabolic_connections:
        renderers.update(
            _draw_metabolic_connections_with_arrowheads(plot, metabolic_connections)
        )

    # Draw regulatory connections with dashed lines
    if regulatory_connections:
        renderer = _draw_regulatory_connections(plot, regulatory_connections)
        if renderer:
            renderers["regulatory_connections"] = renderer

    # Draw subsystem connections as simple lines
    if subsystem_connections:
        renderer = _draw_subsystem_connections(plot, subsystem_connections)
        if renderer:
            renderers["subsystem_connections"] = renderer

    # Draw other connections as simple lines
    if other_connections:
        renderers["other_connections"] = _draw_simple_connections(
            plot, other_connections
        )

    return renderers


def _draw_metabolic_connections_with_arrowheads(plot, metabolic_connections):
    """Draw metabolic connections with arrowheads using enriched data."""
    # Prepare base data
    x0 = [c["x0"] for c in metabolic_connections]
    y0 = [c["y0"] for c in metabolic_connections]
    x1 = [c["x1"] for c in metabolic_connections]
    y1 = [c["y1"] for c in metabolic_connections]
    reg_color = [c["color"] for c in metabolic_connections]
    line_alpha = [c["line_alpha"] for c in metabolic_connections]
    line_dash = [c.get("line_dash", "solid") for c in metabolic_connections]

    # Underlay (glow) line data
    underlay_line_data = {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "color": reg_color,
        # Disable glow for missing connections
        "line_alpha": [
            (
                VisualizationConstants.EDGE_GLOW_ALPHA
                if not c.get("is_missing_connection", False)
                else 0.0
            )
            for c in metabolic_connections
        ],
        "line_dash": line_dash
        if VisualizationConstants.EDGE_GLOW_MATCH_DASH
        else ["solid"] * len(x0),
    }
    underlay_line_source = ColumnDataSource(underlay_line_data)

    # Overlay (black) line data
    overlay_line_data = {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "color": [VisualizationConstants.EDGE_OVERLAY_COLOR] * len(x0),
        "line_alpha": line_alpha,
        "line_dash": line_dash,
    }
    overlay_line_source = ColumnDataSource(overlay_line_data)

    # Arrowhead data (overlay uses black, underlay uses regulation color)
    xs = [c["arrowhead_xs"] for c in metabolic_connections]
    ys = [c["arrowhead_ys"] for c in metabolic_connections]
    fill_alpha = [c.get("fill_alpha", 1.0) for c in metabolic_connections]
    ah_line_alpha = [
        c.get("arrowhead_line_alpha", a)
        for c, a in zip(metabolic_connections, line_alpha)
    ]

    # Compute a slightly larger halo polygon around the arrow tip (first vertex)
    def _scale_polygon(xs_one, ys_one, origin_x, origin_y, scale):
        return (
            [origin_x + (x - origin_x) * scale for x in xs_one],
            [origin_y + (y - origin_y) * scale for y in ys_one],
        )

    halo_xs = []
    halo_ys = []
    if VisualizationConstants.EDGE_GLOW_ENABLED:
        for xs_one, ys_one in zip(xs, ys):
            if xs_one and ys_one:
                tip_x, tip_y = xs_one[0], ys_one[0]
                hx, hy = _scale_polygon(xs_one, ys_one, tip_x, tip_y, 1.4)
                halo_xs.append(hx)
                halo_ys.append(hy)
            else:
                halo_xs.append(xs_one)
                halo_ys.append(ys_one)

    arrowhead_underlay_data = {
        "xs": halo_xs if VisualizationConstants.EDGE_GLOW_ENABLED else xs,
        "ys": halo_ys if VisualizationConstants.EDGE_GLOW_ENABLED else ys,
        "fill_color": reg_color,
        # Disable halo for missing connections
        "fill_alpha": [
            (
                VisualizationConstants.EDGE_GLOW_ALPHA
                if not c.get("is_missing_connection", False)
                else 0.0
            )
            for c in metabolic_connections
        ],
    }
    arrowhead_overlay_data = {
        "xs": xs,
        "ys": ys,
        "fill_color": [VisualizationConstants.EDGE_OVERLAY_COLOR] * len(xs),
        "fill_alpha": fill_alpha,
        "line_alpha": ah_line_alpha,
    }

    # Draw underlay first (if enabled)
    renderers = {}
    if VisualizationConstants.EDGE_GLOW_ENABLED:
        underlay_renderer = plot.segment(
            x0="x0",
            y0="y0",
            x1="x1",
            y1="y1",
            color="color",
            line_alpha="line_alpha",
            line_width=VisualizationConstants.DEFAULT_LINE_WIDTH
            + VisualizationConstants.EDGE_GLOW_WIDTH_DELTA,
            line_dash="line_dash",
            line_cap="round",
            level="image",
            source=underlay_line_source,
        )
        renderers["connection_lines_glow"] = underlay_renderer

    # Draw overlay black line
    overlay_renderer = plot.segment(
        x0="x0",
        y0="y0",
        x1="x1",
        y1="y1",
        color="color",
        line_alpha="line_alpha",
        line_width=VisualizationConstants.DEFAULT_LINE_WIDTH,
        line_dash="line_dash",
        level="image",
        source=overlay_line_source,
    )
    renderers["connection_lines"] = overlay_renderer

    # Draw arrowheads: underlay (colored) then overlay (black)
    arrowhead_underlay_source = ColumnDataSource(arrowhead_underlay_data)
    if VisualizationConstants.EDGE_GLOW_ENABLED:
        ah_underlay_renderer = plot.patches(
            xs="xs",
            ys="ys",
            fill_color="fill_color",
            fill_alpha="fill_alpha",
            line_color=None,
            level="image",
            source=arrowhead_underlay_source,
        )
        renderers["connection_arrowheads_glow"] = ah_underlay_renderer

    arrowhead_overlay_source = ColumnDataSource(arrowhead_overlay_data)
    ah_overlay_renderer = plot.patches(
        xs="xs",
        ys="ys",
        fill_color="fill_color",
        fill_alpha="fill_alpha",
        line_color=None,
        level="image",
        source=arrowhead_overlay_source,
    )
    renderers["connection_arrowheads"] = ah_overlay_renderer

    return renderers


def _local_copies(glyphs, connections, atlas):
    """A far node's copy beside each node it links to across a map, as Metabolic Atlas draws.

    Metabolic Atlas draws a metabolite again beside each reaction far from it; a
    layout that draws it once links it across the map. A straight link longer than
    ``LOCAL_COPY_DISTANCE`` that no Atlas path routes ends instead at a copy of
    its far node: the one of its two ends with more such links, or else its
    metabolite. The copy keeps the node's identifier, colour, name and hover, and
    stands ``LOCAL_COPY_STUB`` from the near node, towards the far one.
    """
    links = atlas.get("links") or {}
    far = VisualizationConstants.LOCAL_COPY_DISTANCE
    by_id = {}
    for glyph in glyphs:
        by_id.setdefault(glyph["id"], glyph)

    def length(connection):
        return math.hypot(connection["x1"] - connection["x0"], connection["y1"] - connection["y0"])

    long_links = [
        index
        for index, connection in enumerate(connections)
        if connection.get("connection_class", "metabolic") == "metabolic"
        and "x0" in connection
        and connection.get("source") not in links
        and connection.get("target") not in links
        and connection.get("source") in by_id
        and connection.get("target") in by_id
        and length(connection) > far
    ]
    if not long_links:
        return glyphs, connections
    counts = {}
    for index in long_links:
        for end in ("source", "target"):
            counts[connections[index][end]] = counts.get(connections[index][end], 0) + 1

    glyphs = list(glyphs)
    connections = list(connections)
    for index in long_links:
        connection = dict(connections[index])
        source, target = connection["source"], connection["target"]
        if counts[source] != counts[target]:
            hub = source if counts[source] > counts[target] else target
        else:
            hub = source if by_id[source]["glyph_class"] == "metabolite" else target
        near = target if hub == source else source
        hub_glyph, near_glyph = by_id[hub], by_id[near]
        dx, dy = hub_glyph["x"] - near_glyph["x"], hub_glyph["y"] - near_glyph["y"]
        distance = math.hypot(dx, dy)
        stub = VisualizationConstants.LOCAL_COPY_STUB / distance
        copy = {
            **hub_glyph,
            "x": near_glyph["x"] + dx * stub,
            "y": near_glyph["y"] + dy * stub,
            "is_local_copy": True,
        }
        glyphs.append(copy)
        # The end at the far node moves to its copy; an arrowhead there moves with it.
        (x0, y0), (x1, y1) = (connection["x0"], connection["y0"]), (connection["x1"], connection["y1"])
        hub_at_end = math.hypot(x1 - hub_glyph["x"], y1 - hub_glyph["y"]) < math.hypot(
            x0 - hub_glyph["x"], y0 - hub_glyph["y"]
        )
        if hub_at_end:
            (connection["x0"], connection["y0"]), (connection["x1"], connection["y1"]) = (
                get_connection_coordinates(near_glyph, copy)
            )
            xs, ys = calculate_arrowhead_vertices(
                connection["x0"], connection["y0"], connection["x1"], connection["y1"]
            )
            connection["arrowhead_xs"], connection["arrowhead_ys"] = xs, ys
        else:
            (connection["x0"], connection["y0"]), (connection["x1"], connection["y1"]) = (
                get_connection_coordinates(copy, near_glyph)
            )
            if connection.get("arrowhead_xs"):
                xs, ys = calculate_arrowhead_vertices(
                    connection["x0"], connection["y0"], connection["x1"], connection["y1"]
                )
                connection["arrowhead_xs"], connection["arrowhead_ys"] = xs, ys
        connections[index] = connection
    return glyphs, connections


def _draw_atlas_links(plot, atlas, connections):
    """Each reaction Metabolic Atlas routes, along its own paths, styled as its links.

    A reaction's paths take the colour, fading and dash its straight links had
    (the glow in its colour under a grey line, as
    :func:`_draw_metabolic_connections_with_arrowheads` draws them). Its
    arrowheads stand where Metabolic Atlas's do, pointing the same way, but are
    drawn as the straight links' are: a filled grey triangle over a halo in the
    reaction's colour. The paths and arrowheads of every reaction sharing a style
    are drawn as one line broken by gaps, rather than one each: a map has
    thousands, and a few dozen lines repaint faster while the view is dragged.
    The arrowheads (8,500 in Mitochondria) are one triangle each, so only those
    in view are drawn, and none while they would be under
    ``ARROWHEAD_MIN_PIXELS`` long on screen, when a map is seen whole. Returns the
    connections left to draw straight: every one but a routed reaction's
    metabolic links.
    """
    links = atlas.get("links") or {}
    heads = atlas.get("heads") or {}
    styles = {}
    remaining = []
    for connection in connections:
        source, target = connection.get("source"), connection.get("target")
        reaction = source if source in links else target if target in links else None
        if reaction is None or connection.get("connection_class", "metabolic") != "metabolic":
            remaining.append(connection)
            continue
        styles.setdefault(reaction, connection)
    if not styles:
        return remaining

    def dash_key(dash):
        return tuple(dash) if isinstance(dash, list) else dash

    # (colour, opacity, dash) -> the paths drawn in it, gapped by NaN; and each
    # arrowhead, with its halo, as a triangle of its own.
    glows = {}
    overlays = {}
    halos = {"xs": [], "ys": [], "color": []}
    tips = {"xs": [], "ys": [], "alpha": []}

    def add(groups, key, flat):
        xs, ys = groups.setdefault(key, ([], []))
        if xs:
            xs.append(math.nan)
            ys.append(math.nan)
        xs.extend(flat[0::2])
        ys.extend(flat[1::2])

    for reaction, style in styles.items():
        dash = dash_key(style.get("line_dash", "solid"))
        for flat in links[reaction]:
            if not style.get("is_missing_connection", False):
                add(glows, (style["color"], VisualizationConstants.EDGE_GLOW_ALPHA, dash), flat)
            add(overlays, (VisualizationConstants.EDGE_OVERLAY_COLOR, style["line_alpha"], dash), flat)
        for flat in heads.get(reaction, ()):
            xs, ys = _arrowhead(flat)
            tips["xs"].append(xs)
            tips["ys"].append(ys)
            tips["alpha"].append(style.get("fill_alpha", style["line_alpha"]))
            if not style.get("is_missing_connection", False):
                # The halo, as the straight links' arrowheads have it: the head
                # grown round its tip, in the reaction's colour.
                tip_x, tip_y = xs[0], ys[0]
                halos["xs"].append([tip_x + (x - tip_x) * 1.4 for x in xs])
                halos["ys"].append([tip_y + (y - tip_y) * 1.4 for y in ys])
                halos["color"].append(style["color"])

    def draw(groups, **style):
        for (color, alpha, dash), (xs, ys) in groups.items():
            plot.line(
                x="x",
                y="y",
                source=ColumnDataSource({"x": xs, "y": ys}),
                line_color=color,
                line_alpha=alpha,
                line_dash=list(dash) if isinstance(dash, tuple) else dash,
                level="image",
                **style,
            )

    width = VisualizationConstants.DEFAULT_LINE_WIDTH
    if VisualizationConstants.EDGE_GLOW_ENABLED:
        draw(
            glows,
            line_width=width + VisualizationConstants.EDGE_GLOW_WIDTH_DELTA,
            line_cap="round",
            line_join="round",
        )
    draw(overlays, line_width=width, line_join="round")
    arrowheads = []
    if VisualizationConstants.EDGE_GLOW_ENABLED and halos["xs"]:
        arrowheads.append(
            plot.patches(
                xs="xs",
                ys="ys",
                source=ColumnDataSource(halos),
                fill_color="color",
                fill_alpha=VisualizationConstants.EDGE_GLOW_ALPHA,
                line_color=None,
                level="image",
            )
        )
    if tips["xs"]:
        arrowheads.append(
            plot.patches(
                xs="xs",
                ys="ys",
                source=ColumnDataSource(tips),
                fill_color=VisualizationConstants.EDGE_OVERLAY_COLOR,
                fill_alpha="alpha",
                line_color=None,
                level="image",
            )
        )
    if arrowheads:
        _hide_while_too_small(plot, arrowheads)
    return remaining


def _hide_while_too_small(plot, renderers):
    """Show ``renderers`` only once an arrowhead would be long enough on screen to read."""
    callback = CustomJS(
        args=dict(
            plot=plot,
            renderers=renderers,
            length=VisualizationConstants.ARROWHEAD_SIZE,
            least=VisualizationConstants.ARROWHEAD_MIN_PIXELS,
        ),
        code="""
        let width
        try { width = plot.inner_width } catch { return }
        const span = Math.abs(plot.x_range.end - plot.x_range.start)
        if (!(width > 0 && span > 0)) return
        const shown = length * width / span >= least
        for (const renderer of renderers) {
          if (renderer.visible !== shown) renderer.visible = shown
        }
        """,
    )
    for attribute in ("start", "end"):
        plot.x_range.js_on_change(attribute, callback)
    plot.js_on_change("inner_width", callback)


def _arrowhead(flat):
    """The straight links' arrowhead where Metabolic Atlas draws one, pointing the same way.

    Each of Metabolic Atlas's arrowheads is a triangle on a short base; its tip
    is the corner across from that base.
    """
    points = list(zip(flat[0::2], flat[1::2]))[:3]
    tip = min(range(3), key=lambda i: math.dist(points[(i + 1) % 3], points[(i + 2) % 3]))
    base = [points[(tip + 1) % 3], points[(tip + 2) % 3]]
    middle = ((base[0][0] + base[1][0]) / 2, (base[0][1] + base[1][1]) / 2)
    return calculate_arrowhead_vertices(middle[0], middle[1], points[tip][0], points[tip][1])


def _draw_simple_connections(plot, connections):
    """Draw simple connections without arrowheads."""
    line_data = {
        "x0": [c["x0"] for c in connections],
        "y0": [c["y0"] for c in connections],
        "x1": [c["x1"] for c in connections],
        "y1": [c["y1"] for c in connections],
        "color": [c["color"] for c in connections],
        "line_alpha": [c["line_alpha"] for c in connections],
        "line_width": [c.get("line_width", 2) for c in connections],
        "line_dash": [c.get("line_dash", "solid") for c in connections],
    }

    source = ColumnDataSource(line_data)
    return plot.segment(
        x0="x0",
        y0="y0",
        x1="x1",
        y1="y1",
        color="color",
        line_alpha="line_alpha",
        line_width="line_width",
        line_dash="line_dash",
        level="image",
        source=source,
    )


def _draw_auxiliary_elements(plot, auxiliary_data):
    """Draw auxiliary visualization elements."""
    # Draw gene influence connections
    if auxiliary_data.get("influence_lines"):
        _draw_gene_influence_connections(plot, auxiliary_data["influence_lines"])

    # Draw enzyme-reaction connections
    if auxiliary_data.get("enzyme_reaction_connections"):
        _draw_enzyme_reaction_connections(
            plot, auxiliary_data["enzyme_reaction_connections"]
        )


def _draw_gene_influence_connections(plot, influence_line_data):
    # Draw all influence connections at once
    if influence_line_data is not None:
        influence_source = ColumnDataSource(influence_line_data)
        plot.segment(
            x0="x0",
            y0="y0",
            x1="x1",
            y1="y1",
            color="line_color",
            line_alpha="line_alpha",
            line_width="line_width",
            level="underlay",
            source=influence_source,
        )


def _draw_enzyme_reaction_connections(
    plot,
    enzyme_reaction_connection_data,
):
    # Draw all enzyme connections at once
    if enzyme_reaction_connection_data is not None:
        enzyme_source = ColumnDataSource(enzyme_reaction_connection_data)
        plot.segment(
            x0="x0",
            y0="y0",
            x1="x1",
            y1="y1",
            color=VisualizationConstants.ENZYME_GROUP_COLOR,
            line_width=VisualizationConstants.ENZYME_CONNECTION_WIDTH,
            line_dash="dashed",
            line_alpha="line_alpha",
            source=enzyme_source,
        )

    return ["enzyme_connections"]


def _draw_regulatory_connections(plot, regulatory_connections):
    """Draw regulatory connections with dashed lines and enzyme group color.

    Regulatory connections are typically from enzyme groups to reactions
    and are styled with dashed lines in the enzyme group color.

    Args:
        plot: Bokeh plot figure
        regulatory_connections: List of regulatory connection dictionaries

    Returns:
        Bokeh renderer for the regulatory connections
    """
    if not regulatory_connections:
        return None

    line_data = {
        "x0": [c["x0"] for c in regulatory_connections],
        "y0": [c["y0"] for c in regulatory_connections],
        "x1": [c["x1"] for c in regulatory_connections],
        "y1": [c["y1"] for c in regulatory_connections],
        "color": [
            c.get("color", VisualizationConstants.ENZYME_GROUP_COLOR)
            for c in regulatory_connections
        ],
        "line_alpha": [c.get("line_alpha", 1.0) for c in regulatory_connections],
        "line_width": [
            c.get("line_width", VisualizationConstants.ENZYME_CONNECTION_WIDTH)
            for c in regulatory_connections
        ],
        "line_dash": [c.get("line_dash", "dashed") for c in regulatory_connections],
    }

    source = ColumnDataSource(line_data)
    # Draw dashed enzyme-colored lines behind nodes
    renderer = plot.segment(
        x0="x0",
        y0="y0",
        x1="x1",
        y1="y1",
        color="color",
        line_alpha="line_alpha",
        line_width="line_width",
        line_dash="line_dash",
        level="image",
        source=source,
    )

    return renderer


def _draw_subsystem_connections(plot, subsystem_connections):
    """Draw subsystem connections as thick lines to show metabolite flow.

    Args:
        plot: Bokeh plot figure
        subsystem_connections: List of subsystem connection dictionaries

    Returns:
        Bokeh renderer for the subsystem connections
    """
    if not subsystem_connections:
        return None

    line_data = {
        "x0": [c["x0"] for c in subsystem_connections],
        "y0": [c["y0"] for c in subsystem_connections],
        "x1": [c["x1"] for c in subsystem_connections],
        "y1": [c["y1"] for c in subsystem_connections],
        "color": [c.get("color", "#2E8B57") for c in subsystem_connections],
        "line_alpha": [c.get("line_alpha", 0.3) for c in subsystem_connections],
        "line_width": [
            c.get("line_width", 3.0) for c in subsystem_connections
        ],  # Width from enriched data
        "line_dash": [c.get("line_dash", "solid") for c in subsystem_connections],
    }

    source = ColumnDataSource(line_data)
    return plot.segment(
        x0="x0",
        y0="y0",
        x1="x1",
        y1="y1",
        color="color",
        line_alpha="line_alpha",
        line_width="line_width",
        line_dash="line_dash",
        level="underlay",
        source=source,
    )


def _draw_enriched_subsystem_categories(plot, category_glyphs):
    """Draw subsystem category boundary glyphs as circular regions."""
    if not category_glyphs:
        return

    # Convert enriched glyphs to ColumnDataSource format
    data = _convert_glyphs_to_bokeh_data(category_glyphs)
    source = ColumnDataSource(data)

    # Draw circles for category boundaries with background level
    renderer = plot.circle(
        x="x",
        y="y",
        radius="radius",  # Use dynamic radius from enriched glyph data
        fill_color="color",
        fill_alpha="alpha",
        line_color="line_color",
        line_alpha="alpha",
        line_width="line_width",
        line_dash="line_dash",
        level="underlay",  # Draw behind other elements
        source=source,
    )

    # Each category's number on its rim, up and to the left, in the logo's hexagon as
    # the side menu numbers the steps; the rankings head the category with the same
    # number. Pointing at it names the category; clicking it, or the circle's
    # background, selects the category (subsystem_link.add_network_focus_link).
    hexes = ColumnDataSource(
        {
            "x": data["x_badge"],
            "y": data["y_badge"],
            "badge": data["badge"],
            "category": data["category"],
            "name": data["display_label"],
            "count": data["subsystem_count"],
            "fill": [theme.INK_NODE] * len(data["x"]),
            "text": ["#ffffff"] * len(data["x"]),
            "alpha": [1.0] * len(data["x"]),
            "text_alpha": [1.0] * len(data["x"]),
        },
        name=CATEGORY_HEXES,
    )
    marks = plot.scatter(
        x="x",
        y="y",
        source=hexes,
        marker="hex",
        angle=math.pi / 6,
        size=VisualizationConstants.CATEGORY_HEXAGON_SIZE,
        fill_color="fill",
        fill_alpha="alpha",
        line_color=theme.SELECTED,
        line_alpha="alpha",
        line_width=1.5,
        level="annotation",
    )
    # A selected category's number takes the logo's own hexagon, stroked in its
    # gradient, under the number (subsystem_link.add_network_focus_link fills it).
    plot.image_url(
        url="url",
        x="x",
        y="y",
        w=SELECTED_HEXAGON_SIZE,
        h=SELECTED_HEXAGON_SIZE,
        w_units="screen",
        h_units="screen",
        anchor="center",
        source=ColumnDataSource({"url": [], "x": [], "y": []}, name=CATEGORY_BADGE),
        level="annotation",
    )
    plot.text(
        x="x",
        y="y",
        text="badge",
        source=hexes,
        text_color="text",
        text_alpha="text_alpha",
        text_font=value(VisualizationConstants.FONT_FAMILY),
        text_font_size=VisualizationConstants.FONT_SIZE_CATEGORY_NUMBER,
        text_font_style="bold",
        text_align="center",
        text_baseline="middle",
        level="annotation",
    )
    plot.add_tools(
        HoverTool(
            renderers=[marks],
            tooltips=hover_panel.panel(
                hover_panel.row("Category @badge", "@name"),
                hover_panel.row("Subsystems", "@count"),
                hover_panel.note("Click it to select the category."),
            ),
            attachment="right",
        )
    )
    source.name = CATEGORY_CIRCLES

    return renderer


def _convert_glyphs_to_bokeh_data(glyphs):
    """Convert enriched glyphs to Bokeh ColumnDataSource format."""
    if not glyphs:
        return {}

    # Extract common fields that all glyphs have
    data = {
        "x": [g["x"] for g in glyphs],
        "y": [g["y"] for g in glyphs],
        "id": [g["id"] for g in glyphs],
        "element_id": [g["id"] for g in glyphs],
        "alpha": [g["alpha"] for g in glyphs],
        "display_label": [g["display_label"] for g in glyphs],
        "color": [g["color"] for g in glyphs],
        "external_url": [g.get("external_url", "") for g in glyphs],
        "line_dash": [g.get("line_dash", "solid") for g in glyphs],
    }

    # Add type-specific fields if they exist
    sample_glyph = glyphs[0]
    glyph_class = sample_glyph["glyph_class"]

    if glyph_class == "metabolite":
        data["size"] = [g.get("size", 20) for g in glyphs]
        # data["hover"] = [g.get("hover", f"Metabolite: {g['id']}") for g in glyphs]
        data["hover"] = [g.get("hover", f"Metabolite: {g['id']}") for g in glyphs]
    elif glyph_class == "gene":
        data["width"] = [g.get("width", 60) for g in glyphs]
        data["height"] = [g.get("height", 30) for g in glyphs]
        data["hover"] = [g.get("hover", f"Gene: {g['id']}") for g in glyphs]
    elif glyph_class == "reaction":
        data["size"] = [g.get("size", 40) for g in glyphs]
        data["angle"] = [g.get("angle", 0.785398) for g in glyphs]
        data["hover"] = [g.get("hover", f"Reaction: {g['id']}") for g in glyphs]
    elif glyph_class == "compartment":
        data["width"] = [g["width"] for g in glyphs]
        data["height"] = [g["height"] for g in glyphs]
        data["line_color"] = [g["line_color"] for g in glyphs]
        data["x_label"] = [g["x_label"] for g in glyphs]
        data["y_label"] = [g["y_label"] for g in glyphs]
    elif glyph_class == "enzyme_group":
        data["width"] = [g["width"] for g in glyphs]
        data["height"] = [g["height"] for g in glyphs]
    elif glyph_class == "subsystem":
        data["width"] = [g["width"] for g in glyphs]
        data["height"] = [g["height"] for g in glyphs]
        data["reaction_count"] = [g.get("reaction_count", 0) for g in glyphs]
        data["metabolite_count"] = [g.get("metabolite_count", 0) for g in glyphs]
        data["is_subsystem"] = [g.get("is_subsystem", False) for g in glyphs]
        data["name"] = [str(g.get("name") or g["display_label"]) for g in glyphs]
        data["category"] = [str(g.get("category", "")) for g in glyphs]
        # Add radius for dynamic sizing
        data["radius"] = [g.get("radius", 20) for g in glyphs]
        # Add hover text
        data["hover"] = [g.get("hover", f"Subsystem: {g['id']}") for g in glyphs]
    elif glyph_class == "subsystem_category":
        data["radius"] = [g["radius"] for g in glyphs]
        data["subsystem_count"] = [g.get("subsystem_count", 0) for g in glyphs]
        data["line_color"] = [g.get("line_color", "black") for g in glyphs]
        data["text_color"] = [g.get("text_color", "black") for g in glyphs]
        data["line_width"] = [g.get("line_width", 1) for g in glyphs]
        data["font_size"] = [g.get("font_size", "10pt") for g in glyphs]
        data["label_alpha"] = [g.get("label_alpha", 1.0) for g in glyphs]
        # The number sits on the rim at 135 degrees, up and to the left (the
        # y-range is flipped, so up is -y).
        data["x_badge"] = [g["x"] - g["radius"] * math.sqrt(0.5) for g in glyphs]
        data["y_badge"] = [g["y"] - g["radius"] * math.sqrt(0.5) for g in glyphs]
        data["badge"] = [str(g.get("number", "")) for g in glyphs]
        data["category"] = [str(g.get("label") or g["display_label"]) for g in glyphs]
        # Add hover text
        data["hover"] = [
            g.get("hover", f"Category: {g['display_label']}") for g in glyphs
        ]

    return data
