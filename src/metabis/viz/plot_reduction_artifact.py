"""Artifact-first Bokeh factories for dimensionality-reduction results.

The sample scatters follow how single-cell tools draw an embedding: points sized
by how many there are, translucent with a white edge so overlaps show, the legend
outside with each group's count, both axes at one scale, PCA axes with their
share of variance, and UMAP and t-SNE axes with no numbers, which carry no unit
(as scanpy and Seurat draw them).

A scatter can instead colour each sample by one reaction or gene, as scanpy's
``pl.umap(color=gene)`` and Seurat's ``FeaturePlot`` do, on the sample heatmap's
own scale: the feature standardized over the grouped samples, clipped at three
standard deviations, so a sample's dot has the colour of its heatmap cell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape

import numpy as np
import pandas as pd
from bokeh.models import (
    Arrow,
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    FactorRange,
    FixedTicker,
    HoverTool,
    LabelSet,
    LinearColorMapper,
    Range1d,
    Span,
    Title,
    VeeHead,
)
from bokeh.transform import transform
from bokeh.palettes import Category10
from bokeh.plotting import figure

from ..analysis.dimensionality_reduction import ReductionArtifact
from ..managers.group_palettes import OKABE_ITO
from ..managers.color_distributions import gradient_palette
from ..managers.hover.panel import extend, panel, row, section
from ..managers.hover.panel import inner as hover_panel_inner
from ..visual_config import theme
from .click_behavior import get_external_reference_data
from .plot_reduced_dimension import (
    calculate_cluster_metrics,
    get_metric_interpretation,
)
from .plot_utils import apply_academic_style, short_row_label

DEFAULT_CATEGORICAL_PALETTE = OKABE_ITO
DEFAULT_SINGLE_COLOR = OKABE_ITO[3]  # Okabe-Ito blue, for samples with no group
# Category10 is kept for callers that imported it from here.
LEGACY_CATEGORICAL_PALETTE = Category10[10]
VARIANCE_MARKS = (0.5, 0.8)  # where the cumulative share is marked
BIPLOT_SECTORS = 8  # one arrow per 45 degrees
BIPLOT_MINIMUM_SHARE = 0.15  # of the longest arrow
LOADING_ROW_HEIGHT = 20
FEATURE_REACH = 3.0  # standard deviations, as the sample heatmap clips its rows
COMPONENT_BAR_ROWS = 5.0  # the variance bars' height, in feature rows
LABEL_LIMIT = 30  # characters of a row name beside another figure
# Named sources the page reads and writes: which features each row holds, and
# the row outlined as the samples' colour.
FEATURE_ROWS = "feature_rows"
FEATURE_CHOICE = "feature_choice"


def point_size(count: int) -> float:
    """Fewer points draw larger: about 9 px for 300 samples, 6 for 1,000, 3 for 10,000."""
    return float(np.clip(160 / math.sqrt(max(count, 1)), 3, 10))


@dataclass(frozen=True)
class ReductionSamplePresentation:
    """Canonical sample identity and its frozen display label."""

    sample_id: str
    display_label: str


@dataclass(frozen=True)
class ReductionGroupPresentation:
    """One ordered group, including exact membership and presentation values."""

    group_id: str
    display_name: str
    color: str
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReductionFeaturePresentation:
    """Canonical feature identity plus frozen display/hover values."""

    feature_id: str
    display_label: str
    hover_html: str = ""
    external_url: str = ""


@dataclass(frozen=True)
class ReductionColouring:
    """One reaction or gene, by whose values the samples are coloured.

    ``values`` follow the artifact's samples; a sample the feature has no value
    for is ``nan`` and drawn grey.
    """

    feature_id: str
    display_label: str
    data_type: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class ReductionPlotPresentation:
    """Every display choice needed to rebuild all four reduction plot cards."""

    samples: tuple[ReductionSamplePresentation, ...]
    groups: tuple[ReductionGroupPresentation, ...]
    features: tuple[ReductionFeaturePresentation, ...]
    width: int = 650
    height: int = 400
    background_color: str = "white"
    border_color: str = "white"
    text_color: str = "black"
    show_sample_labels: bool = False
    highlight_group_id: str | None = None
    show_separation_metrics: bool = True
    max_variance_components: int = 20
    loading_component_count: int = 8
    loading_feature_count: int = 50
    loading_sort_component: int | None = None
    biplot_pc_x: int = 1
    biplot_pc_y: int = 2
    biplot_top_arrows: int | None = 5
    biplot_arrow_scale: float = 1.5
    # A PCA scatter draws ``biplot_pc_x`` against ``biplot_pc_y``, with the
    # strongest features as arrows when asked.
    show_biplot_arrows: bool = False
    colouring: ReductionColouring | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.samples, tuple) or not all(
            isinstance(sample, ReductionSamplePresentation) for sample in self.samples
        ):
            raise TypeError("samples must be frozen ReductionSamplePresentation values")
        if not isinstance(self.groups, tuple) or not all(
            isinstance(group, ReductionGroupPresentation) for group in self.groups
        ):
            raise TypeError("groups must be frozen ReductionGroupPresentation values")
        if not isinstance(self.features, tuple) or not all(
            isinstance(feature, ReductionFeaturePresentation)
            for feature in self.features
        ):
            raise TypeError("features must be frozen ReductionFeaturePresentation values")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("plot dimensions must be positive")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.background_color,
                self.border_color,
                self.text_color,
            )
        ):
            raise ValueError("plot colors must be non-empty strings")
        if self.max_variance_components <= 0:
            raise ValueError("max_variance_components must be positive")
        if self.loading_component_count <= 0 or self.loading_feature_count <= 0:
            raise ValueError("loading component/feature counts must be positive")
        if self.loading_sort_component is not None and self.loading_sort_component <= 0:
            raise ValueError("loading_sort_component must be positive when supplied")
        if self.biplot_pc_x <= 0 or self.biplot_pc_y <= 0:
            raise ValueError("biplot component numbers must be positive")
        if self.biplot_pc_x == self.biplot_pc_y:
            raise ValueError("biplot components must be different")
        if self.biplot_top_arrows is not None and self.biplot_top_arrows <= 0:
            raise ValueError("biplot_top_arrows must be positive when supplied")
        if self.biplot_arrow_scale <= 0:
            raise ValueError("biplot_arrow_scale must be positive")


def _validate_inputs(
    artifact: ReductionArtifact, presentation: ReductionPlotPresentation
) -> None:
    if not isinstance(artifact, ReductionArtifact):
        raise TypeError("artifact must be a ReductionArtifact")
    if not isinstance(presentation, ReductionPlotPresentation):
        raise TypeError("presentation must be a ReductionPlotPresentation")
    sample_ids = tuple(sample.sample_id for sample in presentation.samples)
    if sample_ids != artifact.sample_ids:
        raise ValueError("Presentation samples must exactly align with the artifact")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Presentation sample IDs must be unique")
    feature_ids = tuple(feature.feature_id for feature in presentation.features)
    if feature_ids != artifact.feature_names:
        raise ValueError("Presentation features must exactly align with the artifact")
    group_ids = tuple(group.group_id for group in presentation.groups)
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("Presentation group IDs must be unique")
    if presentation.colouring is not None and len(presentation.colouring.values) != len(sample_ids):
        raise ValueError("The colouring needs one value per artifact sample")
    known_samples = set(artifact.sample_ids)
    for group in presentation.groups:
        if not group.group_id or not group.display_name or not group.color:
            raise ValueError("Presentation groups require ID, name, and color")
        if not isinstance(group.sample_ids, tuple):
            raise TypeError("Presentation group memberships must be tuples")
        unknown = set(group.sample_ids) - known_samples
        if unknown:
            raise ValueError(f"Presentation group contains unknown samples: {sorted(unknown)}")


def _sample_lookup(presentation: ReductionPlotPresentation):
    return {sample.sample_id: sample for sample in presentation.samples}


def _feature_lookup(presentation: ReductionPlotPresentation):
    return {feature.feature_id: feature for feature in presentation.features}


def _apply_presentation_style(plot, presentation: ReductionPlotPresentation):
    """The presentation's background, and its text colour when it is not the default.

    On the default black the plot theme's navy and greys stay; another colour,
    chosen for a dark background, is applied to every title and axis.
    """
    plot.background_fill_color = presentation.background_color
    plot.border_fill_color = presentation.border_color
    if presentation.text_color in {"black", "#000000"}:
        return plot
    if plot.title:
        plot.title.text_color = presentation.text_color
    for axis in plot.xaxis + plot.yaxis:
        axis.axis_label_text_color = presentation.text_color
        axis.major_label_text_color = presentation.text_color
        axis.major_tick_line_color = presentation.text_color
        axis.axis_line_color = presentation.text_color
    return plot


def _axis_labels(artifact: ReductionArtifact, first: int = 1, second: int = 2):
    """PC axes with their share of variance; UMAP and t-SNE axes by name alone."""
    method = str(artifact.key.method)
    variance = artifact.pca.explained_variance_ratio if artifact.pca is not None else None
    if method.upper() == "PCA":
        if variance is not None and len(variance) >= max(first, second):
            return (
                f"PC{first} ({variance[first - 1] * 100:.1f}%)",
                f"PC{second} ({variance[second - 1] * 100:.1f}%)",
            )
        return f"PC{first}", f"PC{second}"
    name = "t-SNE" if method.upper().replace("-", "") == "TSNE" else method.upper()
    return f"{name}{first}", f"{name}{second}"


def _sample_tooltip(artifact: ReductionArtifact, first: int = 1, second: int = 2) -> str:
    """A sample's hover: its ID, label and group in the group's colour, and its place."""
    x_name, y_name = (label.split(" (")[0] for label in _axis_labels(artifact, first, second))
    return panel(
        row("ID", "@sample_id"),
        row("Sample", "@display_label"),
        row("Group", "<span style='color:@color;'>@group</span>"),
        section(row(x_name, "@x{0.000}"), row(y_name, "@y{0.000}")),
    )


def _unitless_axes(plot) -> None:
    """No numbers on an axis whose distances carry no unit."""
    for axis in plot.xaxis + plot.yaxis:
        axis.major_label_text_font_size = "0px"
        axis.major_tick_line_color = None


def _style_points(plot, source, *, color: str, size: float, label: str, name: str | None = None):
    """One group's points: translucent, with a white edge so overlaps show."""
    return plot.scatter(
        "x",
        "y",
        source=source,
        size=size,
        fill_color=color,
        fill_alpha=0.8,
        line_color="#ffffff",
        line_width=0.6,
        legend_label=label,
        name=name,
    )


def _legend_outside(plot) -> None:
    """The legend beside the plot, never over its points; a click hides a group."""
    if not plot.legend:
        return
    legend = plot.legend[0]
    if legend not in plot.right:
        plot.add_layout(legend, "right")
    legend.location = "top_left"
    legend.click_policy = "hide"


def _separation_note(plot, coordinates, labels) -> None:
    """How well the groups separate, as a line under the plot rather than its title."""
    if len(set(labels)) < 2 or len(coordinates) < 2 * len(set(labels)):
        return
    metrics = calculate_cluster_metrics(np.asarray(coordinates), list(labels))
    if "silhouette" not in metrics:
        return
    parts = [f"silhouette {metrics['silhouette']:.2f}"]
    if "davies_bouldin" in metrics:
        parts.append(f"Davies-Bouldin {metrics['davies_bouldin']:.2f}")
    if "calinski_harabasz" in metrics:
        parts.append(f"Calinski-Harabasz {metrics['calinski_harabasz']:.1f}")
    quality = get_metric_interpretation("silhouette", metrics["silhouette"]).lower()
    plot.add_layout(
        Title(
            # Two lines, so the note fits a map drawn at half the page's width.
            text=f"Separation of the groups is {quality}:\n{', '.join(parts)}.",
            text_font_size="12px",
            text_font_style="normal",
            text_color=theme.INK_MUTED,
        ),
        "below",
    )


SELECTION_LABELS_CODE = """
const xs = [], ys = [], names = [], lift = []
const xspan = Math.abs(plot.x_range.end - plot.x_range.start) || 1
const yspan = Math.abs(plot.y_range.end - plot.y_range.start) || 1
const placed = []
for (const source of sources) {
  for (const i of source.selected.indices) {
    const x = source.data.x[i], y = source.data.y[i]
    let clashes = 0
    for (const [px, py] of placed) {
      if (Math.abs(px - x) < 0.12 * xspan && Math.abs(py - y) < 0.04 * yspan) clashes += 1
    }
    placed.push([x, y])
    xs.push(x); ys.push(y); names.push(source.data.display_label[i]); lift.push(-12 * clashes)
  }
}
labels.data = {x: xs, y: ys, display_label: names, y_offset: lift}
"""


def _selection_labels(plot, sources) -> ColumnDataSource:
    """Name only the points a box selection picks, each a line clear of its neighbours.

    Every sample stays named in its hover; a thousand labels at once cannot be read.
    """
    labels = ColumnDataSource({"x": [], "y": [], "display_label": [], "y_offset": []})
    plot.text(
        "x",
        "y",
        text="display_label",
        source=labels,
        text_font_size="11px",
        text_color=theme.INK,
        text_baseline="middle",
        x_offset=6,
        y_offset="y_offset",
        name="selection_labels",
    )
    callback = CustomJS(args=dict(sources=list(sources), labels=labels, plot=plot), code=SELECTION_LABELS_CODE)
    for source in sources:
        source.selected.js_on_change("indices", callback)
    plot.add_tools("box_select")
    return labels


def _reduction_sample_presentations(
    data_manager, artifact: ReductionArtifact
) -> tuple[ReductionSamplePresentation, ...]:
    """Freeze each canonical sample beside its display label from the manager."""
    return tuple(
        ReductionSamplePresentation(
            sample_id=sample_id,
            display_label=str(data_manager.get_sample_display_label(sample_id)),
        )
        for sample_id in artifact.sample_ids
    )


def _reduction_group_presentations(
    data_manager, artifact: ReductionArtifact
) -> tuple[ReductionGroupPresentation, ...]:
    """Freeze the canonical group records, colours, and in-matrix membership.

    Membership is intersected with the artifact's samples: a saved group may
    name a sample the aligned matrix does not carry, and the artifact factories
    reject a group that references an unknown sample. Palette fallback follows
    the record order, matching the pre-artifact scatter's colour assignment.
    """
    known = set(artifact.sample_ids)
    group_manager = getattr(data_manager, "group_manager", None)
    records = getattr(group_manager, "group_records", None) or {}
    try:
        colors = dict(data_manager.get_group_colors() or {})
    except Exception:
        colors = {}
    presentations = []
    for index, (group_id, record) in enumerate(records.items()):
        members = tuple(
            sample_id for sample_id in record.sample_ids if sample_id in known
        )
        if not members:
            continue
        color = colors.get(group_id) or DEFAULT_CATEGORICAL_PALETTE[
            index % len(DEFAULT_CATEGORICAL_PALETTE)
        ]
        presentations.append(
            ReductionGroupPresentation(
                group_id=group_id,
                display_name=record.name or group_id,
                color=color,
                sample_ids=members,
            )
        )
    return tuple(presentations)


def _reduction_feature_presentations(
    data_manager,
    artifact: ReductionArtifact,
    *,
    include_hover: bool,
) -> tuple[ReductionFeaturePresentation, ...]:
    """Freeze every artifact feature's display label, hover, and reference URL.

    Display labels and reference URLs are cheap metadata reads, so they are
    resolved for every feature. Hover HTML is only read by the loadings and
    biplot factories, so ``include_hover`` lets the scatter and variance cards
    skip building it for features they never draw.
    """
    element_type = (
        "reaction" if str(artifact.key.data_type).lower() == "reactions" else "gene"
    )
    presentations = []
    for feature_id in artifact.feature_names:
        reference = get_external_reference_data(data_manager, feature_id, element_type)
        hover_html = ""
        if include_hover:
            try:
                hover_html = (
                    data_manager.get_element_hover(
                        element_id=feature_id, element_type=element_type
                    )
                    or ""
                )
            except Exception:
                hover_html = ""
        presentations.append(
            ReductionFeaturePresentation(
                feature_id=feature_id,
                # A reaction by its short name; its identifier stays in the hover.
                display_label=str(reference.get("short_name") or reference["display_label"]),
                hover_html=hover_html,
                external_url=str(reference["external_url"]),
            )
        )
    return tuple(presentations)


def reduction_colouring(
    data_manager, artifact: ReductionArtifact, data_type: str, feature_id: str
) -> ReductionColouring:
    """Read one feature's values on the artifact's samples from the loaded data.

    A reaction is read from the reaction scores and a gene from the expression
    matrix, whichever the reduction was fitted on: the samples are the same, so
    a gene can colour the reaction map. Raises ``KeyError`` for a feature the
    data does not carry.
    """
    source = getattr(data_manager, "reaction_data" if data_type == "reactions" else "expression_data", None)
    if source is None:
        raise KeyError(f"No {data_type} data is loaded")
    if data_type == "reactions":
        frame = source.get_reaction_df()
        if feature_id not in frame.columns:
            raise KeyError(f"No reaction {feature_id!r} in the reaction data")
        column = frame[feature_id]
        element_type = "reaction"
    elif data_type == "expression":
        column = source.matrix().feature_column(feature_id)
        if column is None:
            raise KeyError(f"No gene {feature_id!r} in the expression data")
        element_type = "gene"
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    values = pd.Series(
        pd.to_numeric(column, errors="coerce").to_numpy(dtype=float), index=column.index.map(str)
    ).reindex(list(artifact.sample_ids))
    reference = get_external_reference_data(data_manager, feature_id, element_type)
    return ReductionColouring(
        feature_id=feature_id,
        display_label=str(reference.get("short_name") or reference["display_label"]),
        data_type=data_type,
        values=tuple(float(value) for value in values.to_numpy(dtype=float)),
    )


def build_reduction_presentation(
    data_manager,
    artifact: ReductionArtifact,
    *,
    width: int = 650,
    height: int = 400,
    show_sample_labels: bool = False,
    highlight_group_id: str | None = None,
    show_separation_metrics: bool = True,
    max_variance_components: int = 20,
    loading_component_count: int = 8,
    loading_feature_count: int = 50,
    loading_sort_component: int | None = None,
    biplot_pc_x: int = 1,
    biplot_pc_y: int = 2,
    biplot_top_arrows: int | None = 5,
    biplot_arrow_scale: float = 1.5,
    show_biplot_arrows: bool = False,
    colour_by: tuple[str, str] | None = None,
    include_feature_hover: bool = True,
    features: tuple[ReductionFeaturePresentation, ...] | None = None,
) -> ReductionPlotPresentation:
    """Freeze a document-thread presentation for the artifact reduction cards.

    This reads live presentation state (display labels, group records, colours,
    hover HTML) once on the document thread and returns an immutable value the
    artifact-first factories render without touching a manager. It is the
    production caller that installs the reduction presentation seam.
    ``colour_by`` is a ``(data_type, feature_id)`` pair the samples are
    coloured by, read with :func:`reduction_colouring`. ``features`` are this
    artifact's feature presentations from an earlier call: they depend on the
    model alone, so a caller drawing one artifact again can skip reading them.
    """
    return ReductionPlotPresentation(
        samples=_reduction_sample_presentations(data_manager, artifact),
        groups=_reduction_group_presentations(data_manager, artifact),
        features=features
        if features is not None
        else _reduction_feature_presentations(data_manager, artifact, include_hover=include_feature_hover),
        width=width,
        height=height,
        show_sample_labels=show_sample_labels,
        highlight_group_id=highlight_group_id,
        show_separation_metrics=show_separation_metrics,
        max_variance_components=max_variance_components,
        loading_component_count=loading_component_count,
        loading_feature_count=loading_feature_count,
        loading_sort_component=loading_sort_component,
        biplot_pc_x=biplot_pc_x,
        biplot_pc_y=biplot_pc_y,
        biplot_top_arrows=biplot_top_arrows,
        biplot_arrow_scale=biplot_arrow_scale,
        show_biplot_arrows=show_biplot_arrows,
        colouring=(
            reduction_colouring(data_manager, artifact, *colour_by) if colour_by is not None else None
        ),
    )


def _is_pca(artifact: ReductionArtifact) -> bool:
    return str(artifact.key.method).upper() == "PCA"


def standardized_colouring(
    colouring: ReductionColouring, sample_ids, groups
) -> np.ndarray:
    """The feature as the sample heatmap draws it: standardized, clipped at three.

    The mean and standard deviation are taken over the grouped samples, as the
    heatmap's are, or over every sample when there are no groups; an ungrouped
    sample is placed on the same scale.
    """
    values = np.asarray(colouring.values, dtype=float)
    grouped = {sample for group in groups for sample in group.sample_ids}
    basis = values[[sample in grouped for sample in sample_ids]] if grouped else values
    basis = basis[np.isfinite(basis)]
    spread = float(np.std(basis, ddof=1)) if basis.size > 1 else 0.0
    centre = float(np.mean(basis)) if basis.size else 0.0
    z = (values - centre) / (spread or 1.0)
    return np.clip(z, -FEATURE_REACH, FEATURE_REACH)


def _colour_bar(mapper, title: str) -> ColorBar:
    return ColorBar(
        color_mapper=mapper,
        ticker=BasicTicker(desired_num_ticks=5),
        label_standoff=8,
        border_line_color=None,
        location=(0, 0),
        major_label_text_font_size="8pt",
        title=title,
        title_text_font_style="normal",
        title_text_font_size="9pt",
        title_standoff=8,
        width=12,
    )


def _coloured_points(plot, artifact, presentation, columns, size):
    """Every sample in one layer, coloured by the feature, the extremes drawn last.

    Samples far from the mean are drawn over the rest, so a few high ones are
    never hidden under a crowd of average ones, as scanpy orders them.
    """
    colouring = presentation.colouring
    assert colouring is not None
    sample_lookup = _sample_lookup(presentation)
    membership: dict[str, ReductionGroupPresentation] = {}
    for group in presentation.groups:
        for sample_id in group.sample_ids:
            membership.setdefault(sample_id, group)
    z = standardized_colouring(colouring, artifact.sample_ids, presentation.groups)
    order = np.argsort(np.nan_to_num(np.abs(z), nan=-1.0), kind="stable")
    ids = [artifact.sample_ids[index] for index in order]
    source = ColumnDataSource(
        {
            "x": artifact.coordinates[order, columns[0]],
            "y": artifact.coordinates[order, columns[1]],
            "sample_id": ids,
            "display_label": [sample_lookup[sample_id].display_label for sample_id in ids],
            "group": [membership[s].display_name if s in membership else "Unassigned" for s in ids],
            "color": [membership[s].color if s in membership else theme.NOT_SIGNIFICANT for s in ids],
            "value": np.asarray(colouring.values, dtype=float)[order],
            "z": z[order],
        }
    )
    mapper = LinearColorMapper(
        palette=_loading_palette(), low=-FEATURE_REACH, high=FEATURE_REACH, nan_color=theme.NOT_SIGNIFICANT
    )
    renderer = plot.scatter(
        "x",
        "y",
        source=source,
        size=size,
        fill_color=transform("z", mapper),
        line_color=theme.MARK_EDGE,
        line_width=0.4,
        name="sample_points",
    )
    plot.add_layout(_colour_bar(mapper, "Standardized score"), "right")
    return source, renderer


def plot_reduction_artifact(
    artifact: ReductionArtifact, presentation: ReductionPlotPresentation
):
    """Build the reduction scatter directly from immutable artifact values.

    A PCA draws ``biplot_pc_x`` against ``biplot_pc_y`` and, with
    ``show_biplot_arrows``, the strongest features as arrows; other methods draw
    their first two components. With a ``colouring`` the samples take the
    feature's colours instead of their groups'.
    """
    _validate_inputs(artifact, presentation)
    if artifact.coordinates.shape[1] < 2:
        raise ValueError("Reduction scatter requires at least two components")
    pca = _is_pca(artifact)
    first, second = (presentation.biplot_pc_x, presentation.biplot_pc_y) if pca else (1, 2)
    if max(first, second) > artifact.coordinates.shape[1]:
        raise ValueError(
            f"Invalid PC selection ({first}, {second}) for "
            f"{artifact.coordinates.shape[1]} components"
        )
    columns = (first - 1, second - 1)
    coordinates = artifact.coordinates
    sample_lookup = _sample_lookup(presentation)
    colouring = presentation.colouring
    title = f"{artifact.key.method} of {artifact.key.data_type.capitalize()} Data"
    if colouring is not None:
        # On a line of its own, so the title fits a map at half the page's width.
        title += f"\ncoloured by {colouring.display_label}"
    plot = apply_academic_style(
        figure(
            width=presentation.width,
            height=presentation.height,
            title=title,
            tools="pan,box_zoom,wheel_zoom,reset,save",
            match_aspect=True,
        )
    )
    size = point_size(len(artifact.sample_ids))

    separation_coordinates: list[np.ndarray] = []
    separation_labels: list[str] = []
    sources = []
    point_renderers = []
    groups = presentation.groups or (
        ReductionGroupPresentation("", "All samples", DEFAULT_SINGLE_COLOR, tuple(artifact.sample_ids)),
    )
    for index, group in enumerate(groups):
        member_set = set(group.sample_ids)
        indices = [
            row for row, sample_id in enumerate(artifact.sample_ids) if sample_id in member_set
        ]
        if not indices:
            continue
        values = coordinates[np.ix_(indices, columns)]
        if presentation.groups:
            separation_coordinates.extend(values)
            separation_labels.extend([group.group_id] * len(values))
        if colouring is not None:
            continue
        canonical_ids = [artifact.sample_ids[row] for row in indices]
        color = group.color or DEFAULT_CATEGORICAL_PALETTE[index % len(DEFAULT_CATEGORICAL_PALETTE)]
        source = ColumnDataSource(
            {
                "x": values[:, 0],
                "y": values[:, 1],
                "sample_id": canonical_ids,
                "display_label": [sample_lookup[sample_id].display_label for sample_id in canonical_ids],
                "group_id": [group.group_id] * len(indices),
                "group": [group.display_name] * len(indices),
                "color": [color] * len(indices),
            }
        )
        point_renderers.append(
            _style_points(
                plot, source, color=color, size=size, label=f"{group.display_name} ({len(indices):,})"
            )
        )
        sources.append(source)
        if presentation.highlight_group_id and presentation.highlight_group_id == group.group_id:
            centroid = values.mean(axis=0)
            plot.text(
                x=[centroid[0]],
                y=[centroid[1]],
                text=[group.display_name],
                text_font_size="13px",
                text_baseline="middle",
                text_align="center",
                text_color=theme.INK,
                text_font_style="bold",
            )
    tooltip = _sample_tooltip(artifact, first, second)
    if colouring is not None:
        source, renderer = _coloured_points(plot, artifact, presentation, columns, size)
        sources, point_renderers = [source], [renderer]
        x_name, y_name = (label.split(" (")[0] for label in _axis_labels(artifact, first, second))
        tooltip = panel(
            row("ID", "@sample_id"),
            row("Sample", "@display_label"),
            row("Group", "<span style='color:@color;'>@group</span>"),
            section(row(escape(colouring.display_label), "@value{0.000}"), row("Standardized", "@z{+0.00}")),
            section(row(x_name, "@x{0.000}"), row(y_name, "@y{0.000}")),
        )

    if presentation.show_sample_labels:
        _selection_labels(plot, sources)
    if pca and presentation.show_biplot_arrows and artifact.pca is not None and artifact.pca.loadings is not None:
        span = coordinates[:, list(columns)]
        _draw_arrows(
            plot,
            artifact,
            presentation,
            columns,
            max(float(np.ptp(span[:, 0])), float(np.ptp(span[:, 1])), 1.0),
        )
    # One panel, the nearest sample's, rather than a column of them where samples crowd.
    plot.add_tools(HoverTool(renderers=point_renderers, tooltips=tooltip, limit=1, name="sample_hover"))
    x_label, y_label = _axis_labels(artifact, first, second)
    plot.xaxis.axis_label = x_label
    plot.yaxis.axis_label = y_label
    if not pca:
        _unitless_axes(plot)
        plot.grid.grid_line_color = None
    _legend_outside(plot)
    if presentation.show_separation_metrics:
        _separation_note(plot, separation_coordinates, separation_labels)
    return _apply_presentation_style(plot, presentation)


def _require_pca_variance(artifact: ReductionArtifact) -> None:
    if artifact.pca is None or artifact.pca.explained_variance_ratio is None:
        raise ValueError("This plot requires a PCA artifact with explained variance")


def _require_pca_loadings(artifact: ReductionArtifact) -> None:
    if artifact.pca is None or artifact.pca.loadings is None:
        raise ValueError("This plot requires a PCA artifact with loadings")


def plot_pca_variance_artifact(
    artifact: ReductionArtifact, presentation: ReductionPlotPresentation
):
    """Each component's share of the variance in percent, and the running total.

    The first components hold the story, so at most ``max_variance_components``
    are drawn (20 by default); the cumulative line is marked where it passes 50%
    and 80%. The download covers every retained component.
    """
    _validate_inputs(artifact, presentation)
    _require_pca_variance(artifact)
    assert artifact.pca is not None
    assert artifact.pca.explained_variance_ratio is not None
    all_variance = artifact.pca.explained_variance_ratio
    retained = len(all_variance)
    count = min(retained, presentation.max_variance_components)
    variance = all_variance[:count]
    cumulative = np.cumsum(all_variance)[:count]
    components = np.arange(1, count + 1)
    source = ColumnDataSource(
        {
            "components": components,
            "explained_variance": variance,
            "cumulative_variance": cumulative,
            "explained_percent": variance * 100,
            "cumulative_percent": cumulative * 100,
            "explained_variance_str": [f"{value * 100:.1f}%" for value in variance],
            "cumulative_variance_str": [f"{value * 100:.1f}%" for value in cumulative],
        }
    )
    title = f"PCA explained variance of {artifact.data_source_name}"
    if count < retained:
        title += f" (first {count} of {retained} PCs from {artifact.n_variable_features_used} features)"
    else:
        title += f" ({retained} PCs from {artifact.n_variable_features_used} features)"
    plot = apply_academic_style(
        figure(
            width=presentation.width,
            height=presentation.height,
            x_range=Range1d(0.5, count + 0.5),
            y_range=Range1d(0, 100),
            title=title,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
    )
    plot.xgrid.grid_line_color = None
    plot.vbar(
        x="components",
        top="explained_percent",
        width=0.7,
        source=source,
        legend_label="Each component",
        fill_color=theme.FRAME_LINE,
        line_color=None,
    )
    plot.line(
        x="components",
        y="cumulative_percent",
        source=source,
        legend_label="Cumulative",
        line_width=2,
        color=theme.INK,
    )
    plot.scatter(
        x="components",
        y="cumulative_percent",
        source=source,
        color=theme.INK,
        size=5,
        marker="circle",
    )
    for share in VARIANCE_MARKS:
        reached = np.nonzero(cumulative >= share)[0]
        plot.add_layout(
            Span(
                location=share * 100,
                dimension="width",
                line_color=theme.THRESHOLD_LINE_COLOR,
                line_dash=theme.THRESHOLD_LINE_DASH,
                line_width=theme.THRESHOLD_LINE_WIDTH,
            )
        )
        text = (
            f"{share:.0%} by PC{reached[0] + 1}"
            if reached.size
            else f"{share:.0%} not reached in {count} PCs"
        )
        plot.add_layout(
            LabelSet(
                x="x",
                y="y",
                text="text",
                source=ColumnDataSource({"x": [count + 0.4], "y": [share * 100], "text": [text]}),
                text_align="right",
                text_baseline="bottom",
                text_font_size="11px",
                text_color=theme.INK_MUTED,
                y_offset=3,
            )
        )
    plot.add_tools(
        HoverTool(
            tooltips=panel(
                row("Component", "@components"),
                row("Variance explained", "@explained_variance_str"),
                row("Cumulative", "@cumulative_variance_str"),
            )
        )
    )
    plot.xaxis.axis_label = "Principal component"
    plot.yaxis.axis_label = "Variance explained (%)"
    # Below the plot, clear of the 50% and 80% marks at its right edge.
    legend = plot.legend[0]
    plot.add_layout(legend, "below")
    legend.orientation = "horizontal"
    legend.location = "center"
    return _apply_presentation_style(plot, presentation)


def _loading_columns(artifact: ReductionArtifact, presentation: ReductionPlotPresentation):
    """The strongest features over the shown components, one row per direction.

    ``loading_sort_component`` orders the features by that component's absolute
    loadings, or by default by their absolute loadings summed over the shown
    components. Features that load identically on the shown components (several
    reactions Compass scores alike, one flux through a chain) share a row named
    after the first, so the rows show distinct directions rather than one
    direction repeated. Returns the component count and each row's members.
    """
    assert artifact.pca is not None and artifact.pca.loadings is not None
    loadings = artifact.pca.loadings
    component_count = min(presentation.loading_component_count, loadings.shape[0])
    shown = loadings[:component_count]
    sort = presentation.loading_sort_component
    if sort is not None:
        if sort > loadings.shape[0]:
            raise ValueError(f"PC for sorting must be between 1 and {loadings.shape[0]}")
        strength = np.abs(loadings[sort - 1])
    else:
        strength = np.sum(np.abs(shown), axis=0)
    feature_count = min(presentation.loading_feature_count, loadings.shape[1])
    members: dict[tuple, list[int]] = {}
    for index in np.argsort(-strength, kind="stable"):
        key = tuple(np.round(shown[:, index], 9))
        if key in members:
            members[key].append(int(index))
        elif len(members) < feature_count:
            members[key] = [int(index)]
    return component_count, list(members.values())


def _loading_palette() -> list[str]:
    """The app's own scale, blue through white to red, as its Delta and effects are drawn."""
    half = 128
    return gradient_palette(theme.DOWN_REGULATED, theme.NEUTRAL, half)[:-1] + gradient_palette(
        theme.NEUTRAL, theme.UP_REGULATED, half
    )


def plot_pca_loadings_artifact(
    artifact: ReductionArtifact, presentation: ReductionPlotPresentation
):
    """The strongest features' loadings on the first components, as a heatmap.

    One row per feature (features loading alike share one, see
    :func:`_loading_columns`), one column per component, each cell coloured by
    its loading on the app's blue to red scale, symmetric about zero. The
    download is the complete matrix.
    """
    _validate_inputs(artifact, presentation)
    _require_pca_loadings(artifact)
    assert artifact.pca is not None
    assert artifact.pca.loadings is not None
    if artifact.pca.loadings.size == 0:
        raise ValueError("Loadings matrix is empty")
    component_count, groups = _loading_columns(artifact, presentation)
    loadings = artifact.pca.loadings
    feature_lookup = _feature_lookup(presentation)
    variance = artifact.pca.explained_variance_ratio
    components = [f"PC{index + 1}" for index in range(component_count)]
    component_labels = [
        f"PC{index + 1} ({variance[index] * 100:.1f}%)" if variance is not None and len(variance) > index else f"PC{index + 1}"
        for index in range(component_count)
    ]

    rows, used = [], set()
    cells = {key: [] for key in ("feature", "component", "loading", "feature_id", "display_label", "hover", "external_url")}
    for group in groups:
        first = artifact.feature_names[group[0]]
        presentation_of = feature_lookup[first]
        name = presentation_of.display_label
        if len(group) > 1:
            name += f" and {len(group) - 1} alike"
        # A name used twice keeps a row of its own.
        while name in used:
            name += "\u200b"
        used.add(name)
        rows.append(name)
        twins = ", ".join(feature_lookup[artifact.feature_names[i]].display_label for i in group[1:11])
        # The feature's own panel, as the graphs show it; the template adds the
        # cell's component and loading.
        hover = (
            hover_panel_inner(presentation_of.hover_html)
            if presentation_of.hover_html
            else row("ID", escape(first)) + row("Feature", escape(presentation_of.display_label))
        )
        if twins:
            hover += section(row("Loading alike", escape(twins)))
        for component, label in enumerate(component_labels):
            cells["feature"].append(name)
            cells["component"].append(label)
            cells["loading"].append(float(loadings[component, group[0]]))
            cells["feature_id"].append(first)
            cells["display_label"].append(presentation_of.display_label)
            cells["hover"].append(hover)
            cells["external_url"].append(presentation_of.external_url)

    reach = max([abs(value) for value in cells["loading"]] + [1e-9])
    mapper = LinearColorMapper(palette=_loading_palette(), low=-reach, high=reach)
    sort = presentation.loading_sort_component
    order = f"ordered by |PC{sort}|" if sort is not None else "ordered by their total over these components"
    plot = apply_academic_style(
        figure(
            width=presentation.width,
            # Every row keeps a readable height, however many are asked for.
            height=max(presentation.height, LOADING_ROW_HEIGHT * len(rows) + 120),
            x_range=FactorRange(factors=component_labels),
            y_range=FactorRange(factors=list(reversed(rows)), range_padding=0.0),
            title=(
                f"PCA loadings of {artifact.data_source_name}, the {len(rows)} strongest features "
                f"over {components[0]} to {components[-1]}, {order}"
                if component_count > 1
                else f"PCA loadings of {artifact.data_source_name}, the {len(rows)} strongest features on PC1"
            ),
            tools="save,pan,box_zoom,wheel_zoom,reset",
            x_axis_location="above",
        ),
        suppress_grid=True,
    )
    plot.axis.axis_line_color = None
    plot.axis.major_tick_line_color = None
    yaxis = plot.yaxis[0]
    yaxis.major_label_text_font_size = "11px"
    yaxis.major_label_text_color = theme.INK
    renderer = plot.rect(
        x="component",
        y="feature",
        width=1,
        height=1,
        source=ColumnDataSource(cells),
        fill_color=transform("loading", mapper),
        line_color="#ffffff",
        line_width=1,
    )
    plot.add_layout(
        ColorBar(
            color_mapper=mapper,
            ticker=BasicTicker(desired_num_ticks=5),
            label_standoff=8,
            border_line_color=None,
            location=(0, 0),
            major_label_text_font_size="8pt",
            title="Loading",
            title_text_font_style="normal",
            title_text_font_size="9pt",
            title_standoff=8,
            width=12,
        ),
        "right",
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=panel(
                "@hover{safe}",
                section(row("Loading on @component", "@loading{0.000}")),
            ),
        )
    )
    return _apply_presentation_style(plot, presentation)


FEATURE_TAP_CODE = """
const y = cb_obj.y
if (y >= 0 && y < rows.length) {
  const index = Math.floor(y)
  choice.data = {y: [index + 0.5]}
  if (feature_event) {
    window.dispatchEvent(new CustomEvent(feature_event, {detail: {feature: ids[index], label: labels[index], data_type}}))
  }
} else if (component_event && y >= base) {
  const column = Math.floor(cb_obj.x)
  if (column >= 0 && column < count) {
    window.dispatchEvent(new CustomEvent(component_event, {detail: {component: column + 1, data_type}}))
  }
}
"""


def feature_rows_source(rows: list[list[str]], centres: list[float]) -> ColumnDataSource:
    """Every feature a row holds beside the row's centre, so the page can outline
    the row of whichever feature colours the samples."""
    features, ys = [], []
    for members, centre in zip(rows, centres):
        features.extend(members)
        ys.extend([centre] * len(members))
    return ColumnDataSource({"feature": features, "y": ys}, name=FEATURE_ROWS)


def plot_pca_components_artifact(
    artifact: ReductionArtifact,
    presentation: ReductionPlotPresentation,
    *,
    plotted: tuple[int, ...] = (),
    feature_event: str | None = None,
    component_event: str | None = None,
):
    """Each component's share of the variance over its loadings, in one figure.

    The bars are the variance each component explains and the cells below them
    the strongest features' loadings on it, as ComplexHeatmap sets a bar
    annotation over a heatmap's columns (Gu, Eils and Schlesner 2016): a
    component's bar and its column share one position, so how much a component
    explains reads beside which features make it. Rows are chosen and merged as
    in :func:`plot_pca_loadings_artifact`, and named short, the full name in the
    hover. The ``plotted`` components (numbered from 1) are outlined.

    A click on a row outlines it and dispatches ``feature_event`` with the
    feature; a click on a bar dispatches ``component_event`` with the component.
    """
    _validate_inputs(artifact, presentation)
    _require_pca_loadings(artifact)
    _require_pca_variance(artifact)
    assert artifact.pca is not None
    assert artifact.pca.loadings is not None and artifact.pca.explained_variance_ratio is not None
    if artifact.pca.loadings.size == 0:
        raise ValueError("Loadings matrix is empty")
    count, groups = _loading_columns(artifact, presentation)
    loadings = artifact.pca.loadings
    lookup = _feature_lookup(presentation)
    variance = np.asarray(artifact.pca.explained_variance_ratio[:count], dtype=float)
    cumulative = np.cumsum(artifact.pca.explained_variance_ratio)[:count]
    components = [f"PC{index + 1}" for index in range(count)]
    noun = "reaction" if str(artifact.key.data_type).lower() == "reactions" else "gene"
    n = len(groups)

    # Rows run top to bottom from the strongest; row i is centred at n - i - 0.5,
    # so the arrays below are kept bottom to top, as the y axis counts.
    names, labels, ticks, hovers, ids = [], [], [], [], []
    for members in groups:
        first = artifact.feature_names[members[0]]
        feature = lookup[first]
        names.append(feature.display_label)
        ticks.append(short_row_label(feature.display_label, len(members) - 1, LABEL_LIMIT))
        hover = (
            hover_panel_inner(feature.hover_html)
            if feature.hover_html
            else row("ID", escape(first)) + row("Feature", escape(feature.display_label))
        )
        twins = ", ".join(lookup[artifact.feature_names[i]].display_label for i in members[1:11])
        if twins:
            hover += section(row("Loading alike", escape(twins)))
        hovers.append(hover)
        ids.append(first)
    cells: dict[str, list] = {"component": [], "y": [], "loading": [], "hover": []}
    for index, members in enumerate(groups):
        for column, component in enumerate(components):
            cells["component"].append(component)
            cells["y"].append(n - index - 0.5)
            cells["loading"].append(float(loadings[column, members[0]]))
            cells["hover"].append(hovers[index])
    reach = max([abs(value) for value in cells["loading"]] + [1e-9])
    mapper = LinearColorMapper(palette=_loading_palette(), low=-reach, high=reach)

    base, tall = n + 0.5, COMPONENT_BAR_ROWS
    tops = base + tall * variance / (variance[0] or 1.0)
    ceiling = base + tall + 2.0
    plot = apply_academic_style(
        figure(
            width=presentation.width,
            height=int(LOADING_ROW_HEIGHT * (n + tall + 2.2) + 110),
            x_range=FactorRange(factors=components),
            y_range=Range1d(0, ceiling),
            title=f"Variance explained (bars) and the {n} strongest {noun}s' loadings (cells)",
            tools="save",
        ),
        suppress_grid=True,
    )
    cell_renderer = plot.rect(
        x="component",
        y="y",
        width=1,
        height=1,
        source=ColumnDataSource(cells),
        fill_color=transform("loading", mapper),
        line_color="#ffffff",
        line_width=1,
    )
    bars = ColumnDataSource(
        {
            "component": components,
            "top": tops,
            "share": [f"{value * 100:.1f}%" for value in variance],
            "cumulative": [f"{value * 100:.1f}%" for value in cumulative],
            "label_y": tops + 0.25,
        }
    )
    bar_renderer = plot.vbar(
        x="component", bottom=base, top="top", width=0.62, source=bars,
        fill_color=theme.FRAME_LINE, line_color=None,
    )
    plot.text(
        x="component", y="label_y", text="share", source=bars,
        text_align="center", text_baseline="bottom", text_font_size="10px", text_color=theme.INK,
    )
    for component in plotted:
        if 1 <= component <= count:
            plot.rect(
                x=[components[component - 1]], y=[(base + tall + 1.6) / 2], width=0.98,
                height=base + tall + 1.6, fill_color=None, line_color=theme.SELECTED, line_width=2.5,
            )
    choice = ColumnDataSource({"y": []}, name=FEATURE_CHOICE)
    plot.rect(
        x=count / 2, y="y", width=count, height=1, source=choice,
        fill_color=None, line_color=theme.SELECTED, line_width=2.5,
    )
    rows = feature_rows_source(
        [[artifact.feature_names[i] for i in members] for members in groups],
        [n - index - 0.5 for index in range(n)],
    )
    plot.tags = [*plot.tags, rows]

    plot.yaxis.ticker = FixedTicker(ticks=[n - index - 0.5 for index in range(n)])
    plot.yaxis.major_label_overrides = {n - index - 0.5: tick for index, tick in enumerate(ticks)}
    plot.axis.axis_line_color = None
    plot.axis.major_tick_line_color = None
    plot.axis.minor_tick_line_color = None
    plot.yaxis.major_label_text_font_size = "11px"
    plot.yaxis.major_label_text_color = theme.INK
    plot.xaxis.major_label_text_color = theme.INK
    plot.add_layout(_colour_bar(mapper, "Loading"), "right")
    plot.add_tools(
        HoverTool(
            renderers=[cell_renderer],
            tooltips=panel("@hover{safe}", section(row("Loading on @component", "@loading{0.000}"))),
        ),
        HoverTool(
            renderers=[bar_renderer],
            tooltips=panel(
                row("Component", "@component"),
                row("Variance explained", "@share"),
                row("Cumulative", "@cumulative"),
            ),
        ),
    )
    if feature_event or component_event:
        plot.js_on_event(
            "tap",
            CustomJS(
                args=dict(
                    choice=choice,
                    # Bottom to top, as the rows' y positions count.
                    rows=list(reversed(ticks)),
                    ids=list(reversed(ids)),
                    labels=list(reversed(names)),
                    data_type=str(artifact.key.data_type),
                    feature_event=feature_event,
                    component_event=component_event,
                    base=base,
                    count=count,
                ),
                code=FEATURE_TAP_CODE,
            ),
        )
    return _apply_presentation_style(plot, presentation)


def biplot_arrows(loadings_x: np.ndarray, loadings_y: np.ndarray, limit: int | None) -> list[int]:
    """The longest arrow in each 45 degree sector, at least 15% of the longest.

    The strongest loadings often point the same way (tied reactions, one chain),
    so the top few by length would draw one arrow several times; one per sector
    shows the directions the variance runs in. At most ``limit``, longest first.
    """
    magnitudes = np.hypot(loadings_x, loadings_y)
    if magnitudes.size == 0 or float(magnitudes.max()) == 0:
        return []
    floor = BIPLOT_MINIMUM_SHARE * float(magnitudes.max())
    angles = np.mod(np.arctan2(loadings_y, loadings_x), 2 * np.pi)
    sectors = np.floor(angles / (2 * np.pi / BIPLOT_SECTORS)).astype(int) % BIPLOT_SECTORS
    chosen = []
    for sector in range(BIPLOT_SECTORS):
        members = np.nonzero((sectors == sector) & (magnitudes >= floor))[0]
        if members.size:
            chosen.append(int(members[np.argmax(magnitudes[members])]))
    chosen.sort(key=lambda index: -magnitudes[index])
    return chosen[:limit] if limit is not None else chosen


def _draw_arrows(plot, artifact: ReductionArtifact, presentation, columns, score_range: float) -> None:
    """The strongest features as arrows from the origin, each named at its tip.

    Arrows are scaled so the longest reaches a fifth of the samples' spread
    times ``biplot_arrow_scale``; see :func:`biplot_arrows` for which are drawn.
    """
    assert artifact.pca is not None and artifact.pca.loadings is not None
    x_index, y_index = columns
    pc_x, pc_y = x_index + 1, y_index + 1
    loadings_x = artifact.pca.loadings[x_index]
    loadings_y = artifact.pca.loadings[y_index]
    magnitudes = np.sqrt(loadings_x**2 + loadings_y**2)
    max_loading = float(np.max(magnitudes)) if magnitudes.size else 1.0
    if max_loading == 0:
        max_loading = 1.0
    scale = score_range / max_loading * 0.20 * presentation.biplot_arrow_scale
    feature_lookup = _feature_lookup(presentation)
    chosen = biplot_arrows(loadings_x, loadings_y, presentation.biplot_top_arrows)
    arrows = pd.DataFrame(
        {
            "x_end": loadings_x[chosen] * scale,
            "y_end": loadings_y[chosen] * scale,
            "feature_id": [artifact.feature_names[index] for index in chosen],
            "magnitude": magnitudes[chosen],
            "x_start": 0.0,
            "y_start": 0.0,
        }
    )
    arrows["display_label"] = [feature_lookup[value].display_label for value in arrows["feature_id"]]
    # The feature's own panel, as the graphs show it, with its loadings added.
    arrows["hover"] = [
        (extend(feature_lookup[feature].hover_html, facts) if feature_lookup[feature].hover_html
         else panel(row("ID", escape(feature)), row("Feature", escape(feature_lookup[feature].display_label)), facts))
        for feature, facts in zip(
            arrows["feature_id"],
            (
                section(
                    row(f"Loading on PC{pc_x}", f"{loadings_x[index]:.3f}"),
                    row(f"Loading on PC{pc_y}", f"{loadings_y[index]:.3f}"),
                    row("Arrow length", f"{magnitudes[index]:.3f}"),
                )
                for index in chosen
            ),
        )
    ]
    arrows["external_url"] = [feature_lookup[value].external_url for value in arrows["feature_id"]]
    # Each name at its tip, set outwards along its arrow so it clears the others.
    arrows["align"] = ["left" if x >= 0 else "right" for x in arrows["x_end"]]
    arrows["baseline"] = ["bottom" if y >= 0 else "top" for y in arrows["y_end"]]
    arrows["x_offset"] = [4 if x >= 0 else -4 for x in arrows["x_end"]]
    arrows["y_offset"] = [-3 if y >= 0 else 3 for y in arrows["y_end"]]
    arrow_source = ColumnDataSource(arrows)
    arrow_color = theme.INK_MUTED
    plot.add_layout(
        Arrow(
            end=VeeHead(size=7, fill_color=arrow_color, line_color=arrow_color, line_width=1),
            x_start="x_start",
            y_start="y_start",
            x_end="x_end",
            y_end="y_end",
            source=arrow_source,
            line_color=arrow_color,
            line_width=1.5,
        )
    )
    arrow_hover_renderer = plot.scatter(
        x="x_end", y="y_end", source=arrow_source, size=10, alpha=0, name="arrow_endpoints"
    )
    plot.text(
        x="x_end",
        y="y_end",
        text="display_label",
        source=arrow_source,
        text_font_size="11px",
        text_color=theme.INK,
        text_align="align",
        text_baseline="baseline",
        x_offset="x_offset",
        y_offset="y_offset",
    )
    plot.add_tools(HoverTool(tooltips="@hover{safe}", renderers=[arrow_hover_renderer], name="arrow_hover"))


def plot_pca_biplot_artifact(
    artifact: ReductionArtifact, presentation: ReductionPlotPresentation
):
    """Build the PCA biplot from frozen scores, loadings, and presentation."""
    _validate_inputs(artifact, presentation)
    _require_pca_loadings(artifact)
    assert artifact.pca is not None
    assert artifact.pca.loadings is not None
    pc_x = presentation.biplot_pc_x
    pc_y = presentation.biplot_pc_y
    if pc_x > artifact.coordinates.shape[1] or pc_y > artifact.coordinates.shape[1]:
        raise ValueError(
            f"Invalid PC selection ({pc_x}, {pc_y}) for "
            f"{artifact.coordinates.shape[1]} components"
        )
    x_index, y_index = pc_x - 1, pc_y - 1
    sample_lookup = _sample_lookup(presentation)
    assignment: dict[str, ReductionGroupPresentation] = {}
    for group in presentation.groups:
        for sample_id in group.sample_ids:
            assignment.setdefault(sample_id, group)
    unassigned = "Unassigned" if presentation.groups else "N/A"
    rows = []
    for index, sample_id in enumerate(artifact.sample_ids):
        group = assignment.get(sample_id)
        rows.append(
            {
                "x": artifact.coordinates[index, x_index],
                "y": artifact.coordinates[index, y_index],
                "sample_id": sample_id,
                "display_label": sample_lookup[sample_id].display_label,
                "group_id": group.group_id if group else "",
                "group": group.display_name if group else unassigned,
                "color": group.color if group else DEFAULT_SINGLE_COLOR,
            }
        )
    scores = pd.DataFrame(rows)
    title = f"PCA biplot of {artifact.data_source_name}, PC{pc_x} and PC{pc_y}"
    plot = apply_academic_style(
        figure(
            width=presentation.width,
            height=presentation.height,
            title=title,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            match_aspect=True,
        )
    )
    x_label, y_label = _axis_labels(artifact, pc_x, pc_y)
    plot.xaxis.axis_label = x_label
    plot.yaxis.axis_label = y_label
    size = point_size(len(scores))
    sample_renderers, sample_sources = [], []
    for group_name in dict.fromkeys(scores["group"].tolist()):
        group_rows = scores[scores["group"] == group_name]
        source = ColumnDataSource(group_rows)
        sample_renderers.append(
            _style_points(
                plot,
                source,
                color=group_rows["color"].iloc[0],
                size=size,
                label=f"{group_name} ({len(group_rows):,})",
                name="sample_points",
            )
        )
        sample_sources.append(source)
    if presentation.show_sample_labels:
        _selection_labels(plot, sample_sources)
    if presentation.highlight_group_id:
        highlighted = scores[scores["group_id"] == presentation.highlight_group_id]
        if not highlighted.empty:
            group = next(
                item for item in presentation.groups if item.group_id == presentation.highlight_group_id
            )
            plot.text(
                x=[highlighted["x"].mean()],
                y=[highlighted["y"].mean()],
                text=[group.display_name],
                text_font_size="13px",
                text_baseline="middle",
                text_align="center",
                text_color=theme.INK,
                text_font_style="bold",
            )

    score_range = max(
        float(scores["x"].max() - scores["x"].min()),
        float(scores["y"].max() - scores["y"].min()),
        1.0,
    )
    _draw_arrows(plot, artifact, presentation, (x_index, y_index), score_range)
    plot.add_tools(
        HoverTool(tooltips=_sample_tooltip(artifact, pc_x, pc_y), renderers=sample_renderers, name="sample_hover", limit=1),
    )
    if presentation.show_separation_metrics:
        valid = scores[~scores["group"].isin(["Unassigned", "N/A"])]
        _separation_note(plot, valid[["x", "y"]].to_numpy(), valid["group_id"].tolist())
    _legend_outside(plot)
    return _apply_presentation_style(plot, presentation)


__all__ = [
    "FEATURE_CHOICE",
    "FEATURE_ROWS",
    "ReductionColouring",
    "ReductionFeaturePresentation",
    "ReductionGroupPresentation",
    "ReductionPlotPresentation",
    "ReductionSamplePresentation",
    "build_reduction_presentation",
    "feature_rows_source",
    "plot_pca_biplot_artifact",
    "plot_pca_components_artifact",
    "plot_pca_loadings_artifact",
    "plot_pca_variance_artifact",
    "plot_reduction_artifact",
    "reduction_colouring",
    "standardized_colouring",
]
