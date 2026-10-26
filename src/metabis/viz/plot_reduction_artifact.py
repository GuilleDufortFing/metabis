"""Artifact-first Bokeh factories for dimensionality-reduction results.

The sample scatters follow how single-cell tools draw an embedding: points sized
by how many there are, translucent with a white edge so overlaps show, the legend
outside with each group's count, both axes at one scale, PCA axes with their
share of variance, and UMAP and t-SNE axes with no numbers, which carry no unit
(as scanpy and Seurat draw them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape

import numpy as np
import pandas as pd
from bokeh.models import (
    Arrow,
    ColumnDataSource,
    CustomJS,
    FactorRange,
    HoverTool,
    LabelSet,
    Range1d,
    Span,
    Title,
    VeeHead,
)
from bokeh.palettes import Category10
from bokeh.plotting import figure

from ..analysis.dimensionality_reduction import ReductionArtifact
from ..managers.group_palettes import OKABE_ITO
from ..managers.hover.panel import extend, panel, row, section
from ..visual_config import theme
from .click_behavior import get_external_reference_data
from .plot_reduced_dimension import (
    calculate_cluster_metrics,
    get_metric_interpretation,
)
from .plot_utils import apply_academic_style

DEFAULT_CATEGORICAL_PALETTE = OKABE_ITO
DEFAULT_SINGLE_COLOR = OKABE_ITO[3]  # Okabe-Ito blue, for samples with no group
# Category10 is kept for callers that imported it from here.
LEGACY_CATEGORICAL_PALETTE = Category10[10]
VARIANCE_MARKS = (0.5, 0.8)  # where the cumulative share is marked
BIPLOT_SECTORS = 8  # one arrow per 45 degrees
BIPLOT_MINIMUM_SHARE = 0.15  # of the longest arrow
LOADING_BAR_COLOR = theme.INK_MUTED  # a loading's sign is a direction, not up or down
LOADING_ROW_HEIGHT = 20


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
    loading_component_count: int = 2
    loading_feature_count: int = 12
    loading_sort_component: int | None = None
    biplot_pc_x: int = 1
    biplot_pc_y: int = 2
    biplot_top_arrows: int | None = 5
    biplot_arrow_scale: float = 1.5

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
            text=f"Separation of the groups is {quality} ({', '.join(parts)}).",
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
    loading_component_count: int = 2,
    loading_feature_count: int = 12,
    loading_sort_component: int | None = None,
    biplot_pc_x: int = 1,
    biplot_pc_y: int = 2,
    biplot_top_arrows: int | None = 5,
    biplot_arrow_scale: float = 1.5,
    include_feature_hover: bool = True,
) -> ReductionPlotPresentation:
    """Freeze a document-thread presentation for the artifact reduction cards.

    This reads live presentation state (display labels, group records, colours,
    hover HTML) once on the document thread and returns an immutable value the
    artifact-first factories render without touching a manager. It is the
    production caller that installs the reduction presentation seam.
    """
    return ReductionPlotPresentation(
        samples=_reduction_sample_presentations(data_manager, artifact),
        groups=_reduction_group_presentations(data_manager, artifact),
        features=_reduction_feature_presentations(
            data_manager, artifact, include_hover=include_feature_hover
        ),
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
    )


def plot_reduction_artifact(
    artifact: ReductionArtifact, presentation: ReductionPlotPresentation
):
    """Build the reduction scatter directly from immutable artifact values."""
    _validate_inputs(artifact, presentation)
    if artifact.coordinates.shape[1] < 2:
        raise ValueError("Reduction scatter requires at least two components")
    coordinates = artifact.coordinates
    sample_lookup = _sample_lookup(presentation)
    title = f"{artifact.key.method} of {artifact.key.data_type.capitalize()} Data"
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
        values = coordinates[indices, :2]
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
        if presentation.groups:
            separation_coordinates.extend(values)
            separation_labels.extend([group.group_id] * len(values))

    if presentation.show_sample_labels:
        _selection_labels(plot, sources)
    # One panel, the nearest sample's, rather than a column of them where samples crowd.
    plot.add_tools(HoverTool(renderers=point_renderers, tooltips=_sample_tooltip(artifact), limit=1))
    x_label, y_label = _axis_labels(artifact)
    plot.xaxis.axis_label = x_label
    plot.yaxis.axis_label = y_label
    if str(artifact.key.method).upper() != "PCA":
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


def _loading_rows(artifact: ReductionArtifact, presentation: ReductionPlotPresentation):
    """Per component, its strongest loadings of each sign, tied features in one bar.

    Features that load identically (several reactions Compass scores alike, one
    flux through a chain) share a bar named after the first, with the number of its
    twins, so the bars show distinct directions rather than one direction repeated.
    """
    assert artifact.pca is not None and artifact.pca.loadings is not None
    loadings = artifact.pca.loadings
    component_count = min(presentation.loading_component_count, loadings.shape[0])
    per_sign = max(1, presentation.loading_feature_count // 2)
    feature_lookup = _feature_lookup(presentation)
    variance = artifact.pca.explained_variance_ratio
    rows = []
    for component in range(component_count):
        values = loadings[component]
        header = f"PC{component + 1}"
        if variance is not None and len(variance) > component:
            header += f" ({variance[component] * 100:.1f}%)"
        rows.append({"header": header})
        tied: dict[float, list[int]] = {}
        for index, value in enumerate(values):
            tied.setdefault(round(float(value), 9), []).append(index)
        distinct = sorted(tied, key=lambda value: -value)
        positives = [value for value in distinct if value > 0][:per_sign]
        negatives = [value for value in reversed(distinct) if value < 0][:per_sign]
        for value in positives + list(reversed(negatives)):
            members = tied[value]
            first = artifact.feature_names[members[0]]
            presentation_of = feature_lookup[first]
            name = presentation_of.display_label
            if len(members) > 1:
                name += f" and {len(members) - 1} identical"
            twins = ", ".join(feature_lookup[artifact.feature_names[i]].display_label for i in members[1:11])
            loading = float(values[members[0]])
            # The feature's own panel, as the graphs show it, with its loading added.
            facts = section(
                row(f"Loading on PC{component + 1}", f"{loading:.3f}"),
                row("Loading alike", escape(twins)) if twins else "",
            )
            rows.append(
                {
                    "PC": f"PC{component + 1}",
                    "feature_id": first,
                    "Feature": name,
                    "loading": loading,
                    "twins": twins,
                    "hover": extend(presentation_of.hover_html, facts)
                    if presentation_of.hover_html
                    else panel(row("ID", escape(first)), row("Feature", escape(name)), facts),
                    "external_url": presentation_of.external_url,
                    "display_label": presentation_of.display_label,
                }
            )
    return component_count, rows


def plot_pca_loadings_artifact(
    artifact: ReductionArtifact, presentation: ReductionPlotPresentation
):
    """The strongest loadings of each sign, per component, as named bars.

    ``loading_component_count`` components are drawn, each opening with a header
    row; ``loading_feature_count`` bars each, half of each sign. The download is
    the complete matrix. ``loading_sort_component`` no longer applies: each
    component is ordered by its own loadings.
    """
    _validate_inputs(artifact, presentation)
    _require_pca_loadings(artifact)
    assert artifact.pca is not None
    assert artifact.pca.loadings is not None
    if artifact.pca.loadings.size == 0:
        raise ValueError("Loadings matrix is empty")
    component_count, rows = _loading_rows(artifact, presentation)

    factors, headers, used = [], [], set()
    for row in rows:
        text = row.get("header") or row["Feature"]
        # A feature can load on several components; each bar keeps its own row.
        while text in used:
            text += "\u200b"
        used.add(text)
        factors.append(text)
        if "header" in row:
            headers.append(text)
        else:
            row["y"] = text
    bars = [row for row in rows if "header" not in row]
    reach = max([abs(row["loading"]) for row in bars] + [1e-9]) * 1.08
    plot = apply_academic_style(
        figure(
            width=presentation.width,
            # Every bar keeps a readable row, however many are asked for.
            height=max(presentation.height, LOADING_ROW_HEIGHT * len(factors) + 90),
            x_range=Range1d(-reach, reach),
            y_range=FactorRange(factors=list(reversed(factors)), range_padding=0.0),
            title=(
                f"PCA loadings of {artifact.data_source_name}, the strongest of each sign "
                f"in the first {component_count} PCs"
            ),
            tools="xpan,xwheel_zoom,reset,save",
        )
    )
    plot.ygrid.grid_line_color = None
    if headers:
        plot.hbar(
            y=headers,
            left=-reach * 2,
            right=reach * 2,
            height=1.0,
            fill_color="#eeeeee",
            line_color=None,
            level="underlay",
        )
    source = ColumnDataSource(pd.DataFrame(bars)) if bars else ColumnDataSource(
        {key: [] for key in ("y", "loading", "PC", "feature_id", "Feature", "twins", "hover", "external_url", "display_label")}
    )
    renderer = plot.hbar(
        y="y",
        left=0,
        right="loading",
        height=0.7,
        source=source,
        fill_color=LOADING_BAR_COLOR,
        line_color=None,
    )
    plot.add_layout(Span(location=0, dimension="height", line_color=theme.INK_MUTED, line_width=1))
    plot.add_tools(HoverTool(renderers=[renderer], tooltips="@hover{safe}"))
    yaxis = plot.yaxis[0]
    yaxis.axis_line_color = None
    yaxis.major_tick_line_color = None
    yaxis.major_label_text_font_size = "11px"
    yaxis.major_label_text_color = theme.INK
    plot.xaxis.axis_label = "Loading"
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

    loadings_x = artifact.pca.loadings[x_index]
    loadings_y = artifact.pca.loadings[y_index]
    score_range = max(
        float(scores["x"].max() - scores["x"].min()),
        float(scores["y"].max() - scores["y"].min()),
        1.0,
    )
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
    plot.add_tools(
        HoverTool(tooltips="@hover{safe}", renderers=[arrow_hover_renderer], name="arrow_hover"),
        HoverTool(tooltips=_sample_tooltip(artifact, pc_x, pc_y), renderers=sample_renderers, name="sample_hover", limit=1),
    )
    if presentation.show_separation_metrics:
        valid = scores[~scores["group"].isin(["Unassigned", "N/A"])]
        _separation_note(plot, valid[["x", "y"]].to_numpy(), valid["group_id"].tolist())
    _legend_outside(plot)
    return _apply_presentation_style(plot, presentation)


__all__ = [
    "ReductionFeaturePresentation",
    "ReductionGroupPresentation",
    "ReductionPlotPresentation",
    "ReductionSamplePresentation",
    "build_reduction_presentation",
    "plot_pca_biplot_artifact",
    "plot_pca_loadings_artifact",
    "plot_pca_variance_artifact",
    "plot_reduction_artifact",
]
