"""Artifact-first Bokeh factories for dimensionality-reduction results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from bokeh.models import (
    Arrow,
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    FactorRange,
    HoverTool,
    LinearColorMapper,
    Range1d,
    VeeHead,
)
from bokeh.palettes import Category10
from bokeh.plotting import figure
from bokeh.transform import transform

from ..analysis.dimensionality_reduction import ReductionArtifact
from ..visual_config import theme
from .click_behavior import get_external_reference_data
from .plot_reduced_dimension import (
    add_cluster_metrics_to_plot,
    calculate_cluster_metrics,
    move_legend_outside_if_crowded,
)
from .plot_utils import apply_academic_style

DEFAULT_CATEGORICAL_PALETTE = Category10[10]
DEFAULT_SINGLE_COLOR = DEFAULT_CATEGORICAL_PALETTE[0]


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
    max_variance_components: int = 50
    loading_component_count: int = 8
    loading_feature_count: int = 50
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
    plot.background_fill_color = presentation.background_color
    plot.border_fill_color = presentation.border_color
    if plot.title:
        plot.title.text_color = presentation.text_color
    for axis in plot.xaxis + plot.yaxis:
        axis.axis_label_text_color = presentation.text_color
        axis.major_label_text_color = presentation.text_color
        axis.major_tick_line_color = presentation.text_color
        axis.minor_tick_line_color = presentation.text_color
        axis.axis_line_color = presentation.text_color
    return plot


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
                display_label=str(reference["display_label"]),
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
    max_variance_components: int = 50,
    loading_component_count: int = 8,
    loading_feature_count: int = 50,
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
        )
    )

    separation_coordinates: list[np.ndarray] = []
    separation_labels: list[str] = []
    if presentation.groups:
        for index, group in enumerate(presentation.groups):
            member_set = set(group.sample_ids)
            indices = [
                row
                for row, sample_id in enumerate(artifact.sample_ids)
                if sample_id in member_set
            ]
            if not indices:
                continue
            values = coordinates[indices, :2]
            canonical_ids = [artifact.sample_ids[row] for row in indices]
            source = ColumnDataSource(
                {
                    "x": values[:, 0],
                    "y": values[:, 1],
                    "sample_id": canonical_ids,
                    "display_label": [
                        sample_lookup[sample_id].display_label
                        for sample_id in canonical_ids
                    ],
                    "group_id": [group.group_id] * len(indices),
                    "group": [group.display_name] * len(indices),
                }
            )
            color = group.color or DEFAULT_CATEGORICAL_PALETTE[
                index % len(DEFAULT_CATEGORICAL_PALETTE)
            ]
            plot.scatter(
                "x",
                "y",
                source=source,
                color=color,
                legend_label=group.display_name,
            )
            if presentation.show_sample_labels:
                plot.text(
                    "x",
                    "y",
                    text="display_label",
                    source=source,
                    text_font_size="7pt",
                    x_offset=4,
                )
            if presentation.highlight_group_id == group.group_id:
                centroid = values.mean(axis=0)
                plot.text(
                    x=[centroid[0]],
                    y=[centroid[1]],
                    text=[group.display_name],
                    text_font_size="10pt",
                    text_baseline="middle",
                    text_align="center",
                    text_color=color,
                    text_font_style="bold",
                )
            separation_coordinates.extend(values)
            separation_labels.extend([group.group_id] * len(values))
    else:
        source = ColumnDataSource(
            {
                "x": coordinates[:, 0],
                "y": coordinates[:, 1],
                "sample_id": list(artifact.sample_ids),
                "display_label": [sample.display_label for sample in presentation.samples],
                "group_id": [""] * len(artifact.sample_ids),
                "group": ["All samples"] * len(artifact.sample_ids),
            }
        )
        plot.scatter(
            "x",
            "y",
            source=source,
            color=DEFAULT_SINGLE_COLOR,
            legend_label="All samples",
        )
        if presentation.show_sample_labels:
            plot.text(
                "x",
                "y",
                text="display_label",
                source=source,
                text_font_size="7pt",
                x_offset=4,
            )

    plot.add_tools(
        HoverTool(
            tooltips=[
                ("Sample ID", "@sample_id"),
                ("Display label", "@display_label"),
                ("Group", "@group"),
                ("X", "@x{0.00}"),
                ("Y", "@y{0.00}"),
            ]
        )
    )
    plot.xaxis.axis_label = f"{artifact.key.method} Component 1"
    plot.yaxis.axis_label = f"{artifact.key.method} Component 2"
    plot.legend.location = "right"
    plot.legend.orientation = "vertical"
    move_legend_outside_if_crowded(
        plot, len(presentation.groups) if presentation.groups else 1
    )

    if (
        presentation.show_separation_metrics
        and len(set(separation_labels)) >= 2
        and len(separation_coordinates) >= 2 * len(set(separation_labels))
    ):
        metrics = calculate_cluster_metrics(
            np.asarray(separation_coordinates), separation_labels
        )
        add_cluster_metrics_to_plot(plot, metrics, title)
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
    """Build explained/cumulative variance from the PCA artifact."""
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
            "explained_variance_str": [f"{value * 100:.1f}%" for value in variance],
            "cumulative_variance_str": [
                f"{value * 100:.1f}%" for value in cumulative
            ],
        }
    )
    title = f"PCA Explained Variance: {artifact.data_source_name}"
    if count < retained:
        title += (
            f" (first {count} of {retained} PCs from "
            f"{artifact.n_variable_features_used} features)"
        )
    else:
        title += (
            f" ({retained} PCs from {artifact.n_variable_features_used} features)"
        )
    plot = apply_academic_style(
        figure(
            width=presentation.width,
            height=presentation.height,
            x_range=Range1d(0.5, count + 0.5),
            title=title,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
    )
    plot.xgrid.grid_line_color = None
    plot.vbar(
        x="components",
        top="explained_variance",
        width=0.7,
        source=source,
        legend_label="Explained Variance per PC",
        color=DEFAULT_CATEGORICAL_PALETTE[0],
        alpha=0.7,
        line_color="#333333",
        line_width=0.5,
    )
    plot.line(
        x="components",
        y="cumulative_variance",
        source=source,
        legend_label="Cumulative Explained Variance",
        line_width=2.5,
        color=DEFAULT_CATEGORICAL_PALETTE[1],
    )
    plot.scatter(
        x="components",
        y="cumulative_variance",
        source=source,
        color=DEFAULT_CATEGORICAL_PALETTE[1],
        size=6,
        marker="circle",
    )
    plot.add_tools(
        HoverTool(
            tooltips=[
                ("PC", "@components"),
                ("Explained Var.", "@explained_variance_str"),
                ("Cumulative Var.", "@cumulative_variance_str"),
            ]
        )
    )
    plot.xaxis.axis_label = "Principal Component"
    plot.yaxis.axis_label = "Proportion of Variance Explained"
    plot.y_range.start = 0
    plot.y_range.end = max(1.0, float(np.max(cumulative)) * 1.05)
    plot.legend.location = "center_right"
    return _apply_presentation_style(plot, presentation)


def _selected_loadings(
    artifact: ReductionArtifact, presentation: ReductionPlotPresentation
):
    assert artifact.pca is not None
    assert artifact.pca.loadings is not None
    loadings = artifact.pca.loadings
    component_count = min(presentation.loading_component_count, loadings.shape[0])
    if presentation.loading_sort_component is not None:
        if presentation.loading_sort_component > loadings.shape[0]:
            raise ValueError(
                f"PC for sorting must be between 1 and {loadings.shape[0]}"
            )
        sort_index = presentation.loading_sort_component - 1
        order = np.argsort(np.abs(loadings[sort_index]))[::-1]
    else:
        order = np.argsort(np.sum(np.abs(loadings[:component_count]), axis=0))[::-1]
    feature_count = min(presentation.loading_feature_count, loadings.shape[1])
    indices = order[:feature_count]
    return component_count, indices, loadings[:component_count, indices]


def plot_pca_loadings_artifact(
    artifact: ReductionArtifact, presentation: ReductionPlotPresentation
):
    """Build the PCA loading heatmap without a manager/cache dependency."""
    _validate_inputs(artifact, presentation)
    _require_pca_loadings(artifact)
    assert artifact.pca is not None
    assert artifact.pca.loadings is not None
    if artifact.pca.loadings.size == 0:
        raise ValueError("Loadings matrix is empty")
    component_count, feature_indices, selected = _selected_loadings(
        artifact, presentation
    )
    feature_lookup = _feature_lookup(presentation)
    feature_ids = [artifact.feature_names[index] for index in feature_indices]
    feature_labels = [feature_lookup[feature_id].display_label for feature_id in feature_ids]
    pc_labels = [f"PC{index + 1}" for index in range(component_count)]
    frame = (
        pd.DataFrame(selected, index=pc_labels, columns=feature_ids)
        .stack()
        .rename("loading")
        .reset_index()
    )
    frame.columns = ["PC", "feature_id", "loading"]
    frame["Feature"] = [feature_lookup[value].display_label for value in frame["feature_id"]]
    frame["display_label"] = frame["Feature"]
    frame["hover"] = [
        feature_lookup[value].hover_html or f"Feature: {value}"
        for value in frame["feature_id"]
    ]
    frame["external_url"] = [
        feature_lookup[value].external_url for value in frame["feature_id"]
    ]
    max_abs = float(np.max(np.abs(selected))) if selected.size else 1.0
    if max_abs == 0:
        max_abs = 1.0
    mapper = LinearColorMapper(palette="RdBu11", low=-max_abs, high=max_abs)
    title = (
        f"PCA Loadings: {artifact.data_source_name} "
        f"(Top {component_count} PCs, {len(feature_ids)} Features)"
    )
    if presentation.loading_sort_component is None:
        title += ", sorted by sum of abs. loadings"
    else:
        title += f", sorted by PC{presentation.loading_sort_component}"
    plot = apply_academic_style(
        figure(
            width=presentation.width,
            height=presentation.height,
            x_range=FactorRange(factors=feature_labels),
            y_range=FactorRange(factors=list(reversed(pc_labels))),
            title=title,
            tools="save,pan,box_zoom,wheel_zoom,reset",
            x_axis_location="above",
        ),
        suppress_grid=True,
    )
    plot.axis.axis_line_color = None
    plot.axis.major_tick_line_color = None
    plot.rect(
        x="Feature",
        y="PC",
        width=1,
        height=1,
        source=frame,
        fill_color=transform("loading", mapper),
        line_color=None,
    )
    plot.add_layout(
        ColorBar(
            color_mapper=mapper,
            ticker=BasicTicker(desired_num_ticks=7),
            label_standoff=8,
            border_line_color=None,
            location=(0, 0),
            major_label_text_font_size="8pt",
            title="Loading",
            title_text_font_style="normal",
            title_text_font_size="9pt",
            title_standoff=8,
        ),
        "right",
    )
    plot.add_tools(
        HoverTool(
            tooltips="""
                <div>
                    <div><strong>Feature ID:</strong> @feature_id</div>
                    <div><strong>Feature:</strong> @display_label</div>
                    <div><strong>PC:</strong> @PC</div>
                    <div><strong>Loading:</strong> @loading{0.000}</div>
                    @hover{safe}
                </div>
            """
        )
    )
    plot.xaxis.major_label_orientation = np.pi / 4
    plot.xaxis.axis_label = "Features"
    plot.yaxis.axis_label = "Principal Components"
    return _apply_presentation_style(plot, presentation)


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
    title = f"PCA Biplot (PC{pc_x} vs PC{pc_y}) for {artifact.data_source_name}"
    plot = apply_academic_style(
        figure(
            width=presentation.width,
            height=presentation.height,
            title=title,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
    )
    variance = artifact.pca.explained_variance_ratio
    if variance is None:
        plot.xaxis.axis_label = f"PC{pc_x}"
        plot.yaxis.axis_label = f"PC{pc_y}"
    else:
        plot.xaxis.axis_label = f"PC{pc_x} ({variance[x_index] * 100:.1f}%)"
        plot.yaxis.axis_label = f"PC{pc_y} ({variance[y_index] * 100:.1f}%)"
    sample_renderers = []
    for group_name in dict.fromkeys(scores["group"].tolist()):
        group_rows = scores[scores["group"] == group_name]
        # The labels draw off the same source as the points they name, so the
        # biplot carries one copy of each group's coordinates.
        source = ColumnDataSource(group_rows)
        renderer = plot.scatter(
            x="x",
            y="y",
            source=source,
            legend_label=str(group_name),
            color=group_rows["color"].iloc[0],
            name="sample_points",
        )
        sample_renderers.append(renderer)
        if presentation.show_sample_labels:
            plot.text(
                "x",
                "y",
                text="display_label",
                source=source,
                text_font_size="7pt",
                x_offset=4,
            )
    if presentation.highlight_group_id:
        highlighted = scores[scores["group_id"] == presentation.highlight_group_id]
        if not highlighted.empty:
            group = next(
                item
                for item in presentation.groups
                if item.group_id == presentation.highlight_group_id
            )
            plot.text(
                x=[highlighted["x"].mean()],
                y=[highlighted["y"].mean()],
                text=[group.display_name],
                text_font_size="10pt",
                text_baseline="middle",
                text_align="center",
                text_color=group.color,
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
    arrows = pd.DataFrame(
        {
            "x_end": loadings_x * scale,
            "y_end": loadings_y * scale,
            "feature_id": artifact.feature_names,
            "magnitude": magnitudes,
            "x_start": 0,
            "y_start": 0,
        }
    )
    if (
        presentation.biplot_top_arrows is not None
        and presentation.biplot_top_arrows < len(arrows)
    ):
        arrows = arrows.nlargest(presentation.biplot_top_arrows, "magnitude")
    arrows["display_label"] = [
        feature_lookup[value].display_label for value in arrows["feature_id"]
    ]
    arrows["hover"] = [
        feature_lookup[value].hover_html or f"Feature: {value}"
        for value in arrows["feature_id"]
    ]
    arrows["external_url"] = [
        feature_lookup[value].external_url for value in arrows["feature_id"]
    ]
    arrow_source = ColumnDataSource(arrows)
    arrow_color = theme.ACTIVITY_UP
    plot.add_layout(
        Arrow(
            end=VeeHead(
                size=7,
                fill_color=arrow_color,
                line_color=arrow_color,
                line_width=1,
            ),
            x_start="x_start",
            y_start="y_start",
            x_end="x_end",
            y_end="y_end",
            source=arrow_source,
            line_color=arrow_color,
            line_width=1.5,
            line_alpha=0.7,
        )
    )
    arrow_hover_renderer = plot.scatter(
        x="x_end",
        y="y_end",
        source=arrow_source,
        size=10,
        alpha=0,
        name="arrow_endpoints",
    )
    plot.text(
        x="x_end",
        y="y_end",
        text="display_label",
        source=arrow_source,
        text_font_size="7pt",
        text_color="#333333",
        text_alpha=0.85,
        text_baseline="middle",
        text_align="left",
        x_offset=4,
        y_offset=4,
    )
    plot.add_tools(
        HoverTool(
            tooltips="""
                <div>
                    <div><strong>Feature ID:</strong> @feature_id</div>
                    <div><strong>Feature:</strong> @display_label</div>
                    <div><strong>Loading Magnitude:</strong> @magnitude{0.000}</div>
                    @hover{safe}
                </div>
            """,
            renderers=[arrow_hover_renderer],
            name="arrow_hover",
        ),
        HoverTool(
            tooltips=[
                ("Sample ID", "@sample_id"),
                ("Display label", "@display_label"),
                ("Group", "@group"),
                (f"PC{pc_x}", "@x{0.00}"),
                (f"PC{pc_y}", "@y{0.00}"),
            ],
            renderers=sample_renderers,
            name="sample_hover",
        ),
    )
    if presentation.show_separation_metrics:
        valid = scores[~scores["group"].isin(["Unassigned", "N/A"])]
        if len(set(valid["group_id"])) >= 2:
            add_cluster_metrics_to_plot(
                plot,
                calculate_cluster_metrics(
                    valid[["x", "y"]].to_numpy(), valid["group_id"].tolist()
                ),
                title,
            )
    plot.legend.location = "top_right"
    move_legend_outside_if_crowded(
        plot, len(plot.legend[0].items) if plot.legend else 0
    )
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
