"""Colour distribution registration, without Bokeh.

Slice 3a split the *readable* half of ``ColorManager`` into the frozen
:class:`~metabis.managers.color_lookup.GraphColorLookup`, on the assumption that
a graph worker only ever asks what an element is painted. Measuring the real
enrichment path showed that assumption is one step short: a worker also
*registers* distributions while it runs. Activity scoring publishes reaction,
gene, and subsystem activity distributions, and a subsystem graph asked for a
delta configuration the document thread has not already computed registers that
distribution mid-enrichment (measured: 0 registrations when the configuration is
already installed, 1 when it is not). Today those writes land on the ``deepcopy``
of the colour manager that ``_build_detached_workspace`` makes per job, and are
discarded.

So the worker needs the *writable* half too -- and this module is the evidence
that wanting it does not mean wanting Bokeh. Registration is binning plus RGB
interpolation: ``DistributionBinner`` over the values, a gradient between the
mode's neutral and up/down colours, and a bin->hex map. None of that needs
``bokeh.palettes``. What does need it, and therefore stays in ``ColorManager``,
is *generating* the two influence palettes (``Greens``/``Oranges``) and the
categorical group palettes (turbo, viridis, Category20, Set1, Set3).

``ColorManager`` delegates here rather than keeping a second copy, for the same
reason it delegates its reads to ``GraphColorLookup``: one implementation, so a
worker's registry and the live manager cannot disagree about what a value means.
See ``MetabisReview/architecture-migration/am10-inventory.md``.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd

from ..analysis.distribution_binner import DistributionBinner
from ..visual_config import theme
from .color_lookup import DistributionKey, GraphColorLookup
from .workspace_mode import AnalysisWorkspaceMode

Color = Union[str, Tuple[int, int, int]]


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert a ``#RRGGBB`` string to an RGB triple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color format: {hex_color}")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert an RGB triple to a ``#RRGGBB`` string."""
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def ensure_rgb(color: Color) -> Tuple[int, int, int]:
    """Accept either colour spelling and return an RGB triple."""
    if isinstance(color, str):
        return hex_to_rgb(color)
    if isinstance(color, (tuple, list)) and len(color) == 3:
        return (int(color[0]), int(color[1]), int(color[2]))
    raise ValueError(
        f"Invalid color format: {color}. Expected hex string or RGB tuple."
    )


def gradient_color(
    base_rgb: Tuple[int, int, int],
    target_rgb: Tuple[int, int, int],
    fraction: float,
) -> str:
    """Interpolate linearly between two RGB colours and return hex."""
    fraction = max(0.0, min(1.0, fraction))
    red = int(base_rgb[0] + fraction * (target_rgb[0] - base_rgb[0]))
    green = int(base_rgb[1] + fraction * (target_rgb[1] - base_rgb[1]))
    blue = int(base_rgb[2] + fraction * (target_rgb[2] - base_rgb[2]))
    return f"#{red:02X}{green:02X}{blue:02X}"


def gradient_palette(start_color: Color, end_color: Color, n_steps: int) -> List[str]:
    """Build an ``n_steps`` gradient from ``start_color`` to ``end_color``."""
    if n_steps <= 1:
        return [end_color if isinstance(end_color, str) else rgb_to_hex(ensure_rgb(end_color))]

    start_rgb = ensure_rgb(start_color)
    end_rgb = ensure_rgb(end_color)
    return [
        gradient_color(start_rgb, end_rgb, step / (n_steps - 1))
        for step in range(n_steps)
    ]


class ColorDistributionRegistry:
    """Registered colour distributions and the mode configurations that paint them.

    The registry is deliberately mutable: registration is how an analysis
    publishes a new scientific distribution, and a graph worker does that during
    its own run. What makes that safe is ownership, not immutability -- a worker
    holds its own registry, produced by :meth:`detached_copy`, so its writes stay
    in the job. The value it hands to enrichment is still the frozen
    ``GraphColorLookup``.
    """

    def __init__(
        self,
        configurations: MutableMapping[AnalysisWorkspaceMode, Dict[str, Any]],
        current_mode: AnalysisWorkspaceMode,
        distributions: Optional[
            MutableMapping[DistributionKey, Dict[str, Any]]
        ] = None,
        group_colors: Optional[Mapping[str, str]] = None,
        positive_influence_palette: Sequence[str] = (),
        negative_influence_palette: Sequence[str] = (),
    ) -> None:
        self.configurations = configurations
        self.current_mode = current_mode
        # Registered binners and their bin->hex maps, keyed by
        # (mode, element_type, metric_name).
        self.distributions: Dict[DistributionKey, Dict[str, Any]] = (
            dict(distributions) if distributions is not None else {}
        )
        # Already-resolved palettes. Generating these is the Bokeh half and stays
        # on ColorManager; a registry only ever carries the hex it produced.
        self.group_colors: Dict[str, str] = dict(group_colors or {})
        self.positive_influence_palette: Tuple[str, ...] = tuple(
            positive_influence_palette
        )
        self.negative_influence_palette: Tuple[str, ...] = tuple(
            negative_influence_palette
        )

    def detached_copy(self) -> "ColorDistributionRegistry":
        """Return a registry a worker can register into without touching this one.

        The configurations are copied because a worker may switch mode, and the
        distribution *mapping* is copied so a registration during the job cannot
        reach the document thread. The binners inside it are shared rather than
        deep-copied: they are never mutated after registration, and copying them
        per render is the ADR-0007 per-render tax AM8 slice 4 already refused
        once.
        """
        return ColorDistributionRegistry(
            configurations={
                mode: dict(config) for mode, config in self.configurations.items()
            },
            current_mode=self.current_mode,
            distributions=self.distributions,
            group_colors=self.group_colors,
            positive_influence_palette=self.positive_influence_palette,
            negative_influence_palette=self.negative_influence_palette,
        )

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def lookup(self) -> GraphColorLookup:
        """The frozen read view of this registry's current state.

        Built per call rather than cached. The registry is mutable by design --
        a mode switch, a group-colour assignment, or a distribution registered
        mid-run all change what a colour means -- and a cache over that is a
        staleness bug waiting for the one caller that mutates in place. The cost
        is a shallow wrap of state the registry already holds: ~1 us against the
        ~164 ms graph render that performs these reads.
        """
        config = self.configurations[self.current_mode]
        return GraphColorLookup(
            neutral_color=config["neutral_color"],
            up_color=config["up_color"],
            down_color=config["down_color"],
            group_colors=self.group_colors,
            positive_influence_palette=self.positive_influence_palette,
            negative_influence_palette=self.negative_influence_palette,
            distributions=self.distributions,
        )

    # The reads a graph render performs, delegated so that a registry and the
    # lookup it produces cannot answer differently.

    def get_element_color_from_distribution(self, *args: Any, **kwargs: Any) -> str:
        """Colour for one element from its registered distribution."""
        return self.lookup().get_element_color_from_distribution(*args, **kwargs)

    def get_element_colors_from_distribution(self, *args: Any, **kwargs: Any):
        """Colours for many elements at once, indexed by element ID."""
        return self.lookup().get_element_colors_from_distribution(*args, **kwargs)

    def get_color_limit(self, *args: Any, **kwargs: Any) -> Optional[float]:
        """Distance from the centre at which the registered palette saturates."""
        return self.lookup().get_color_limit(*args, **kwargs)

    def get_distribution_details(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Histogram, bin, colour, and value for one element's distribution."""
        return self.lookup().get_distribution_details(*args, **kwargs)

    def get_group_colors(self) -> Dict[str, str]:
        """Every stored group colour, as a mutable copy."""
        return self.lookup().get_group_colors()

    def get_group_color(self, group_id: str) -> str:
        """Colour for one group."""
        return self.lookup().get_group_color(group_id)

    def has_group_colors(self) -> bool:
        """Whether any group colour is registered."""
        return self.lookup().has_group_colors()

    def get_influence_color(self, *args: Any, **kwargs: Any) -> str:
        """Colour for a gene-influence bar."""
        return self.lookup().get_influence_color(*args, **kwargs)

    @property
    def up_color(self) -> str:
        """The current mode's up-regulated end."""
        return self.configurations[self.current_mode]["up_color"]

    @property
    def down_color(self) -> str:
        """The current mode's down-regulated end."""
        return self.configurations[self.current_mode]["down_color"]

    @property
    def neutral_color(self) -> str:
        """The current mode's neutral centre."""
        return self.configurations[self.current_mode]["neutral_color"]

    def configuration_for(self, mode: AnalysisWorkspaceMode) -> Dict[str, Any]:
        """Configuration for ``mode``, falling back to the current mode's."""
        return self.configurations.get(mode) or self.configurations[self.current_mode]

    def register_distribution(
        self,
        mode: AnalysisWorkspaceMode,
        element_type: str,
        data: Union[pd.Series, pd.DataFrame],
        metric_name: str = "default",
        value_column: Optional[str] = None,
        center_method: Literal["mean", "median", "zero"] = "zero",
        spread_method: Literal[
            "std", "mad", "saturating", "asymmetric_saturating"
        ] = "std",
        max_z: float = 4.0,
        n_bins: int = 30,
        saturation_quantile: float = 0.98,
    ) -> None:
        """Bin a distribution and store its bin->colour map.

        Args:
            mode: The analysis mode the distribution belongs to.
            element_type: The type of element (e.g. "reaction", "subsystem").
            data: The values to bin.
            metric_name: The metric name (e.g. "log2_fc", "default").
            value_column: Column name if ``data`` is a DataFrame.
            center_method: How to locate the centre ("mean", "median", "zero").
            spread_method: How to scale ("std", "mad", "saturating", or
                "asymmetric_saturating").
            max_z: Max Z-score for binning.
            n_bins: Number of bins per direction.
            saturation_quantile: Quantile of |value - center| that saturates the
                palette under the "saturating" spread method.
        """
        binner = DistributionBinner(
            data=data,
            value_column=value_column,
            center_method=center_method,
            spread_method=spread_method,
            max_z=max_z,
            n_bins=n_bins,
            saturation_quantile=saturation_quantile,
        )

        config = self.configuration_for(mode)
        up_color = config["up_color"]
        down_color = config["down_color"]
        neutral_color = config["neutral_color"]

        # p-value metrics are not diverging: significance runs one way, so the
        # ramp is re-pointed rather than mirrored.
        lowered = metric_name.lower()
        is_pval = "pval" in lowered or "p-value" in lowered
        is_neg_log = "neg_log" in lowered or "-log" in lowered

        if is_pval:
            significant_color = config.get(
                "significant_pval_color", theme.SIGNIFICANT_PVAL
            )
            if is_neg_log:
                # Higher -log(p) is more significant: neutral stays neutral.
                up_color = significant_color
            else:
                # Lower raw p is more significant, and the gradient generator
                # only runs neutral -> up, so the two ends are swapped.
                up_color, neutral_color = neutral_color, significant_color

        # n_bins + 1 colours interpolate neutral -> full intensity; bin 0 is the
        # neutral centre, so the generated neutral end is dropped.
        pos_palette = gradient_palette(neutral_color, up_color, n_bins + 1)[1:]
        neg_palette = gradient_palette(neutral_color, down_color, n_bins + 1)[1:]

        bin_colors = {0: neutral_color}
        for index in range(n_bins):
            bin_index = index + 1
            bin_colors[bin_index] = pos_palette[index]
            bin_colors[-bin_index] = neg_palette[index]

        self.distributions[(mode, element_type, metric_name)] = {
            "binner": binner,
            "bin_colors": bin_colors,
        }

    def clear(self) -> None:
        """Forget every registered distribution."""
        self.distributions.clear()

    def current_configuration(self) -> Mapping[str, Any]:
        """The configuration the current mode paints with."""
        return self.configurations[self.current_mode]
