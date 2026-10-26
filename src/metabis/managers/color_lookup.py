"""The plain, Bokeh-free half of colour resolution.

``ColorManager`` does two separable things. It *builds* palettes -- which needs
``bokeh.palettes`` and is the sole reason an ``AnalysisWorkspace`` cannot be
imported without Bokeh (AM10 inventory, cause 3) -- and it *reads* already-built
palettes back, which needs nothing but the values it already holds. This module
owns the reading half.

``GraphColorLookup`` is a frozen value carrying exactly the state those reads
touch: the registered distributions, the group colours, the mode configurations,
and the two influence palettes as plain hex tuples. Every method on it is a pure
function of that state. It imports no Bokeh, because a palette that has already
been resolved to hex strings no longer needs the library that generated it.

This is what makes M-12 (A') work. Graph enrichment runs on a worker thread and
needs colour lookups for hover text, so it cannot simply be handed pre-resolved
colours for everything -- but it does not need a *manager*, only the manager's
data. The document thread captures this value in ``ColorManager.color_lookup()``
at ~0 ms and the worker reads it, so no colour manager, and therefore no Bokeh,
crosses the thread boundary.

``ColorManager`` delegates its own lookup methods here rather than keeping a
second copy: one implementation, so a captured lookup and a live manager cannot
disagree about what an element is painted. See
``MetabisReview/architecture-migration/am10-inventory.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .workspace_mode import AnalysisWorkspaceMode

# A distribution key is (mode, element_type, metric_name); the value is the
# binner and its bin->hex mapping, exactly as ColorManager registers them.
DistributionKey = Tuple[AnalysisWorkspaceMode, str, str]


@dataclass(frozen=True)
class GraphColorLookup:
    """Frozen colour state for one graph run, readable without Bokeh.

    Construct it with :meth:`ColorManager.color_lookup`, never by hand in server
    code: the point of the type is that the library owns what a colour means.

    The mappings are wrapped in ``MappingProxyType`` at construction, so a worker
    holding this value cannot mutate the distributions the document thread
    captured -- the AM8 detachment property, applied to colour.
    """

    neutral_color: str
    up_color: str
    down_color: str
    group_colors: Mapping[str, str]
    positive_influence_palette: Tuple[str, ...]
    negative_influence_palette: Tuple[str, ...]
    distributions: Mapping[DistributionKey, Dict[str, Any]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_colors", MappingProxyType(dict(self.group_colors)))
        object.__setattr__(
            self, "distributions", MappingProxyType(dict(self.distributions))
        )
        object.__setattr__(
            self, "positive_influence_palette", tuple(self.positive_influence_palette)
        )
        object.__setattr__(
            self, "negative_influence_palette", tuple(self.negative_influence_palette)
        )

    # ------------------------------------------------------------------
    # Distribution reads
    # ------------------------------------------------------------------

    def get_element_color_from_distribution(
        self,
        mode: AnalysisWorkspaceMode,
        element_type: str,
        element_id: str,
        metric_name: str = "default",
    ) -> str:
        """Colour for one element from its registered distribution."""
        dist_info = self.distributions.get((mode, element_type, metric_name))
        if not dist_info:
            return self.neutral_color

        binner = dist_info["binner"]
        bin_colors = dist_info["bin_colors"]
        return bin_colors.get(binner.get_bin(element_id), bin_colors[0])

    def get_color_for_value(
        self,
        mode: AnalysisWorkspaceMode,
        element_type: str,
        metric_name: str,
        value: float,
    ) -> str:
        """Colour a value as its registered distribution colours an element of it."""
        dist_info = self.distributions.get((mode, element_type, metric_name))
        if not dist_info:
            return self.neutral_color
        bin_colors = dist_info["bin_colors"]
        return bin_colors.get(dist_info["binner"].bin_for_value(value), bin_colors[0])

    def has_value(
        self,
        mode: AnalysisWorkspaceMode,
        element_type: str,
        element_id: str,
        metric_name: str = "default",
    ) -> bool:
        """Whether the distribution holds a finite value for this element.

        An element it does not hold has no data: it would otherwise fall to the
        neutral centre, indistinguishable from a measured zero.
        """
        dist_info = self.distributions.get((mode, element_type, metric_name))
        if not dist_info:
            return False
        value = dist_info["binner"].get_value(element_id)
        try:
            return value is not None and bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False

    def get_element_colors_from_distribution(
        self,
        mode: AnalysisWorkspaceMode,
        element_type: str,
        metric_name: str = "default",
        element_ids: Optional[List[str]] = None,
    ) -> pd.Series:
        """Colours for many elements at once, indexed by element ID."""
        dist_info = self.distributions.get((mode, element_type, metric_name))
        if not dist_info:
            if element_ids is not None:
                return pd.Series(self.neutral_color, index=element_ids)
            return pd.Series()

        binner = dist_info["binner"]
        bin_colors = dist_info["bin_colors"]

        if element_ids is None:
            bins = binner.get_all_bins()
        else:
            # Reindex so every requested ID appears; an unregistered element
            # falls to bin 0, the neutral centre, rather than going missing.
            bins = binner.get_all_bins().reindex(element_ids, fill_value=0).astype(int)

        return bins.map(bin_colors).fillna(bin_colors[0])

    def get_color_limit(
        self,
        mode: AnalysisWorkspaceMode,
        element_type: str,
        metric_name: str = "default",
    ) -> Optional[float]:
        """Distance from the centre at which the registered palette saturates."""
        dist_info = self.distributions.get((mode, element_type, metric_name))
        if not dist_info:
            return None
        return dist_info["binner"].color_limit

    def get_distribution_details(
        self,
        mode: AnalysisWorkspaceMode,
        element_type: str,
        element_id: str,
        metric_name: str = "default",
    ) -> Dict[str, Any]:
        """Histogram, bin, colour, and value for one element's distribution.

        This is what the hover colour-reference section renders, so it runs on
        the worker thread for every element that carries a hover.
        """
        dist_info = self.distributions.get((mode, element_type, metric_name))
        if not dist_info:
            return {
                "histogram": [],
                "element_bin": 0,
                "element_color": self.neutral_color,
                "element_value": None,
            }

        binner = dist_info["binner"]
        bin_colors = dist_info["bin_colors"]

        histogram = [
            {
                **bin_data,
                "color": bin_colors.get(
                    bin_data.get("color_bin", bin_data["bin"]), bin_colors[0]
                ),
            }
            for bin_data in binner.get_histogram_data()
        ]

        element_bin = binner.get_bin(element_id)
        return {
            "histogram": histogram,
            "element_bin": element_bin,
            "element_color": bin_colors.get(element_bin, bin_colors[0]),
            "element_value": binner.get_value(element_id),
        }

    # ------------------------------------------------------------------
    # Group colours
    # ------------------------------------------------------------------

    def get_group_colors(self) -> Dict[str, str]:
        """Every stored group colour, as a mutable copy."""
        return dict(self.group_colors)

    def get_group_color(self, group_id: str) -> str:
        """Colour for one group.

        Raises:
            KeyError: if the group has no registered colour.
        """
        if group_id not in self.group_colors:
            raise KeyError(f"Group '{group_id}' not found in group colors")
        return self.group_colors[group_id]

    def has_group_colors(self) -> bool:
        """Whether any group colour is registered."""
        return bool(self.group_colors)

    # ------------------------------------------------------------------
    # Influence colours
    # ------------------------------------------------------------------

    def get_influence_color(
        self,
        correlation: float,
        influence: float,
        max_influence: float = 100.0,
        use_gray_scale: bool = False,
        gray_palette: Optional[List[str]] = None,
    ) -> str:
        """Colour for a gene-influence bar.

        A pure function of two floats and a palette, which is why the influence
        palettes can travel as hex tuples instead of a ``ColorManager``: the sign
        of ``correlation`` picks the palette and the magnitude of ``influence``
        picks the index.
        """
        if pd.isna(correlation):
            correlation = 0.0
        if pd.isna(influence):
            influence = 0.0

        clamped_influence = max(0.0, min(influence, max_influence))
        normalized_influence = (
            clamped_influence / max_influence if max_influence > 0 else 0
        )

        if use_gray_scale:
            if not gray_palette:
                return "#BEBEBE"
            palette: Tuple[str, ...] | List[str] = gray_palette
        elif correlation >= 0:
            palette = self.positive_influence_palette
        else:
            palette = self.negative_influence_palette

        if not palette:
            return "#000000"

        palette_size = len(palette)
        color_index = min(int(normalized_influence * palette_size), palette_size - 1)
        return palette[color_index]
