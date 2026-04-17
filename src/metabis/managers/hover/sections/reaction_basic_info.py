from typing import Any, Dict
import numpy as np
from ..interfaces import HoverSection
from ..models import HoverContext, HoverData
from ....viz.metabolic_visualization.generation.data_structures.constants import (
    VisualizationConstants,
)
from ....visual_config import theme

class ReactionBasicInfoSection(HoverSection):
    """Section for basic reaction information (name, ID, equation, etc.)."""

    def should_include(self, data: HoverData, context: HoverContext) -> bool:
        """Basic info is always included."""
        return True

    def build(self, data: HoverData, context: HoverContext) -> str:
        """Build basic information section."""
        if not data.get("is_in_model", False):
            element_id = data.get("reaction_id", "Unknown")
            return f"<div>Reaction {element_id} not found</div>"

        parts = []
        name = data.get("name", "N/A")
        reaction_id = data.get("reaction_id", "N/A")
        subsystem = data.get("subsystem", "N/A")
        category = data.get("subsystem_category") or (
            data.get("metadata", {}) or {}
        ).get("subsystem_category")
        
        # Check reversibility based on data if available, otherwise equation
        equation = data.get("equation", "")
        is_reversible = data.get("is_reversible", False)
        # If reversible but equation doesn't show it, fix the equation for display
        if is_reversible and "<=>" not in equation:
            if "=>" in equation:
                equation = equation.replace("=>", "<=>")
            elif "<=" in equation:
                equation = equation.replace("<=", "<=>")
            elif "->" in equation:
                equation = equation.replace("->", "<=>")
            elif "<-" in equation:
                equation = equation.replace("<-", "<=>")
        
        # If NOT reversible (per data) but equation shows it is, fix it
        elif not is_reversible and "<=>" in equation:
            equation = equation.replace("<=>", "=>")
        
        suffix = ""
        direction = "forward"
        if is_reversible:
            if context.element_id and context.element_id.endswith("_back"):
                suffix = " (reverse reaction)"
                direction = "backward"
            else:
                suffix = " (forward)"

        parts.append(f"<div><strong>ID:</strong> {reaction_id}{suffix}</div>")

        # 2. Reaction name (conditional - only if valid)
        if name and name != "N/A" and not (isinstance(name, float) and np.isnan(name)):
            parts.append(f"<div><strong>Reaction:</strong> {name}</div>")

        # 3. Subsystem with category
        if subsystem != "N/A" and category:
            # Use CategoryManager for dynamic color lookup based on category file order
            if (
                context.data_provider is not None
                and hasattr(context.data_provider, 'metabolic_model')
                and context.data_provider.metabolic_model is not None
            ):
                color = context.data_provider.metabolic_model.get_category_color(category)
            else:
                # Fallback if model is not available (shouldn't happen in practice)
                color = VisualizationConstants.DEFAULT_CATEGORY_LINE_COLOR
            category_span = (
                f" <span style='color:{color};'>({category})</span>"
            )
        else:
            category_span = ""

        parts.append(
            f"<div><strong>Subsystem:</strong> {subsystem}{category_span}</div>"
        )

        # 4. Equation with direction-aware arrow coloring
        if equation:
            formatted_equation = self._format_equation_with_direction(
                equation, direction
            )
            parts.append(f"<div><strong>Equation:</strong> {formatted_equation}</div>")

        return "".join(parts)

    def _format_equation_with_direction(self, equation: str, direction: str) -> str:
        """Format equation with direction-aware arrow coloring.
        
        Args:
            equation: The reaction equation string (e.g., "A + B => C" or "A + B <=> C")
            direction: Either "forward" or "backward"
            
        Returns:
            HTML-formatted equation with colored arrows
        """
        # Colors
        highlight_color = theme.ACTIVITY_UP  # Highlighted direction
        gray_color = "#888"  # Gray for non-highlighted direction
        
        # Check if equation contains bidirectional arrow
        if "<=>" in equation:
            # Split by the bidirectional arrow
            parts = equation.split("<=>")
            if len(parts) == 2:
                left, right = parts
                if direction == "forward":
                    # Highlight => in magenta, <= in gray
                    arrow_html = (
                        f"<span style='color:{gray_color};'>&lt;</span>"
                        f"<span style='color:{highlight_color}; font-weight:bold;'>=&gt;</span>"
                    )
                else:  # backward
                    # Highlight <= in magenta, => in gray
                    arrow_html = (
                        f"<span style='color:{highlight_color}; font-weight:bold;'>&lt;=</span>"
                        f"<span style='color:{gray_color};'>&gt;</span>"
                    )
                return f"{left}{arrow_html}{right}"
        
        # Check if equation contains unidirectional arrow
        elif "=>" in equation:
            parts = equation.split("=>")
            if len(parts) == 2:
                left, right = parts
                if direction == "forward":
                    # Forward: highlight => in magenta
                    arrow_html = f"<span style='color:{highlight_color}; font-weight:bold;'>=&gt;</span>"
                else:  # backward
                    # Backward: convert to <=>, highlight <= in magenta
                    arrow_html = (
                        f"<span style='color:{highlight_color}; font-weight:bold;'>&lt;=</span>"
                        f"<span style='color:{gray_color};'>&gt;</span>"
                    )
                return f"{left}{arrow_html}{right}"
        
        # If no recognized arrow pattern, return as-is
        return equation
