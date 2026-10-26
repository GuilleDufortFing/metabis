from typing import List

from ..workspace_mode import AnalysisWorkspaceMode
from .interfaces import HoverSection
from .models import HoverContext, HoverData
from .panel import PANEL_STYLE


class HoverTextBuilder:
    """Builder class for creating hover text with configurable sections."""

    def __init__(self):
        """Initialize the hover text builder."""
        self.sections: List[HoverSection] = []
        self.container_style = PANEL_STYLE

    def add_section(self, section: HoverSection) -> "HoverTextBuilder":
        """Add a section to the builder.

        Args:
            section: HoverSection instance to add

        Returns:
            Self for method chaining
        """
        self.sections.append(section)
        return self

    def set_container_style(self, style: str) -> "HoverTextBuilder":
        """Set the container style for the hover text.

        Args:
            style: CSS style string for the container div

        Returns:
            Self for method chaining
        """
        self.container_style = style
        return self

    def build(self, data: HoverData, context: HoverContext) -> str:
        """Build the complete hover text.

        Args:
            data: The data dictionary for the element
            context: Additional context (data_manager, settings, etc.)

        Returns:
            str: Complete HTML hover text
        """
        element_type = context.element_type or "element"
        element_id = data.get(f"{element_type}_id", "Unknown")

        # Check if element was found
        if not data.get("is_in_model", True):  # Default to True for backward compatibility
            # Highlighted notice when element was not found in the model
            group_id = data.get("experiment_stats", {}).get("group_id")
            group_fragment = (
                f" for group <span style='font-weight:bold;'>{group_id}</span>" if group_id else ""
            )
            return (
                # "<div style='margin-top:8px; border-top:1px solid #ddd; padding-top:8px;'>"
                "<div style='background:#fff8e1; border:1px solid #ffe7a3; padding:8px 10px; "
                "border-radius:4px; font-size:11px; line-height:1.35;'>"
                "<div style='font-weight:bold; color:#8a6d00; text-transform:uppercase; letter-spacing:0.5px;'>"
                "No Metadata</div>"
                f"<div style='margin-top:4px; color:#5c4f1a;'>No metadata available{group_fragment} "
                f"for {element_type} <span style='font-family:monospace;'>{element_id}</span>."
                "</div>"
                "<div style='margin-top:6px; font-size:10px; color:#7a6a24;'>"
                "This element was not found in the Human1 metabolic model."
                "<div style='margin-top:8px; border-top:1px solid #ddd; padding-top:8px;'>"
                "</div>"
                "</div></div>"
            )


        if element_type == 'reaction' or element_type == 'gene':
            mode = context.mode

            if mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS:
                has_statistical_data = data.get("experiment_stats", {}).get("has_experiment_data", False)
            elif mode == AnalysisWorkspaceMode.GROUP_COMPARISON:
                has_statistical_data = data.get("comparison_stats", {}).get("has_comparison_data", False)
            else:
                has_statistical_data = False
        else:
            has_statistical_data = True # Assume true for other types or handle accordingly

        # Accumulate regular sections first (basic info, etc.)
        section_contents = []
        # Build all applicable sections first
        for section in self.sections:
            if section.should_include(data, context):
                content = section.build(data, context)
                if content.strip():  # Only add non-empty content
                    section_contents.append(content)

        # Append banner last if no stats
        if (element_type == 'reaction' or element_type == 'gene') and not has_statistical_data and context.show_stats:
            group_id = data.get("experiment_stats", {}).get("group_id")
            group_fragment = (
                f" for group <span style='font-weight:bold;'>{group_id}</span>" if group_id else ""
            )
            no_data_banner = (
                "<div style='margin-top:8px; border-top:1px solid #ddd; padding-top:8px;'>"
                "<div style='background:#fff8e1; border:1px solid #ffe7a3; padding:8px 10px; "
                "border-radius:4px; font-size:11px; line-height:1.35;'>"
                "<div style='font-weight:bold; color:#8a6d00; text-transform:uppercase; letter-spacing:0.5px;'>"
                "No Experimental Data</div>"
                f"<div style='margin-top:4px; color:#5c4f1a;'>No experimental data available{group_fragment} "
                f"for {element_type} <span style='font-family:monospace;'>{element_id}</span>."
                "</div>"
                "<div style='margin-top:6px; font-size:10px; color:#7a6a24;'>"
                "Data may be missing because this element wasn't measured, filtered out, or the analysis hasn't been computed yet."
                "</div>"
                "</div></div>"
            )
            section_contents.append(no_data_banner)

        if not section_contents:
            return f"<div style='{self.container_style}'>No information available</div>"

        # Combine all sections
        return f"<div style='{self.container_style}'>{''.join(section_contents)}</div>"
