from typing import Any, Dict
from ..interfaces import HoverSection
from ..models import HoverContext, HoverData

class MetaboliteBasicInfoSection(HoverSection):
    """Section for basic metabolite information."""

    def should_include(self, data: HoverData, context: HoverContext) -> bool:
        """Basic info is always included."""
        return True

    def build(self, data: HoverData, context: HoverContext) -> str:
        """Build basic information section."""
        if not data.get("is_in_model", False):
            element_id = data.get("metabolite_id", "Unknown")
            return f"<div>Metabolite {element_id} not found</div>"

        parts = []
        name = data.get("name", data.get("label", "N/A"))
        parts.append(f"<div><strong>Metabolite:</strong> {name}</div>")
        # Show IDs and chemical info if available
        meta_id = data.get("metabolite_id") or data.get("primary_id")
        if meta_id:
            parts.append(f"<div><strong>ID:</strong> {meta_id}</div>")

        # Display subsystems participation with neighbor highlighting if available
        self._add_metabolite_subsystems_with_neighbors(parts, data, context)

        return "".join(parts)

    def _add_metabolite_subsystems_with_neighbors(self, parts, data, context):
        """Add metabolite subsystems with same_category highlighting based on categories.
        
        Displays subsystems where the metabolite is produced or consumed, with special
        highlighting for the current subsystem (bold) and same category subsystems
        (same color but not bold).
        
        Args:
            parts: List to append HTML parts to
            data: Metabolite data dictionary
            context: Context dictionary containing current_subsystem and other info
        """
        # Get current subsystem and category from context
        current_subsystem = context.current_subsystem
        current_category = context.current_category
        
        # Get metabolic model from data provider if available
        data_provider = context.data_provider
        metabolic_model = None
        if data_provider:
            metabolic_model = getattr(data_provider, "metabolic_model", None)

        # Helper function to determine if a subsystem is in the same category
        def is_same_category_subsystem(subsystem_name):
            if not current_category or not metabolic_model:
                return False
            try:
                # Get category using enhanced method that handles both IDs and names
                subsystem_category = metabolic_model.get_subsystem_category(subsystem_name)
                
                return (
                    subsystem_category == current_category and 
                    subsystem_category != "Uncategorized" and
                    subsystem_name != current_subsystem
                )
            except Exception:
                return False

        # Helper function to check if a subsystem name matches the current subsystem
        def is_current_subsystem(subsystem_name):
            if not current_subsystem:
                return False
            
            # Direct match (both are IDs or both are names)
            if subsystem_name == current_subsystem:
                return True
            
            # Check if current_subsystem is an ID and subsystem_name is a display name
            if metabolic_model:
                try:
                    # Find subsystem by ID and compare with display name
                    current_subsystem_entity = metabolic_model.subsystems.get(current_subsystem)
                    if current_subsystem_entity and hasattr(current_subsystem_entity, 'name'):
                        if current_subsystem_entity.name == subsystem_name:
                            return True
                    
                    # Or find subsystem by name and compare with ID
                    for subsystem in metabolic_model.subsystems.get_all():
                        if hasattr(subsystem, 'name') and subsystem.name == subsystem_name:
                            if subsystem.id == current_subsystem:
                                return True
                            break
                except Exception:
                    pass
            
            return False

        # Helper function to format subsystem with highlighting
        def format_subsystem_item(subsystem_name, max_length=50):
            # Truncate long subsystem names
            display_name = subsystem_name
            if len(display_name) > max_length:
                display_name = display_name[:max_length-3] + "..."
            
            is_same_category = is_same_category_subsystem(subsystem_name)
            is_current = is_current_subsystem(subsystem_name)
            
            if is_current:
                # Highlight current subsystem in blue and bold
                return f"<div style='margin-left: 20px; font-size: 10px; color: #0066cc; font-weight: bold;'>• {display_name}</div>"
            elif is_same_category:
                # Same category subsystems in blue but not bold
                return f"<div style='margin-left: 20px; font-size: 10px; color: #0066cc;'>• {display_name}</div>"
            else:
                # Regular subsystems in default color
                return f"<div style='margin-left: 20px; font-size: 10px;'>• {display_name}</div>"

        # Helper function to sort subsystems: current, same_category, then rest
        def sort_subsystems_by_priority(subsystems, max_count=8):
            if not subsystems:
                return []
            
            current_subsystems = []
            same_category_subsystems = []
            other_subsystems = []
            
            for subsystem in subsystems:
                if is_current_subsystem(subsystem):
                    current_subsystems.append(subsystem)
                elif is_same_category_subsystem(subsystem):
                    same_category_subsystems.append(subsystem)
                else:
                    other_subsystems.append(subsystem)
            
            # Combine in priority order
            sorted_subsystems = current_subsystems + same_category_subsystems + other_subsystems
            return sorted_subsystems[:max_count]

        # Prefer top-level, fallback to nested metadata (backward-compatible)
        meta = data.get("metadata", {}) or {}
        # Process consumed_by_subsystems
        consumed_by_subsystems = (
            data.get("consumed_by_subsystems")
            or meta.get("consumed_by_subsystems")
            or []
        )
        if consumed_by_subsystems:
            parts.append(f"<div><strong>Consumed by Subsystems:</strong></div>")
            max_consumed_by_subsystems = 8
            sorted_consumed = sort_subsystems_by_priority(consumed_by_subsystems, max_consumed_by_subsystems)
            for subsystem in sorted_consumed:
                parts.append(format_subsystem_item(subsystem))
            if len(consumed_by_subsystems) > max_consumed_by_subsystems:
                parts.append(f"<div style='margin-left: 20px; font-size: 10px;'>… and {len(consumed_by_subsystems) - max_consumed_by_subsystems} more</div>")

        # Process produced_by_subsystems
        produced_by_subsystems = (
            data.get("produced_by_subsystems")
            or meta.get("produced_by_subsystems")
            or []
        )
        if produced_by_subsystems:
            parts.append(f"<div><strong>Produced by Subsystems:</strong></div>")
            max_produced_by_subsystems = 8
            sorted_produced = sort_subsystems_by_priority(produced_by_subsystems, max_produced_by_subsystems)
            for subsystem in sorted_produced:
                parts.append(format_subsystem_item(subsystem))
            if len(produced_by_subsystems) > max_produced_by_subsystems:
                parts.append(f"<div style='margin-left: 20px; font-size: 10px;'>… and {len(produced_by_subsystems) - max_produced_by_subsystems} more</div>")
        
        # Add legend if there are same_category or current subsystem
        if current_subsystem and (consumed_by_subsystems or produced_by_subsystems):
            has_current = any(is_current_subsystem(sub) for sub in (consumed_by_subsystems + produced_by_subsystems))
            has_same_category = any(is_same_category_subsystem(sub) for sub in (consumed_by_subsystems + produced_by_subsystems))
            
            if has_current or has_same_category:
                parts.append(f"<div style='margin-top: 4px; font-size: 9px; color: #666; font-style: italic;'>")
                legend_items = []
                if has_current:
                    legend_items.append("<span style='color: #0066cc; font-weight: bold;'>current</span>")
                if has_same_category:
                    legend_items.append(f"<span style='color: #0066cc;'>same_category</span> ({current_category})")
                parts.append(f"Legend: {', '.join(legend_items)}")
                parts.append("</div>")
