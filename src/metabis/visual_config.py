from dataclasses import dataclass


@dataclass
class VisualTheme:
    """Central configuration for visual styles and colors."""

    # Significance Status
    NOT_SIGNIFICANT: str = "#c7c7c7"  # Neutral gray for non-significant elements

    # Group Comparison Mode Colors (Regulation)
    UP_REGULATED: str = "#FF0000"    # Red
    DOWN_REGULATED: str = "#0000FF"  # Blue
    # Significant p-value Colors
    SIGNIFICANT_PVAL: str = "#B700FF" # Purple for significant p-values

    # Single-Group Activity Mode Colors (Experiment Analysis)
    # Activity is a magnitude, not a direction, so its ramp is sequential rather than
    # diverging: low activity is white, typical activity is amber, high activity is
    # purple. Deliberately not the red/blue of the comparison heatmap - reusing those
    # would invite reading "up-regulated" into a view where nothing is compared.
    # Lightness decreases monotonically along the ramp, so "darker = more active"
    # survives greyscale printing and most colour-vision deficiencies.
    ACTIVITY_HIGH: str = "#B700FF"    # Purple: highest relative activity
    ACTIVITY_MID: str = "#FFC300"     # Amber: typical activity for that feature
    ACTIVITY_LOW: str = "#FFFFFF"     # White: lowest relative activity
    ACTIVITY_UNAVAILABLE: str = "#c7c7c7"  # Gray: not measured / below coverage threshold

    # Retained aliases: the gradient machinery names its endpoints up/down.
    ACTIVITY_UP: str = "#B700FF"
    ACTIVITY_DOWN: str = "#FFFFFF"

    # General / UI
    NEUTRAL: str = "#FFFFFF"          # Standard white

    # Plot Elements
    THRESHOLD_LINE_COLOR: str = "#B700FF" # Red
    THRESHOLD_LINE_DASH: str = "dashed"
    THRESHOLD_LINE_WIDTH: int = 1.2
    THRESHOLD_SVG_DASH: str = "3,1"

# Global instance
theme = VisualTheme()
