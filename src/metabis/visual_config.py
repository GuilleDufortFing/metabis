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
    
    # Activity Mode Colors (Experiment Analysis)
    ACTIVITY_UP: str = "#B700FF"      # Purple
    ACTIVITY_DOWN: str = "#FFFFFF"    # White
    
    # General / UI
    NEUTRAL: str = "#FFFFFF"          # Standard white
    
    # Plot Elements
    THRESHOLD_LINE_COLOR: str = "#B700FF" # Red
    THRESHOLD_LINE_DASH: str = "dashed"
    THRESHOLD_LINE_WIDTH: int = 1.2
    THRESHOLD_SVG_DASH: str = "3,1"

# Global instance
theme = VisualTheme()
