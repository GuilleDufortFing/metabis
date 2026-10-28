from dataclasses import dataclass


@dataclass
class VisualTheme:
    """Central configuration for visual styles and colors."""

    # Significance Status
    NOT_SIGNIFICANT: str = "#c7c7c7"  # Neutral gray for non-significant elements

    # Group Comparison Mode Colors (Regulation)
    # Pure red and blue, as in the figures of the GigaScience note: the gradient
    # runs from white to them, as Supplementary Section 1.5 describes.
    UP_REGULATED: str = "#FF0000"    # Red
    DOWN_REGULATED: str = "#0000FF"  # Blue
    # Significant p-values have no direction: purple marks them, against the light
    # grey of the rest. Single-group activity uses purple too, in another mode.
    SIGNIFICANT_PVAL: str = "#B700FF"

    # Single-Group Activity Mode Colors (Experiment Analysis), and any one
    # feature's level against its average (the sample heatmap, a reduction
    # coloured by a feature): purple below the average, light grey at it, gold
    # above, as R Shiny tools draw a centred value (iSEE's purple < black <
    # yellow, Seurat's PurpleAndYellow), with a light grey middle so the average
    # recedes, and gold for yellow so it stands out from that grey. Deliberately
    # not the red/blue of the comparison heatmap - reusing those would invite
    # reading "up-regulated" into a view where nothing is compared - and checked
    # apart from them under colour-vision deficiencies (gold to red, deuteranopia,
    # ΔE 13.8; orange, dark gold and green fall into red).
    ACTIVITY_HIGH: str = "#E6AB02"    # Gold: highest relative activity
    ACTIVITY_MID: str = "#E0E0E0"     # Light grey: typical activity for that feature
    ACTIVITY_LOW: str = "#542788"     # Purple (ColorBrewer PuOr): lowest relative activity
    ACTIVITY_UNAVAILABLE: str = "#c7c7c7"  # Gray: not measured / below coverage threshold

    # Retained aliases: the gradient machinery names its endpoints up/down.
    ACTIVITY_UP: str = "#E6AB02"
    ACTIVITY_DOWN: str = "#542788"

    # General / UI
    NEUTRAL: str = "#FFFFFF"          # Standard white
    # The edge of a point or bar painted on the gradient: grey, so a mark near the
    # gradient's white middle stays visible on a white plot.
    MARK_EDGE: str = "#808080"
    MARK_EDGE_WIDTH: float = 0.6
    # Whatever is selected on the overview: the logo's blue, as the side menu marks
    # the current step (web/src/index.css --logo-blue).
    SELECTED: str = "#4facfe"
    SELECTED_END: str = "#00f2fe"  # the logo's gradient runs from SELECTED to this
    INK_NODE: str = "#0b1118"  # the logo's node fill, the side menu's step hexagons

    # Plot Elements
    # Thresholds are reference lines, not data: a dashed dark grey, the client's ink grey.
    THRESHOLD_LINE_COLOR: str = "#606060"
    THRESHOLD_LINE_DASH: str = "dashed"
    THRESHOLD_LINE_WIDTH: int = 1.2
    THRESHOLD_SVG_DASH: str = "3,1"

    # Plot text and frame, from the React client's tokens (web/src/index.css).
    INK: str = "#152030"        # --navy-2, the client's text colour
    INK_MUTED: str = "#606060"  # --ink-grey, the step cards' secondary text
    FRAME_LINE: str = "#949494"  # --line, the client's control borders
    GRID_LINE: str = "#eeeeee"  # --card-tint
    ROW_SHADE: str = "#f8f8f8"  # --sheet, the step cards' legend box
    FONT: str = "Arial"         # the logo's wordmark, as the client


# Global instance
theme = VisualTheme()
