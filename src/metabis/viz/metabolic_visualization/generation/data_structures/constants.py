"""Configuration constants and default values for graph visualization.

This module contains all the configuration classes and constants used throughout
the metabolic graph visualization system.

Classes:
    - DefaultDimensions: Default sizes for different glyph types
    - DefaultStyling: Default visual styling parameters
    - DefaultLayout: Layout and positioning parameters
    - DefaultEnzymeGroupLayout: Enzyme group specific layout
    - DefaultOptimization: Graph optimization parameters
    - VisualizationConstants: Runtime visualization styling constants
"""

# ============================================================================
# Default Values and Configuration
# ============================================================================


class DefaultDimensions:
    """Default dimensions for different glyph types."""

    REACTION_WIDTH = 50
    REACTION_HEIGHT = 50

    METABOLITE_MAIN_WIDTH = 60
    METABOLITE_MAIN_HEIGHT = 40

    METABOLITE_INTERMEDIATE_WIDTH = 40
    METABOLITE_INTERMEDIATE_HEIGHT = 30

    GENE_WIDTH = 60
    GENE_HEIGHT = 30

    SUBSYSTEM_WIDTH = 120
    SUBSYSTEM_HEIGHT = 80

    SUBSYSTEM_CATEGORY_RADIUS = 300  # Default radius for category circles

    # Default dimensions for enzyme groups and compartments
    ENZYME_GROUP_WIDTH = 100
    ENZYME_GROUP_HEIGHT = 80

    COMPARTMENT_WIDTH = 800
    COMPARTMENT_HEIGHT = 600

    # Fallback default dimensions
    DEFAULT_WIDTH = 80
    DEFAULT_HEIGHT = 60

    COMPARTMENT_PADDING = 25


class DefaultStyling:
    """Default styling for different glyph types."""

    ENZYME_GROUP_COLOR = "#7B68EE"  # Medium slate blue
    ENZYME_GROUP_WIDTH = 2
    ENZYME_GROUP_DASH = [3, 3]

    ENZYME_GROUP_COLOR_ALT = "#F8C471"  # Alternative color
    ENZYME_GROUP_DASH_ALT = "dashed"


class DefaultLayout:
    """Default layout configuration for graph arrangement."""

    # Compartment arrangement
    COMPARTMENT_PADDING = 150  # Fixed padding between compartments
    MAX_COMPARTMENTS_PER_ROW = 3  # Maximum number of compartments per row

    # Position scaling factors. Graphviz is given node sizes in inches
    # (pixels / 72) and returns coordinates in points, so one point of layout
    # output is one pixel of glyph space and no rescaling is needed. The
    # previous 0.03 compensated for pixel dimensions being read as inches: it
    # divided the resulting 72x drawing back down to a plausible size, which is
    # why the unit error was invisible in the output scale while it distorted
    # every proportion inside it.
    X_SCALE = 1.0
    Y_SCALE = 1.0

    # Compartment boundary padding
    COMPARTMENT_BOUNDARY_PADDING = 25
    COMPARTMENT_BOUNDARY_WIDTH_PADDING = 150
    COMPARTMENT_BOUNDARY_HEIGHT_PADDING = 300  # 2 * 150

    # Subsystem category arrangement
    CATEGORY_PADDING = 200  # Fixed padding between categories
    MAX_CATEGORIES_PER_ROW = 2  # Maximum number of categories per row

    # Category boundary padding (circular)
    CATEGORY_BOUNDARY_PADDING = 50
    CATEGORY_BOUNDARY_RADIUS_PADDING = 100  # Extra radius for category circles


class DefaultEnzymeGroupLayout:
    """Default layout configuration for enzyme groups."""

    # Gene spacing within enzyme groups
    GENE_PADDING = 0  # Padding between genes
    BORDER_PADDING = 5  # Padding between enzyme group border and genes

    # Grid layout thresholds
    SINGLE_COLUMN_THRESHOLD = 4  # Use single column for fewer genes


class DefaultOptimization:
    """Default parameters for layout calculation."""

    # Graphviz measures node size in inches and reports positions in points.
    # Glyph dimensions are pixels, so this converts between the two.
    POINTS_PER_INCH = 72.0

    # GraphViz layout arguments, one flag per option. The previous value was
    # '-Grankdir="LR", -Gsplines=ortho': the comma belongs to no Graphviz
    # syntax, so rankdir was set to `LR,`, rejected, and silently defaulted to
    # top-to-bottom. Every "LR" metabolic graph this project produced was laid
    # out TB. `splines` is not set because nothing consumes Graphviz's routing —
    # every edge is drawn as a straight segment between node centres.
    GRAPHVIZ_RANKDIR = "LR"  # Metabolic maps read left to right
    GRAPHVIZ_NODESEP_INCHES = 0.5  # 36 px between nodes in one rank
    GRAPHVIZ_RANKSEP_INCHES = 0.75  # 54 px between ranks
    GRAPHVIZ_ARGS = (
        f"-Grankdir={GRAPHVIZ_RANKDIR} "
        f"-Gnodesep={GRAPHVIZ_NODESEP_INCHES} "
        f"-Granksep={GRAPHVIZ_RANKSEP_INCHES}"
    )

    # Spring layout parameters
    SPRING_K = 0.8  # Spring layout spring constant
    SPRING_ITERATIONS = 100  # Spring layout iterations

    # Topology preparation and component packing (the automatic-layout redesign).
    #
    # A shared "main" metabolite touched by more than HUB_METABOLITE_DEGREE
    # reactions is a star whose arms cross everything between them, and it is the
    # dominant source of crossings in the automatic maps. Curators answer this by
    # duplicating the metabolite beside a reaction — but only on the arms that are
    # shortcuts, never on the arms that carry the pathway, which is why a curated
    # map still reads as one network. split_hub_metabolites duplicates exactly the
    # arms a spanning forest does not need, so this threshold decides which
    # metabolites are *eligible* for duplication and connectivity decides which of
    # their arms actually are.
    #
    # Duplicating every arm above this degree instead — the rule that shipped
    # between 2026-07-25 and this change — cut the largest connected component of a
    # compartment to roughly a third of the curated map's and turned the drawings
    # into fields of detached motifs.
    #
    # Every copy keeps the parent's original_id, so colouring and selection still
    # resolve to one entity — the same duplication the per-reaction intermediate
    # metabolites already use.
    HUB_METABOLITE_DEGREE = 3

    # Connectivity is not the only thing worth preserving. A metabolite that is
    # the sole path to dozens of reactions — a shared CoA or carnitine pool rather
    # than a pathway junction — has no redundant arm to duplicate, so the rule
    # above keeps all of them and hands the layout engine a 66-armed star. The
    # curators do not draw that: across the 104 curated Human1 maps the highest
    # metabolite degree is 14 and only one map exceeds 12, while our uncapped
    # layouts put 15 of 145 above 12. Above this cap a metabolite keeps the arms
    # holding the largest structures and the rest become per-reaction copies, which
    # is the curators' own budget applied to the pools.
    HUB_METABOLITE_DEGREE_CAP = 12

    # After splitting, a compartment is a forest of small motifs. Laying each
    # connected component out on its own and shelf-packing the results to a target
    # aspect ratio replaces Graphviz's one-dimensional component packing, which
    # spread everything along a single axis into a letterbox band.
    COMPONENT_TARGET_ASPECT = 1.6  # width / height goal for the packed drawing
    COMPONENT_PACKING_GAP = 120.0  # pixels of clear space between packed components

    # Enzyme groups are demoted to satellites: they carry exactly one edge, they
    # are large boxes, and letting them take part in rank assignment distorts the
    # metabolic drawing. They take no part in it, but their footprint is reserved
    # on the reaction they belong to, this far above it, so the layout engine keeps
    # the space clear and the group cannot land on a neighbouring glyph. Reserving
    # nothing and reattaching afterwards — the behaviour that shipped on
    # 2026-07-25 — put an enzyme group on top of another component's reactions and
    # metabolites in every subsystem measured.
    ENZYME_SATELLITE_GAP = 70.0


class MetabolicSerpentineLayoutDefaults:
    """Default configuration for the custom serpentine metabolic layout."""
    MULTIPLY = 6

    # This layout emits its own coordinates rather than Graphviz points, and its
    # spacing constants below are sized for a drawing scaled down by this
    # factor. It used to be applied to every engine's output as
    # DefaultLayout.X_SCALE; it belongs to this one.
    POSITION_SCALE = 0.03

    REACTIONS_PER_ROW = 12   # Maximum number of reactions per serpentine row
    REACTION_HORIZONTAL_SPACING = 3200 * MULTIPLY # Distance between reactions in a row
    REACTION_VERTICAL_SPACING = 2600  * MULTIPLY# Vertical distance between rows
    SERPENTINE_ROW_INDENT = 1600  * MULTIPLY# Horizontal indent applied to odd rows

    METABOLITE_HORIZONTAL_OFFSET = 1900  * MULTIPLY# Base horizontal distance from reaction
    METABOLITE_VERTICAL_SPACING = 900  * MULTIPLY# Vertical spacing between stacked metabolites
    INTERMEDIATE_VERTICAL_OFFSET = 1200  * MULTIPLY# Offset for intermediate metabolites (above/below)
    SHARED_METABOLITE_JITTER = 450  * MULTIPLY# Small shift to reduce overlaps on reused metabolites

    ENZYME_VERTICAL_OFFSET = 1800  * MULTIPLY# Distance to place enzyme groups above reactions

    MIN_EDGE_LENGTH = 1600  * MULTIPLY# Minimum edge length when snapping to angle grid
    ANGLE_SNAP_DEGREES = [0, 45, -45, 90, -90, 135, -135, 180]  # Allowed edge angles

    FIXED_METABOLITE_ANGLES = {}  # Optional overrides per metabolite original_id → angle

    BASE_ANCHOR_X = 0  # Layout origin X
    BASE_ANCHOR_Y = 0  # Layout origin Y


# ============================================================================
# Visualization Constants
# ============================================================================


class VisualizationConstants:
    """Constants for visualization styling."""

    # Alpha values
    HIGH_ALPHA = 1.0
    LOW_ALPHA = 0.5
    TRANSPARENT_ALPHA = 0.0
    INFLUENCE_EDGE_ALPHA = 0.8
    SUBSYSTEM_CONNECTION_MIN_ALPHA = 0.35
    SUBSYSTEM_CONNECTION_MAX_ALPHA = 0.8

    # Default sizes
    REACTION_SIZE = 40
    MAIN_METABOLITE_SIZE = 20
    INTERMEDIATE_METABOLITE_SIZE = 15

    # Default dimensions
    GENE_WIDTH = 60
    GENE_HEIGHT = 30
    MAIN_METABOLITE_WIDTH = 60
    MAIN_METABOLITE_HEIGHT = 30
    INTERMEDIATE_METABOLITE_RADIUS = 4

    # Angles
    REACTION_ANGLE = 0.785398  # 45 degrees in radians

    # Colors
    DEFAULT_GENE_COLOR = "gray"
    DEFAULT_REACTION_COLOR = "gray"
    DEFAULT_SUBSYSTEM_COLOR = "#ADD8E6"  # Light blue
    ENZYME_GROUP_COLOR = "#F8C471"
    ENZYME_GROUP_DASH = "dashed"
    CYTOSOL_LINE_COLOR = "white"
    DEFAULT_LINE_COLOR = "#606060"
    # Global stroke/border color for glyph outlines and edge overlays
    GLOBAL_BORDER_COLOR = "#606060"
    # GLOBAL_BORDER_COLOR = DEFAULT_LINE_COLOR

    # Line widths
    DEFAULT_LINE_WIDTH = 1.0
    ENZYME_CONNECTION_WIDTH = 1.0
    SUBSYSTEM_CONNECTION_WIDTH = 1.0
    INFLUENCE_LINE_WIDTH = 1.5
    COMPARTMENT_LINE_WIDTH = 2
    ARROWHEAD_LINE_WIDTH = 0.25

    # Edge glow/halo styling (for always-visible edges with colored background)
    EDGE_GLOW_ENABLED = True  # Toggle to enable/disable glow rendering
    # Foreground stroke color (always visible) — centralized to GLOBAL_BORDER_COLOR
    EDGE_OVERLAY_COLOR = GLOBAL_BORDER_COLOR
    EDGE_NEUTRAL_COLOR = "#d0d0d0"  # Optional neutral when metric ~ 0
    EDGE_GLOW_WIDTH_DELTA = 3  # Extra pixels for underlay vs overlay width
    EDGE_GLOW_ALPHA = 0.7  # Underlay alpha
    EDGE_GLOW_MATCH_DASH = True  # Whether underlay follows dash style
    EDGE_GLOW_SECOND_LAYER = False  # Optional second blur-like layer
    EDGE_GLOW_SECOND_DELTA = 7
    EDGE_GLOW_SECOND_ALPHA = 0.35

    # Plot styling
    # No frame around a graph: the page's card already frames it.
    PLOT_OUTLINE_WIDTH = 1
    PLOT_OUTLINE_ALPHA = 1.0
    PLOT_OUTLINE_COLOR = None
    COMPARTMENT_BORDER_RADIUS = 10

    # Font sizes (CSS format)
    FONT_SIZE_METABOLITE_MAIN = "8pt"
    FONT_SIZE_METABOLITE_INTERMEDIATE = "6pt"
    FONT_SIZE_GENE = "6pt"
    FONT_SIZE_REACTION = "8pt"
    FONT_SIZE_COMPARTMENT = "13px"
    FONT_SIZE_SUBSYSTEM = "8pt"

    # Label positioning offsets
    LABEL_OFFSET_DOWN = -5  # For metabolites, genes, reactions
    LABEL_OFFSET_UP = 5  # For intermediate metabolites

    # Arrowhead styling
    ARROWHEAD_SIZE = 18
    ARROWHEAD_ANGLE_RADIANS = 3.14159 / 6  # 30 degrees = π/6 radians

    # Connection geometry
    CONNECTION_OFFSET = 20  # Default offset from node edges

    # Missing data styling
    MISSING_DATA_COLOR = "white"
    MISSING_DATA_LINE_DASH = "dashed"  # Dashed line pattern
    MISSING_DATA_ALPHA = 0.7
    MISSING_DATA_OUTLINE_COLOR = "gray"  # For dashed borders
    MISSING_CONNECTION_COLOR = "gray"  # Color for missing connections

    # Subsystem circle sizing constants
    SUBSYSTEM_MIN_RADIUS = 5
    SUBSYSTEM_MAX_RADIUS = 50
    SUBSYSTEM_BASE_RADIUS = 10
    SUBSYSTEM_SIZE_SCALE_FACTOR = 1.0  # More conservative scaling

    # Subsystem sizing scaling method: 'log', 'sqrt', 'linear', 'power'
    SUBSYSTEM_SCALING_METHOD = "linear"  # Square root scaling for better visuals
    SUBSYSTEM_POWER_EXPONENT = 0.7  # Used when scaling method is 'power'

    # Subsystem category boundary styling
    # Categories are neutral containers, in the React client's tokens: the step
    # cards' legend-box fill (--sheet) inside a GOV.UK border grey (--rule-grey),
    # named in the client's navy (--navy-2).
    DEFAULT_CATEGORY_COLOR = "#f8f8f8"
    DEFAULT_CATEGORY_LINE_COLOR = "#b1b4b6"
    DEFAULT_CATEGORY_LINE_WIDTH = 1.2
    DEFAULT_CATEGORY_LINE_DASH = "solid"  # Solid lines for categories
    DEFAULT_CATEGORY_FILL_ALPHA = 1.0
    CATEGORY_LABEL_COLOR = "#152030"
    # A subsystem with no data is neutral grey, as Supplementary Section 1.5 says.
    SUBSYSTEM_NO_DATA_COLOR = "#c7c7c7"
    CATEGORY_LABEL_ALPHA = 0.8  # Slightly transparent category labels

    # Category boundary sizing
    CATEGORY_MIN_RADIUS = 100  # Minimum category boundary radius
    CATEGORY_DEFAULT_RADIUS = 200  # Default category boundary radius
    CATEGORY_BOUNDARY_PADDING = 50  # Extra padding around subsystems in category
    DEFAULT_SUBSYSTEM_CATEGORY_ALPHA = 0.7  # Default alpha for subsystem categories

    # Category font styling
    FONT_SIZE_CATEGORY = "11px"  # Category names, under each category
    CATEGORY_LABEL_OFFSET = 60  # Layout units between a category's circle and its name
    CATEGORY_LABEL_LINE_LENGTH = 16  # Characters per line of a category's name (spaced_network)
    CATEGORY_LABEL_BAND = 1000  # Layout units the name may take below its circle
    FONT_FAMILY = "Arial"  # As the React client

    # Subsystem name text wrapping
    SUBSYSTEM_NAME_MAX_LINE_LENGTH = 25  # Maximum characters per line
    SUBSYSTEM_NAME_MAX_LINES = 3  # Maximum number of lines for subsystem names

    # Default category color for unmapped categories
    DEFAULT_CATEGORY_COLOR_UNMAPPED = "#34495E"  # Dark gray

    # Dynamic alpha for subsystem labels based on zoom level
    SUBSYSTEM_LABEL_ALPHA_MIN = 0  # Minimum alpha for subsystem labels
    SUBSYSTEM_LABEL_ALPHA_MAX = 0.9  # Maximum alpha for subsystem labels
    SUBSYSTEM_LABEL_ZOOM_THRESHOLD_MIN = 0.12  # Zoom level for minimum alpha
    SUBSYSTEM_LABEL_ZOOM_THRESHOLD_MAX = 0.18  # Zoom level for maximum alpha
    SUBSYSTEM_UNSELECTED_ALPHA = 0.1 # For unselected subsystem in metabolic_network graph_type

    # Different thresholds for other glyph types
    METABOLITE_LABEL_ALPHA_MIN = 0.0  # Metabolites can go fully transparent
    METABOLITE_LABEL_ALPHA_MAX = 1.0
    METABOLITE_LABEL_ZOOM_THRESHOLD_MIN = 0.15  # Show at higher zoom levels
    METABOLITE_LABEL_ZOOM_THRESHOLD_MAX = 0.2

    GENE_LABEL_ALPHA_MIN = 0.0  # Genes can go fully transparent
    GENE_LABEL_ALPHA_MAX = 1.0
    GENE_LABEL_ZOOM_THRESHOLD_MIN = 0.15  # Show at higher zoom levels
    GENE_LABEL_ZOOM_THRESHOLD_MAX = 0.2

    REACTION_LABEL_ALPHA_MIN = 0  # Reactions always slightly visible
    REACTION_LABEL_ALPHA_MAX = 1.0
    REACTION_LABEL_ZOOM_THRESHOLD_MIN = 0.15
    REACTION_LABEL_ZOOM_THRESHOLD_MAX = 0.2

    COMPARTMENT_LABEL_ALPHA_MIN = 1  # Compartments always visible
    COMPARTMENT_LABEL_ALPHA_MAX = 1.0
    COMPARTMENT_LABEL_ZOOM_THRESHOLD_MIN = 0.1  # Show at all zoom levels
    COMPARTMENT_LABEL_ZOOM_THRESHOLD_MAX = 0.2

    CATEGORY_UNSELECTED_ALPHA = 0.1

    # ============================================================================
    # Subsystem Comparison Plot Label Visibility Constants
    # ============================================================================
    # These control zoom-based label visibility in the subsystem comparison plot
    # When zoomed out, only category labels are visible; when zoomed in, subsystem labels appear

    # Subsystem labels in comparison plot
    COMPARISON_SUBSYSTEM_LABEL_ALPHA_MIN = 0.0  # Fully transparent when zoomed out
    COMPARISON_SUBSYSTEM_LABEL_ALPHA_MAX = 1.0  # Fully visible when zoomed in
    # Zoom thresholds based on visible y-range span (number of factors visible)
    # When span > threshold_max factors, labels are at min alpha
    # When span < threshold_min factors, labels are at max alpha
    COMPARISON_SUBSYSTEM_ZOOM_THRESHOLD_MIN = 8  # Show labels when <= 8 subsystems visible
    COMPARISON_SUBSYSTEM_ZOOM_THRESHOLD_MAX = 25  # Hide labels when >= 25 subsystems visible

    # Category labels in comparison plot (always visible)
    COMPARISON_CATEGORY_LABEL_ALPHA_MIN = 0.6  # Slightly dimmed when zoomed in
    COMPARISON_CATEGORY_LABEL_ALPHA_MAX = 1.0  # Full visibility when zoomed out
    COMPARISON_CATEGORY_ZOOM_THRESHOLD_MIN = 8  # Dim when few subsystems visible
    COMPARISON_CATEGORY_ZOOM_THRESHOLD_MAX = 25  # Full visibility when many visible

    # Font sizes for comparison plot labels
    COMPARISON_SUBSYSTEM_FONT_SIZE = "9pt"
    COMPARISON_CATEGORY_FONT_SIZE = "11pt"

    # Label positioning offsets
    COMPARISON_LABEL_X_OFFSET = 5  # Pixels from left edge
    COMPARISON_CATEGORY_LABEL_Y_OFFSET = 0.3  # Offset within category group
