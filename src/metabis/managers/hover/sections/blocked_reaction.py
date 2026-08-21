from ..interfaces import HoverSection
from ..models import HoverContext, HoverData


class BlockedReactionSection(HoverSection):
    """States that Compass reported no flux for a direction of this reaction.

    A reaction removed as blocked otherwise renders exactly like one that was
    never in the input, which turns a Compass *result* ("no flux under this
    model and medium") into an *absence* ("no data"). WP10/D-65.

    The claim is deliberately scoped to *this dataset*: blockedness depends on
    the medium and model configuration of the Compass run that produced the file,
    which is unknowable for an uploaded matrix, so a claim about the model itself
    would not be supportable.
    """

    def should_include(self, data: HoverData, context: HoverContext) -> bool:
        return bool(data.get("is_in_model", False)) and bool(
            data.get("blocked_directions")
        )

    def build(self, data: HoverData, context: HoverContext) -> str:
        directions = tuple(data.get("blocked_directions") or ())
        if not directions:
            return ""

        if len(directions) > 1 or not data.get("is_reversible", False):
            scope = "this reaction"
            consequence = "it is excluded from the statistical comparison"
        else:
            scope = f"the {directions[0]} direction of this reaction"
            consequence = "only the other direction is compared"

        return (
            "<div><strong>Blocked in this dataset:</strong> Compass reports no "
            f"flux for {scope} in any sample, so {consequence}.</div>"
        )
