from typing import Dict, Optional
from metabis.domain.metabolic_model import MetabolicModel


class FocusInstance:
    """
    Recursively retrieve a single parent chain
    (reaction → subsystem → category).
    """

    # Each element type has one direct parent
    DIRECT_PARENT = {
        "reaction": "subsystem",
        "subsystem": "category",
        "category": None,
        "compartment": None,
    }

    def __init__(
        self,
        model: MetabolicModel,
        element_id: str = None,
        element_type: Optional[str] = None,
    ):

        self.model = model
        self.element_id = element_id
        self.element_type = element_type
        if self.element_id and self.element_type is None:
            self.element_type = self.classify_element()

    def __eq__(self, id):
        return self.element_id == id

    def has_element_type(self, element_type:str) -> bool:
        if self.element_id and self.element_type:
            return element_type in self.get_ancestor_ids()
        return False

    # ------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------
    def detect_element_name(self):
        for rxn in self.model.reactions.get_all():
            if self.element_id == rxn.name:
                self.element_id = rxn.id
                return "reaction"
        for subsystem in self.model.subsystems.get_all():
            if self.element_id in subsystem.name:
                self.element_id = subsystem.id
                return "subsystem"

        return

    def classify_element(self) -> str:
        if self.element_id in self.model.get_reactions():
            return "reaction"
        if self.element_id in [subsystem.id for subsystem in self.model.subsystems.get_all()]:
            return "subsystem"
        if self.element_id in self.model.get_subsystem_categories():
            return "category"
        
        element_type = self.detect_element_name()
        if element_type:
            return element_type
        else:
            raise ValueError(f"Unknown element ID: {self.element_id}")

    # ------------------------------------------------------------
    # Single parent type
    # ------------------------------------------------------------
    def get_parent_type(self) -> Optional[str]:
        if not self.element_type:
            return None
        return self.DIRECT_PARENT.get(self.element_type)

    # ------------------------------------------------------------
    # Direct parent ID resolution
    # ------------------------------------------------------------
    def get_parent_id(self) -> Optional[str]:
        parent_type = self.get_parent_type()
        if parent_type is None:
            return None

        if self.element_type == "reaction":
            # find the reaction object
            for rxn in self.model.reactions.get_all():
                if rxn.id == self.element_id:
                    return getattr(rxn, "subsystem_id", None)
            return None

        if self.element_type == "subsystem":
            subs_by_cat = self.model.get_subsystems_by_category()
            for cat, subs in subs_by_cat.items():
                if self.element_id in subs:
                    return cat
            return None

        # category has no parent
        return None

    # ------------------------------------------------------------
    # Recursive chain builder
    # ------------------------------------------------------------
    def get_ancestor_ids(self) -> Dict[str, str]:
        """
        Returns a recursive mapping of {type: id} for all parent levels.
        Example: reaction → {"subsystem": S, "category": C}
        """
        if self.element_id and self.element_type:
            chain = {self.element_type: self.element_id}
            parent_type = self.get_parent_type()
            parent_id = self.get_parent_id()

            if parent_type and parent_id:
                chain[parent_type] = parent_id

                # recurse upward
                parent_inst = FocusInstance(self.model, parent_id)
                chain.update(parent_inst.get_ancestor_ids())

            return chain
        return {}
