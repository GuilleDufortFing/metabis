"""Core data structures for metabolic entities.

This module defines the fundamental biological entities used in the metabolic model:
- Reaction: Metabolic reactions with stoichiometry and gene associations
- Metabolite: Chemical compounds with properties and compartment information
- Gene: Genes with symbols, aliases, and reaction associations
- Subsystem: Metabolic pathways containing related reactions and genes
- Compartment: Cellular compartments containing reactions and metabolites

All entities use ID-based references for relationships to enable O(1) lookups
and maintain data consistency.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class Reaction:
    """Core data structure for metabolic reactions.

    Contains all reaction metadata and cross-references to related entities.
    Data loaded from the reference-model reaction catalog and formulas tables.
    All relationships are stored as ID references for O(1) access.
    """

    # Primary identification
    id: str  # Primary ID from the reaction catalog
    name: str  # Human readable name from the reaction catalog

    # Chemical information
    equation: str  # Human-readable chemical equation from the reaction catalog
    # Structured stoichiometric formula from the reaction formulas table
    formula: Optional[str] = None

    # Biological context
    subsystem_id: str = ""  # Reference to subsystem from the reaction catalog
    compartment_id: str = ""  # Reference to compartment from the reaction catalog

    # Metabolite relationships (stored as IDs)
    reactants: List[str] = field(default_factory=list)  # Metabolite IDs (substrates)
    products: List[str] = field(default_factory=list)  # Metabolite IDs (products)
    is_reversible: bool = False  # True if reaction is reversible (from meta.csv)
    # metabolite_id -> coefficient
    stoichiometry: Dict[str, float] = field(default_factory=dict)

    # Gene associations
    # Gene IDs (from GENE ASSOCIATION)
    associated_genes: List[str] = field(default_factory=list)
    gene_rule: str = ""  # GPR rule as string from the reaction catalog
    # GPR rule with gene symbols from the reaction catalog
    gene_rule_symbol: str = ""

    # Quality and annotation information
    confidence_score: Optional[float] = None  # Confidence score from the reaction catalog
    references: str = ""  # Literature references from the reaction catalog
    miriam: str = ""  # MIRIAM cross-references from the reaction catalog

    # ID translations
    # format -> alternative_id from the reaction identifier crosswalk
    id_mappings: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metabolite:
    """Core data structure for metabolites/compounds.

    Contains metabolite metadata and cross-references to reactions.
    Data loaded from the metabolite identifier crosswalk table.

    Note: Chemical properties like formula, charge, and mass are not available
    in the source data files and have been removed from this structure.
    """

    # Primary identification
    id: str  # Primary ID from the metabolite identifier crosswalk
    name: str  # Common name from the metabolite identifier crosswalk

    # Biological context
    # Can exist in multiple compartments (extracted from metabolite ID suffixes)
    compartment_ids: Set[str] = field(default_factory=set)
    is_intermediate: bool = False  # True for cofactors like ATP, NADH

    # Reaction relationships
    # Reaction IDs that produce this
    produced_by: List[str] = field(default_factory=list)
    # Reaction IDs that consume this
    consumed_by: List[str] = field(default_factory=list)

    # ID translations
    # format -> alternative_id
    id_mappings: Dict[str, str] = field(default_factory=dict)


@dataclass
class Gene:
    """Core data structure for genes.

    Contains gene metadata and cross-references to reactions and subsystems.
    Data loaded from the gene identifier crosswalk table.
    """

    # Primary identification
    id: str  # Primary ID from the gene identifier crosswalk
    symbol: str = ""  # Gene symbol from the gene identifier crosswalk
    name: str = ""  # Full gene name from the gene identifier crosswalk

    # Gene information
    aliases: List[str] = field(default_factory=list)  # Alternative names

    # Biological context
    # Cellular locations from the gene identifier crosswalk
    compartments: Set[str] = field(default_factory=set)
    # Subsystems this gene participates in (computed from reactions)
    subsystem_ids: Set[str] = field(default_factory=set)

    # Reaction relationships
    # Reaction IDs this gene catalyzes (computed from gene associations)
    associated_reactions: List[str] = field(default_factory=list)

    # ID translations from the gene identifier crosswalk columns
    # format -> alternative_id
    id_mappings: Dict[str, str] = field(default_factory=dict)


@dataclass
class Subsystem:
    """Core data structure for metabolic subsystems/pathways.

    Contains all entities belonging to this subsystem.
    """

    # Primary identification
    id: str  # Primary ID (cleaned subsystem name)
    name: str  # Human readable name

    # Classification
    category: str = "Uncategorized"  # Metabolic category

    # Content (stored as ID sets for fast lookup)
    reaction_ids: Set[str] = field(default_factory=set)  # Reactions in this subsystem
    gene_ids: Set[str] = field(default_factory=set)  # Genes involved
    metabolite_ids: Set[str] = field(default_factory=set)  # Metabolites involved

    # Statistics (computed)
    reaction_count: int = 0
    gene_count: int = 0
    metabolite_count: int = 0


@dataclass
class Compartment:
    """Core data structure for cellular compartments.

    Contains all entities located in this compartment.
    """

    # Primary identification
    id: str  # Short code (c, m, r, etc.)
    name: str  # Full name (Cytosol, Mitochondria, etc.)

    # Content (stored as ID sets for fast lookup)
    reaction_ids: Set[str] = field(default_factory=set)  # Reactions in this compartment
    # Metabolites in this compartment
    metabolite_ids: Set[str] = field(default_factory=set)
    gene_ids: Set[str] = field(default_factory=set)  # Genes active in this compartment

    # Statistics (computed)
    reaction_count: int = 0
    metabolite_count: int = 0
    gene_count: int = 0


@dataclass
class SubsystemNeighborRelationship:
    """Represents a directional neighbor relationship between subsystems.

    Subsystem B is a neighbor of Subsystem A if A produces a main metabolite
    that is consumed by B. This captures metabolic flow between pathways.
    """

    # Source subsystem that produces the metabolite
    producing_subsystem_id: str

    # Target subsystem that consumes the metabolite
    consuming_subsystem_id: str

    # The main metabolite that flows between the subsystems
    metabolite_id: str

    # Reactions in producing subsystem that produce this metabolite
    producing_reactions: List[str] = field(default_factory=list)

    # Reactions in consuming subsystem that consume this metabolite
    consuming_reactions: List[str] = field(default_factory=list)

    def __hash__(self) -> int:
        """Enable use in sets and as dict keys."""
        return hash(
            (
                self.producing_subsystem_id,
                self.consuming_subsystem_id,
                self.metabolite_id,
            )
        )

    def __eq__(self, other) -> bool:
        """Enable comparison for uniqueness."""
        if not isinstance(other, SubsystemNeighborRelationship):
            return False
        return (
            self.producing_subsystem_id == other.producing_subsystem_id
            and self.consuming_subsystem_id == other.consuming_subsystem_id
            and self.metabolite_id == other.metabolite_id
        )
