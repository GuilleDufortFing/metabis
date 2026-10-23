"""Modern implementation of the MetabolicModel interface.

This module provides a new implementation of the MetabolicModel interface that uses
structured data types and follows modern design patterns for better integration
with visualization components and improved maintainability.

Key features:
- Type-safe data structures using dataclasses
- O(1) entity access with EntityCollection
- Modular data loading and processing
- Better separation of concerns
- Integration with visualization glyph system
- Improved error handling and validation
"""

import copy
import hashlib
import io
import os
import pickle
import time
from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

from metabis.resource_registry import DEFAULT_RESOURCES_ROOT, ResourceRegistry

from .category_manager import CategoryManager
from .collections import EntityCollection
from .entities import (
    Compartment,
    Gene,
    Metabolite,
    Reaction,
    Subsystem,
    SubsystemNeighborRelationship,
)
from .loaders import (
    build_gene_reaction_associations,
    build_metabolite_reaction_relationships,
    build_subsystem_content_relationships,
    build_subsystem_neighbor_relationships,
    create_compartments_from_entities,
    create_subsystems_from_reactions,
    load_genes_from_file,
    load_metabolites_from_file,
    load_reactions_from_files,
    load_subsystem_metabolite_classification,
    split_transport_by_compartment,
    TRANSPORT_BETWEEN,
    TRANSPORT_PARTS,
    TRANSPORT_SUBSYSTEM,
    TRANSPORT_WITHIN,
    _clean_subsystem_name,
)
from .reaction_names import short_reaction_name

# The subsystems the divided transport is drawn from (loaders.transport_part).
TRANSPORT_PART_IDS = frozenset(
    _clean_subsystem_name(name)
    for name in (*TRANSPORT_PARTS.values(), TRANSPORT_BETWEEN, TRANSPORT_WITHIN)
)

# How the model is built from its source files; the cache's name includes it, so a
# model built another way is never read back from an old cache. "transport by
# compartment": Human1's transport divided as Recon3D divides it (loaders).
MODEL_BUILD_REVISION = "transport by compartment"


class MetabolicModelInterface(ABC):
    """Abstract base class defining the interface for metabolic model implementations.

    This interface defines all the methods that a metabolic model must implement,
    allowing for different implementations while maintaining API compatibility.
    """

    @abstractmethod
    def get_reaction_metadata(
        self, reaction_id: str = None
    ) -> Dict[str, Dict[str, str]]:
        """Get metadata for reactions."""
        pass

    @abstractmethod
    def get_gene_metadata(self, gene_id: str = None) -> Dict[str, Dict[str, str]]:
        """Get metadata for genes."""
        pass

    @abstractmethod
    def get_reactions(self, format: str = None) -> List[str]:
        """Get a list of reaction IDs in the specified format."""
        pass

    @abstractmethod
    def get_genes(self, format: str = None) -> List[str]:
        """Get a list of gene IDs in the specified format."""
        pass

    @abstractmethod
    def get_metabolites(self) -> List[str]:
        """Get a list of metabolite IDs in the specified format."""
        pass

    @abstractmethod
    def get_general_intermediate_metabolites(self) -> List[str]:
        """Get a list of general intermediate metabolite IDs."""
        pass

    @abstractmethod
    def get_subsystems(self) -> List[str]:
        """Get a list of all subsystems in the model."""
        pass

    @abstractmethod
    def get_reactions_for_subsystem(self, subsystem: str) -> List[str]:
        """Get all reactions in a specific subsystem."""
        pass

    @abstractmethod
    def get_genes_for_subsystem(self, subsystem: str) -> List[str]:
        """Get all genes in a specific subsystem."""
        pass

    @abstractmethod
    def get_available_reaction_formats(self) -> List[str]:
        """Get available reaction name formats."""
        pass

    @abstractmethod
    def get_available_gene_formats(self) -> List[str]:
        """Get available gene name formats."""
        pass

    @abstractmethod
    def filter_by_subsystem_and_compartment(self, filters: List[str]) -> None:
        """Filter the model to include only specified subsystems and compartments."""
        pass

    @abstractmethod
    def translate_gene(
        self,
        gene: str,
        input_format: str = "ensmbl_id",
        output_format: str = "geneSymbols",
    ) -> str:
        """Translate a gene ID from one format to another."""
        pass

    @abstractmethod
    def translate_reaction(
        self,
        reaction: str,
        input_format: str = "mar_id",
        output_format: str = "rxnBiGGID",
    ) -> str:
        """Translate a reaction ID from one format to another."""
        pass

    @abstractmethod
    def translate_metabolite(
        self,
        metabolite: str,
        input_format: str = "met_id",
        output_format: str = "metName",
    ) -> str:
        """Translate a metabolite ID from one format to another."""
        pass

    @abstractmethod
    def retrieve_element_id(
        self, element_name: str, graph_type: str
    ) -> Tuple[str, str]:
        """Retrieve element ID and display title from element name.

        Args:
            element_name: Human-readable name (e.g., "Glycolysis / Gluconeogenesis")
            graph_type: One of "subsystem", "compartment", "metabolic_network"

        Returns:
            Tuple of (element_id, display_title)

        Raises:
            ValueError: If element not found or graph_type invalid
        """
        pass


class MetabolicModel(MetabolicModelInterface):
    """Modern implementation of the metabolic model interface.

    This implementation uses structured data types and follows modern design patterns
    for better integration with visualization components and improved maintainability.

    Key features:
    - Type-safe data structures using dataclasses
    - Modular data loading and processing
    - Better separation of concerns
    - Integration with visualization glyph system
    - Improved error handling and validation
    """

    def __init__(
        self,
        base_path: str = DEFAULT_RESOURCES_ROOT,
        lazy_loading: bool = False,
        use_cache: bool = True,
        use_subsystem_classification: bool = True,
        subsystem_classification_file_path: Optional[str] = None,
    ):
        """Initialize the modern metabolic model.

        Args:
            base_path (str): Path to the directory containing the model files
            lazy_loading (bool): If True, data will be loaded on-demand rather
                than at initialization
            use_cache (bool): If True, use file-based caching to speed up
                initialization (default: True)
            use_subsystem_classification (bool): If True, load subsystem
                categories from classification file (default: True)
            subsystem_classification_file_path (str): Optional custom path to
                subsystem classification file. If None, uses default path.
        """
        self.resource_registry = ResourceRegistry(resources_root=base_path)
        self.base_path = self.resource_registry.resources_root
        self.lazy_loading = lazy_loading
        self.use_cache = use_cache
        self.use_subsystem_classification = use_subsystem_classification
        self.subsystem_classification_file_path = subsystem_classification_file_path

        # Modern storage containers using EntityCollection
        self.reactions: EntityCollection[Reaction] = EntityCollection()
        self.metabolites: EntityCollection[Metabolite] = EntityCollection()
        self.genes: EntityCollection[Gene] = EntityCollection()
        self.subsystems: EntityCollection[Subsystem] = EntityCollection()
        self.compartments: EntityCollection[Compartment] = EntityCollection()

        # Per-subsystem metabolite classification
        # Maps subsystem names to dicts of {"main": [...], "intermediate": [...]}
        self.subsystem_metabolite_classification: Dict[str, Dict[str, List[str]]] = {}

        # Subsystem neighbor relationships
        # List of directional relationships between subsystems
        self.subsystem_neighbor_relationships: List[SubsystemNeighborRelationship] = []

        # Legacy relationship mappings - maintained for compatibility
        self._gene_to_subsystems: Optional[Dict[str, List[str]]] = None
        self._subsystem_to_genes: Optional[Dict[str, List[str]]] = None
        self._reaction_to_metabolites: Optional[Dict[str, List[str]]] = None
        self._metabolite_to_reactions: Optional[Dict[str, List[str]]] = None

        # Cache for expensive operations
        self._cache: Dict[str, any] = {}

        # Category manager for subsystem categories and colors
        # Initialized after model data is loaded
        self._category_manager: Optional[CategoryManager] = None

        # Initialize data unless lazy loading is enabled
        if not lazy_loading:
            self._initialize_all_data()

    # ============================================================================
    # Initialization and Data Loading Methods
    # ============================================================================

    def reload_all_data(self) -> None:
        """Reload the full reference model, discarding any subsystem filter.

        The public inverse of :meth:`filter_by_subsystem_and_compartment`: it
        returns the model to its complete state so a caller can reset a filter
        without reaching into initialization internals.
        """
        self._initialize_all_data()

    def copy(self) -> "MetabolicModel":
        """Return a fresh, fully independent copy of this model.

        The copy carries all of this model's current state -- applied alignment
        corrections, an active subsystem/compartment filter, and any custom
        subsystem categories -- but is a distinct object with its own entity
        collections and category manager. Mutating one never reaches the other.

        This is the primitive behind a *model transition* (AM8): a caller that
        wants to filter, reset, or recategorize the installed model copies it,
        applies the change to the copy, and installs the copy through
        :meth:`AnalysisWorkspace.apply_model_transition`. Working on a fresh
        object is required, not stylistic -- source identity is keyed on object
        identity (ADR-0007), so re-identifying a model mutated in place would
        return its stale fingerprint, and an in-flight worker still holding the
        original could observe a torn read (MR-01).
        """
        return copy.deepcopy(self)

    def _initialize_all_data(self) -> None:
        """Initialize all data structures using individual loaders."""
        # Try to load from cache first
        if self.use_cache and self._is_cache_valid():
            if self._load_from_cache():
                return  # Successfully loaded from cache

        # Fall back to normal loading
        try:
            reference_model_files = self.resource_registry.get_reference_model_file_paths()
            meta_file = reference_model_files["reaction_catalog"]
            gem_file = reference_model_files["reaction_formulas"]
            reactions_mapping_file = reference_model_files[
                "reaction_identifier_crosswalk"
            ]
            metabolites_file = reference_model_files[
                "metabolite_identifier_crosswalk"
            ]
            genes_file = reference_model_files["gene_identifier_crosswalk"]

            with redirect_stdout(io.StringIO()):
                reactions = load_reactions_from_files(
                    meta_file, gem_file, reactions_mapping_file
                )
                split_transport_by_compartment(reactions)
                metabolites = load_metabolites_from_file(metabolites_file)
                genes = load_genes_from_file(genes_file)

                # Create subsystems without categories (categories loaded separately)
                subsystems = create_subsystems_from_reactions(reactions)
                compartments = create_compartments_from_entities(
                    reactions, metabolites, genes
                )

                # Build cross-entity relationships
                build_metabolite_reaction_relationships(metabolites, reactions)
                build_gene_reaction_associations(genes, reactions)
                build_subsystem_content_relationships(
                    subsystems, reactions, genes, metabolites
                )

                # Load per-subsystem metabolite classification
                general_intermediates = [
                    "FADH2",
                    "H+",
                    "H2O",
                    "O2",
                    "NADH",
                    "FAD",
                    "NAD+",
                    "NADPH",
                    "Pi",
                    "ADP",
                    "ATP",
                    "UDP",
                    "CoA",
                    "NADP+",
                    "H2O2",
                    "CO2",
                ]
                self.subsystem_metabolite_classification = (
                    load_subsystem_metabolite_classification(
                        subsystems,
                        metabolites,
                        general_intermediates,
                    )
                )

                # Build subsystem neighbor relationships
                self.subsystem_neighbor_relationships = (
                    build_subsystem_neighbor_relationships(
                        reactions,
                        subsystems,
                        metabolites,
                        self.subsystem_metabolite_classification,
                    )
                )

            # Populate EntityCollections with loaded data
            for reaction in reactions:
                self.reactions.add(reaction)

            for metabolite in metabolites:
                self.metabolites.add(metabolite)

            for gene in genes:
                self.genes.add(gene)

            for subsystem in subsystems:
                self.subsystems.add(subsystem)

            for compartment in compartments:
                self.compartments.add(compartment)

            # Initialize CategoryManager and load categories
            # This is done after subsystems are populated so categories can be assigned
            self._category_manager = CategoryManager(self)
            if self.use_subsystem_classification:
                # Load categories from custom or default file
                category_file = self.subsystem_classification_file_path  # May be None
                with redirect_stdout(io.StringIO()):
                    self._category_manager.load_categories(category_file)

            # Save to cache for next time (excludes categories)
            self._save_to_cache()

        except Exception as e:
            print(f"Error: Failed to initialize MetabolicModel: {e}")
            import traceback

            traceback.print_exc()
            # Continue with empty collections for graceful degradation

    def _build_relationship_mappings(self) -> None:
        """Build legacy relationship mappings from entity relationships.

        Note: This method exists for compatibility but the modern approach is to use
        direct entity relationships built by the loaders.
        """
        # Initialize mapping dictionaries
        self._gene_to_subsystems = {}
        self._subsystem_to_genes = {}
        self._reaction_to_metabolites = {}
        self._metabolite_to_reactions = {}

        # Build gene to subsystems mapping
        for gene in self.genes.get_all():
            gene_subsystems = []
            if hasattr(gene, "associated_reactions"):
                for reaction_id in gene.associated_reactions:
                    reaction = self.reactions.get(reaction_id, "primary")
                    if (
                        reaction
                        and hasattr(reaction, "subsystem_id")
                        and reaction.subsystem_id
                    ):
                        subsystem = self.subsystems.get(
                            reaction.subsystem_id, "primary"
                        )
                        if subsystem and subsystem.name not in gene_subsystems:
                            gene_subsystems.append(subsystem.name)
            self._gene_to_subsystems[gene.id] = gene_subsystems

        # Build subsystem to genes mapping
        for subsystem in self.subsystems.get_all():
            if hasattr(subsystem, "gene_ids"):
                self._subsystem_to_genes[subsystem.name] = list(subsystem.gene_ids)

        # Build reaction to metabolites mapping
        for reaction in self.reactions.get_all():
            metabolites = []
            if hasattr(reaction, "reactants"):
                metabolites.extend(reaction.reactants)
            if hasattr(reaction, "products"):
                metabolites.extend(reaction.products)
            self._reaction_to_metabolites[reaction.id] = metabolites

        # Build metabolite to reactions mapping
        for metabolite in self.metabolites.get_all():
            reactions = []
            if hasattr(metabolite, "produced_by"):
                reactions.extend(metabolite.produced_by)
            if hasattr(metabolite, "consumed_by"):
                reactions.extend(metabolite.consumed_by)
            self._metabolite_to_reactions[metabolite.id] = list(set(reactions))

    # ============================================================================
    # Entity Access Methods - Primary Interface
    # ============================================================================

    def _ensure_data_loaded(self, data_type: str) -> None:
        """Ensure specific data is loaded when lazy loading is enabled."""
        if not self.lazy_loading:
            return

        if data_type == "reactions" and len(self.reactions) == 0:
            self._initialize_all_data()
        elif data_type == "metabolites" and len(self.metabolites) == 0:
            self._initialize_all_data()
        elif data_type == "genes" and len(self.genes) == 0:
            self._initialize_all_data()
        elif data_type == "relationships" and self._gene_to_subsystems is None:
            self._build_relationship_mappings()
        elif (
            data_type == "subsystem_metabolite_classification"
            and not self.subsystem_metabolite_classification
        ):
            self._initialize_all_data()

    # ============================================================================
    # Interface Implementation - Core Data Access
    # ============================================================================

    def filter_by_subsystem_and_compartment(self, filters: List[str]) -> None:
        """Filter the model to include only specified subsystems and compartments.

        Args:
            filters: List of subsystems and compartments to include
        """
        # Identify which filters are subsystems and which are compartments
        subsystem_names = set(self.get_subsystems())
        compartment_ids = set(c.id for c in self.compartments.get_all())

        # Separate filters into subsystems and compartments
        filter_subsystems = set(f for f in filters if f in subsystem_names)
        filter_compartments = set(f for f in filters if f in compartment_ids)

        if not filter_subsystems and not filter_compartments:
            return 1

        # Get reactions that match the filters
        valid_reaction_ids = set()

        # Add reactions from matching subsystems
        if filter_subsystems:
            for subsystem in self.subsystems.get_all():
                if subsystem.name in filter_subsystems:
                    if hasattr(subsystem, "reaction_ids"):
                        valid_reaction_ids.update(subsystem.reaction_ids)

        # Add reactions from matching compartments
        if filter_compartments:
            for reaction in self.reactions.get_all():
                if (
                    hasattr(reaction, "compartment_id")
                    and reaction.compartment_id in filter_compartments
                ):
                    valid_reaction_ids.add(reaction.id)

        # Filter all collections based on valid reactions
        # Keep only reactions that match
        filtered_reactions = [
            r for r in self.reactions.get_all() if r.id in valid_reaction_ids
        ]

        # Keep metabolites involved in valid reactions
        valid_metabolite_ids = set()
        for reaction in filtered_reactions:
            if hasattr(reaction, "reactants"):
                valid_metabolite_ids.update(reaction.reactants)
            if hasattr(reaction, "products"):
                valid_metabolite_ids.update(reaction.products)

        filtered_metabolites = [
            m for m in self.metabolites.get_all() if m.id in valid_metabolite_ids
        ]

        # Keep genes involved in valid reactions
        valid_gene_ids = set()
        for reaction in filtered_reactions:
            if hasattr(reaction, "associated_genes"):
                valid_gene_ids.update(reaction.associated_genes)

        filtered_genes = [g for g in self.genes.get_all() if g.id in valid_gene_ids]

        # Keep subsystems and compartments that are in the filter
        if filter_subsystems:
            filtered_subsystems = [
                s for s in self.subsystems.get_all() if s.name in filter_subsystems
            ]
        else:
            # No subsystem filter: keep all subsystems
            filtered_subsystems = list(self.subsystems.get_all())

        if filter_compartments:
            filtered_compartments = [
                c for c in self.compartments.get_all() if c.id in filter_compartments
            ]
        else:
            # No compartment filter: keep all compartments
            filtered_compartments = list(self.compartments.get_all())

        # Clear and repopulate collections with filtered data
        self.reactions = EntityCollection()
        self.metabolites = EntityCollection()
        self.genes = EntityCollection()
        self.subsystems = EntityCollection()
        self.compartments = EntityCollection()

        for reaction in filtered_reactions:
            self.reactions.add(reaction)
        for metabolite in filtered_metabolites:
            self.metabolites.add(metabolite)
        for gene in filtered_genes:
            self.genes.add(gene)
        for subsystem in filtered_subsystems:
            self.subsystems.add(subsystem)
        for compartment in filtered_compartments:
            self.compartments.add(compartment)

        return 0

    def get_general_intermediate_metabolites(
        self, format: str = "metName"
    ) -> List[str]:
        """Get a list of general intermediate metabolite IDs.

        These are metabolites that are commonly considered intermediate across
        many metabolic pathways (e.g., ATP, NADH, H2O, etc.).

        Args:
            format: Format of the returned metabolite IDs

        Returns:
            List of metabolite IDs in the specified format
        """
        intermediate_metabolite_names = [
            "FADH2",
            "H+",
            "H2O",
            "O2",
            "NADH",
            "FAD",
            "NAD+",
            "NADPH",
            "Pi",
            "ADP",
            "ATP",
            "UDP",
            "CoA",
            "NADP+",
            "H2O2",
            "CO2",
        ]

        # If metName format is requested, return the names directly
        if format == "metName":
            return intermediate_metabolite_names

        # Otherwise, find metabolites whose names match and translate to format
        intermediate_metabolites = []

        # Search through all metabolites to find those with matching names
        for metabolite in self.metabolites.get_all():
            if (
                hasattr(metabolite, "name")
                and metabolite.name in intermediate_metabolite_names
            ):
                # Get the metabolite in the requested format
                if format == "primary":
                    intermediate_metabolites.append(metabolite.id)
                elif hasattr(metabolite, format):
                    value = getattr(metabolite, format)
                    if value:
                        intermediate_metabolites.append(value)
                elif format in metabolite.id_mappings:
                    # Check if format exists in ID mappings
                    mapped_id = metabolite.id_mappings.get(format)
                    if mapped_id:
                        intermediate_metabolites.append(mapped_id)
                else:
                    # Fallback to primary ID if format not found
                    intermediate_metabolites.append(metabolite.id)

        return intermediate_metabolites

    def get_available_reaction_formats(self) -> List[str]:
        """Get available reaction name formats.

        Returns:
            List of available format names
        """
        formats = self.reactions.get_available_formats()
        # Map internal format names to legacy format names
        legacy_formats = []
        for fmt in formats:
            if fmt == "primary":
                legacy_formats.append("mar_id")
            else:
                legacy_formats.append(fmt)
        return legacy_formats

    def get_available_gene_formats(self) -> List[str]:
        """Get available gene name formats.

        Returns:
            List of available format names
        """
        # Start from formats declared in the genes collection
        formats = set(self.genes.get_available_formats())

        # Map internal format names to legacy format names
        legacy_formats: List[str] = []
        for fmt in formats:
            if fmt == "primary":
                legacy_formats.append("ensmbl_id")
            else:
                legacy_formats.append(fmt)

        # Include legacy-friendly derived fields when available on data
        try:
            has_symbol = any(getattr(g, "symbol", None) for g in self.genes.get_all())
        except Exception:
            has_symbol = False
        try:
            has_name = any(getattr(g, "name", None) for g in self.genes.get_all())
        except Exception:
            has_name = False

        if has_symbol:
            legacy_formats.append("geneSymbols")
        if has_name:
            legacy_formats.append("geneNames")

        # De-duplicate and provide a stable order
        return sorted(set(legacy_formats))

    def available_metabolite_formats(self) -> List[str]:
        """Get available metabolite name formats.

        Returns:
            List of available format names
        """
        formats = self.metabolites.get_available_formats()
        # Map internal format names to legacy format names and add legacy formats
        legacy_formats = []
        for fmt in formats:
            if fmt == "primary":
                continue  # Don't include primary in legacy list
            else:
                legacy_formats.append(fmt)

        # Add special legacy formats
        legacy_formats.extend(["met_id", "metName"])
        return sorted(list(set(legacy_formats)))

    def get_reactions(self, format: str = "primary") -> List[str]:
        """Get a list of reaction IDs in the specified format.

        Args:
            format: Format for the reaction IDs (default: "primary" which is MAR IDs)

        Returns:
            List of reaction IDs in the requested format
        """
        reaction_ids = []
        for reaction in self.reactions.get_all():
            if format == "primary" or format == "mar_id":
                # MAR IDs are the primary format
                reaction_id = reaction.id
            else:
                # Try to translate to other formats
                reaction_id = self.reactions.translate_id(
                    reaction.id, "primary", format
                )
                # Fall back to primary ID if translation fails
                if reaction_id is None:
                    reaction_id = reaction.id
            reaction_ids.append(reaction_id)
        return reaction_ids

    def get_genes(self, format: str = "primary") -> List[str]:
        """Get a list of gene IDs in the specified format.

        Args:
            format: Format for the gene IDs (default: "primary" which is ENSG IDs)

        Returns:
            List of gene IDs in the requested format
        """
        gene_ids = []
        for gene in self.genes.get_all():
            if format == "primary" or format == "ensmbl_id":
                # ENSG IDs are the primary format
                gene_id = gene.id
            elif format == "geneSymbols":
                # Get gene symbol from the symbol attribute
                gene_id = getattr(gene, "symbol", "") or gene.id
            elif format == "geneNames":
                # Get gene name from the name attribute
                gene_id = getattr(gene, "name", "") or gene.id
            else:
                # Try to translate to other formats
                gene_id = self.genes.translate_id(gene.id, "primary", format)
                # Fall back to primary ID if translation fails
                if gene_id is None:
                    gene_id = gene.id

            # Only add non-empty gene IDs
            if gene_id:
                gene_ids.append(gene_id)
        return gene_ids

    def get_subsystems(self, format: str = "id") -> List[str]:
        """Get a list of all subsystems in the model.

        Returns:
            List of subsystem ids or names based on the specified format
        """
        if format == "name":
            return [subsystem.name for subsystem in self.subsystems.get_all()]
        else:
            return [subsystem.id for subsystem in self.subsystems.get_all()]

    def get_subsystem_name(self, subsystem_id: str) -> Optional[str]:
        """Get the human-readable name for a subsystem by its ID.

        Args:
            subsystem_id: The subsystem ID to look up

        Returns:
            The subsystem name if found, None otherwise
        """
        subsystem = self.subsystems.get(subsystem_id)
        if subsystem:
            return subsystem.name
        return None

    def get_metabolites(self) -> List[str]:
        """Get a list of metabolite IDs in the specified format.

        Args:
            format: Format for the metabolite IDs

        Returns:
            List of metabolite IDs in the requested format
        """
        metabolite_ids = []
        for metabolite in self.metabolites.get_all():
            metabolite_ids.append(metabolite.id)

        return metabolite_ids

    def translate_gene(
        self,
        gene: str,
        input_format: str = "primary",
        output_format: str = "geneSymbols",
    ) -> str:
        """Translate a gene ID from one format to another.

        Args:
            gene: Gene ID to translate
            input_format: Format of the input gene ID (default: "primary")
            output_format: Desired format for the output gene ID

        Returns:
            Translated gene ID or original ID if translation fails
        """
        # Handle legacy 'ensmbl_id' format mapping
        if input_format == "ensmbl_id":
            input_format = "primary"
        if output_format == "ensmbl_id":
            output_format = "primary"

        # Special handling for geneSymbols - it's stored in the symbol attribute
        if output_format == "geneSymbols":
            gene_entity = self.genes.get(gene, input_format)
            if gene_entity and hasattr(gene_entity, "symbol"):
                return gene_entity.symbol or gene
            return gene

        # Special handling for geneNames - it's stored in the name attribute
        if output_format == "geneNames":
            gene_entity = self.genes.get(gene, input_format)
            if gene_entity and hasattr(gene_entity, "name"):
                return gene_entity.name or gene
            return gene

        # For reverse translation from symbol/name to primary
        if input_format == "geneSymbols":
            # Find gene by symbol
            for gene_entity in self.genes.get_all():
                if hasattr(gene_entity, "symbol") and gene_entity.symbol == gene:
                    if output_format == "primary":
                        return gene_entity.id
                    else:
                        # Translate from primary to target format
                        return (
                            self.genes.translate_id(
                                gene_entity.id, "primary", output_format
                            )
                            or gene
                        )
            return gene

        if input_format == "geneNames":
            # Find gene by name
            for gene_entity in self.genes.get_all():
                if hasattr(gene_entity, "name") and gene_entity.name == gene:
                    if output_format == "primary":
                        return gene_entity.id
                    else:
                        # Translate from primary to target format
                        return (
                            self.genes.translate_id(
                                gene_entity.id, "primary", output_format
                            )
                            or gene
                        )
            return gene

        # Use standard translation for other formats
        translated_id = self.genes.translate_id(gene, input_format, output_format)
        return translated_id if translated_id else gene

    def translate_reaction(
        self,
        reaction: str,
        input_format: str = "primary",
        output_format: str = "rxnBiGGID",
    ) -> str:
        """Translate a reaction ID from one format to another.

        Args:
            reaction: Reaction ID to translate
            input_format: Format of the input reaction ID (default: "primary")
            output_format: Desired format for the output reaction ID

        Returns:
            Translated reaction ID or original ID if translation fails
        """
        # Handle legacy 'mar_id' format mapping
        if input_format == "mar_id":
            input_format = "primary"
        if output_format == "mar_id":
            output_format = "primary"

        translated_id = self.reactions.translate_id(
            reaction, input_format, output_format
        )
        return translated_id if translated_id else reaction

    def translate_metabolite(
        self,
        metabolite: str,
        input_format: str = "primary",
        output_format: str = "metName",
    ) -> str:
        """Translate a metabolite ID from one format to another.

        Args:
            metabolite: Metabolite ID to translate
            input_format: Format of the input metabolite ID
            output_format: Desired format for the output metabolite ID

        Returns:
            Translated metabolite ID or original ID if translation fails
        """
        # Special handling for met_id format (maps to metHMR2ID)
        # Use standard translation for other formats
        translated_id = self.metabolites.translate_id(
            metabolite, input_format, output_format
        )
        return translated_id if translated_id else metabolite

    # ============================================================================
    # Interface Implementation - Relationship Methods
    # ============================================================================

    def get_reactions_for_subsystem(self, subsystem: str) -> List[str]:
        """Get all reactions for a specific subsystem.

        Args:
            subsystem: Name or ID of the subsystem

        Returns:
            List of reaction IDs in the subsystem, sorted

        Note:
            ``Subsystem.reaction_ids`` is a set, so an unsorted ``list()`` of it
            orders differently in every process. That order reaches the automatic
            subsystem layout as the order reactions are placed in, which made the
            generated graph resources irreproducible (D-42). Sorting here is what
            makes a regenerated layout comparable to the one it replaces.
        """
        # Get the subsystem entity by name (try both original and cleaned names)
        subsystem_entity = None
        for sub in self.subsystems.get_all():
            if sub.name == subsystem or sub.id == subsystem:
                subsystem_entity = sub
                break

        if not subsystem_entity:
            return []

        # Use direct relationships from the subsystem entity
        if hasattr(subsystem_entity, "reaction_ids") and subsystem_entity.reaction_ids:
            return sorted(subsystem_entity.reaction_ids)

        return []

    def get_genes_for_subsystem(self, subsystem: str) -> List[str]:
        """Get all genes for a specific subsystem.

        Args:
            subsystem: Name or ID of the subsystem

        Returns:
            List of gene IDs for the subsystem, sorted

        Note:
            Sorted for the same reason as ``get_reactions_for_subsystem``: this is
            the gene universe WP6 reports as an auditable ORA input, and a set's
            iteration order makes that audit unreproducible across processes.
        """
        # Get the subsystem entity by name (try both original and cleaned names)
        subsystem_entity = None
        for sub in self.subsystems.get_all():
            if sub.name == subsystem or sub.id == subsystem:
                subsystem_entity = sub
                break

        if not subsystem_entity:
            return []

        # Use direct relationships from the subsystem entity
        if hasattr(subsystem_entity, "gene_ids") and subsystem_entity.gene_ids:
            return sorted(subsystem_entity.gene_ids)

        return []

    def get_reactions_for_gene(self, gene_id: str) -> List[str]:
        """Get all reactions that involve a specific gene.

        Args:
            gene_id: Gene ID to search for

        Returns:
            List of reaction IDs involving the gene
        """
        # Get the gene entity
        gene = self.genes.get(gene_id)
        if not gene:
            return []

        # Use direct relationships from the gene entity
        if hasattr(gene, "associated_reactions") and gene.associated_reactions:
            return list(gene.associated_reactions)

        return []

    def get_metabolites_for_reaction(self, reaction_id: str) -> Dict[str, List[str]]:
        """Get all metabolites involved in a specific reaction.

        Args:
            reaction_id: Reaction ID to search for

        Returns:
            Dict with 'reactants' and 'products' lists of metabolite IDs
        """
        # Get the reaction entity
        reaction = self.reactions.get(reaction_id)
        if not reaction:
            return {"reactants": [], "products": []}

        result = {"reactants": [], "products": []}

        if hasattr(reaction, "reactants") and reaction.reactants:
            result["reactants"] = list(reaction.reactants)
        if hasattr(reaction, "products") and reaction.products:
            result["products"] = list(reaction.products)

        return result

    def is_reaction_reversible(
        self, reaction_id: str, input_format: str = "primary"
    ) -> bool:
        """Check if a reaction is reversible.

        Args:
            reaction_id: Reaction ID to check
            input_format: Format of the input reaction ID (default: "primary")

        Returns:
            True if the reaction is reversible, False otherwise.
            Returns False if reaction is not found in the model.
        """
        # Handle legacy 'mar_id' format mapping
        if input_format == "mar_id":
            input_format = "primary"

        # Try to get the reaction directly if format is primary
        if input_format == "primary":
            reaction = self.reactions.get(reaction_id)
        else:
            # Translate to primary format first
            primary_id = self.reactions.translate_id(
                reaction_id, input_format, "primary"
            )
            if primary_id:
                reaction = self.reactions.get(primary_id)
            else:
                reaction = None

        if reaction and hasattr(reaction, "is_reversible"):
            return reaction.is_reversible

        return False

    def get_reactions_for_metabolite(self, metabolite_id: str) -> Dict[str, List[str]]:
        """Get all reactions involving a specific metabolite.

        Args:
            metabolite_id: Metabolite ID to search for

        Returns:
            Dict with 'as_reactant' and 'as_product' lists of reaction IDs
        """
        # Get the metabolite entity
        metabolite = self.metabolites.get(metabolite_id)
        if not metabolite:
            return {"as_reactant": [], "as_product": []}

        result = {"as_reactant": [], "as_product": []}

        # Use direct relationships from the metabolite entity
        if hasattr(metabolite, "consumed_by") and metabolite.consumed_by:
            result["as_reactant"] = list(metabolite.consumed_by)
        if hasattr(metabolite, "produced_by") and metabolite.produced_by:
            result["as_product"] = list(metabolite.produced_by)

        return result

    # ============================================================================
    # Subsystem Category Methods
    # ============================================================================

    def get_subsystems_by_category(self) -> Dict[str, List[str]]:
        """Return subsystems grouped by category.

        Returns:
            Dict mapping category names to lists of subsystem IDs
        """
        self._ensure_data_loaded("subsystems")
        categories = {}
        for subsystem in self.subsystems.get_all():
            category = subsystem.category
            if category not in categories:
                categories[category] = []
            categories[category].append(subsystem.id)
        return categories

    def get_subsystem_categories(self) -> List[str]:
        """Return list of all subsystem categories.

        Returns:
            List of category names
        """
        return list(self.get_subsystems_by_category().keys())

    def get_subsystem_category(self, subsystem_id_or_name: str) -> str:
        """Get the category for a specific subsystem.

        Args:
            subsystem_id_or_name: Subsystem ID or display name to get category for

        Returns:
            Category name, or "Uncategorized" if not found
        """
        self._ensure_data_loaded("subsystems")

        # First try to get by ID (exact lookup)
        subsystem = self.subsystems.get(subsystem_id_or_name)
        if subsystem:
            return subsystem.category

        # If not found by ID, try to find by display name
        subsystem = self.find_subsystem_by_name(subsystem_id_or_name)
        if subsystem:
            return subsystem.category

        return "Uncategorized"

    def find_subsystem_by_name(self, subsystem_name: str) -> Optional["Subsystem"]:
        """Find a subsystem by its display name.

        Args:
            subsystem_name: Display name of the subsystem

        Returns:
            Subsystem entity if found, None otherwise
        """
        self._ensure_data_loaded("subsystems")
        for subsystem in self.subsystems.get_all():
            if hasattr(subsystem, "name") and subsystem.name == subsystem_name:
                return subsystem
        return None

    # ============================================================================
    # CategoryManager Delegation Methods
    # ============================================================================

    @property
    def category_manager(self) -> CategoryManager:
        """Get the CategoryManager instance.

        Returns:
            CategoryManager instance for this model
        """
        if self._category_manager is None:
            self._category_manager = CategoryManager(self)
        return self._category_manager

    def load_categories(self, file_path: Optional[str] = None) -> Dict[str, any]:
        """Load or reload subsystem categories from a file.

        This method allows fast category reloading without recreating the model.
        All subsystems will have their categories updated based on the file.
        Subsystems not in the file will be set to "Uncategorized".

        Args:
            file_path: Path to category file. If None, uses default file.

        Returns:
            Dict with loading statistics:
                - num_categories: Number of unique categories loaded
                - num_subsystems_updated: Number of subsystems that got a category
                - num_uncategorized: Number of subsystems set to "Uncategorized"
                - categories: List of category names in file order
        """
        return self.category_manager.load_categories(file_path)

    def get_category_color(self, category_name: str) -> str:
        """Get the color for a category.

        Colors are assigned based on the category's position in the file.

        Args:
            category_name: Name of the category

        Returns:
            Hex color string (e.g., "#E74C3C")
        """
        return self.category_manager.get_category_color(category_name)

    def is_custom_category_file_loaded(self) -> bool:
        """Check if a custom (non-default) category file is loaded.

        Returns:
            True if a custom file is loaded, False if using default
        """
        return self.category_manager.is_custom_file_loaded()

    def get_custom_category_file_path(self) -> Optional[str]:
        """Get path of custom category file if one was loaded.

        Returns:
            Path to custom file, or None if using default
        """
        return self.category_manager.get_custom_file_path()

    def get_categories(self) -> List[str]:
        """Get list of all category names in their display order.

        Returns:
            List of category names in the order they appear in the file
        """
        return self.category_manager.get_categories()

    @staticmethod
    def _get_first_mapping_value(value: Optional[str]) -> str:
        """Return the first non-empty identifier from a possibly multi-valued field."""
        if value is None:
            return ""

        for candidate in str(value).split(";"):
            candidate = candidate.strip()
            if candidate:
                return candidate

        return ""

    def _resolve_reaction_entity(
        self, reaction_id: str
    ) -> Tuple[Optional[Reaction], str]:
        """Resolve a reaction, falling back from `_back` pseudo-IDs to the base reaction."""
        reaction = self.reactions.get(reaction_id)
        if reaction:
            return reaction, reaction_id

        if reaction_id and reaction_id.endswith("_back"):
            base_reaction_id = reaction_id[:-5]
            reaction = self.reactions.get(base_reaction_id)
            if reaction:
                return reaction, base_reaction_id

        return None, reaction_id

    def _extract_reaction_kegg_id(self, reaction: Reaction) -> str:
        """Extract the preferred KEGG reaction identifier for a reaction."""
        kegg_id = self._get_first_mapping_value(
            reaction.id_mappings.get("rxnKEGGID", "")
        )
        if kegg_id:
            return kegg_id

        if reaction.miriam:
            for entry in str(reaction.miriam).split(";"):
                entry = entry.strip()
                marker = "kegg.reaction/"
                if marker in entry:
                    return entry.split(marker, 1)[1].strip()

        return ""

    def get_gene_reference_data(self, gene_id: str) -> Dict[str, str]:
        """Get display and external reference data for a gene."""
        gene = self.genes.get(gene_id)
        if not gene:
            return {
                "raw_id": gene_id,
                "primary_id": gene_id,
                "display_label": gene_id,
                "external_url": "",
                "external_source": "",
                "external_id": "",
            }

        symbol = getattr(gene, "symbol", "") or ""
        display_label = symbol or gene.id
        if symbol:
            external_url = (
                f"https://www.genecards.org/cgi-bin/carddisp.pl?gene={quote(symbol)}"
            )
            external_source = "GeneCards"
            external_id = symbol
        else:
            external_url = (
                f"https://www.ensembl.org/Homo_sapiens/Gene/Summary?g={quote(gene.id)}"
            )
            external_source = "Ensembl"
            external_id = gene.id

        return {
            "raw_id": gene.id,
            "primary_id": gene.id,
            "display_label": display_label,
            "external_url": external_url,
            "external_source": external_source,
            "external_id": external_id,
        }

    def get_reaction_reference_data(self, reaction_id: str) -> Dict[str, str]:
        """Get display and external reference data for a reaction."""
        reaction, _ = self._resolve_reaction_entity(reaction_id)
        if not reaction:
            return {
                "raw_id": reaction_id,
                "primary_id": reaction_id,
                "display_label": reaction_id,
                "short_name": reaction_id,
                "external_url": "",
                "external_source": "",
                "external_id": "",
            }

        raw_id = reaction_id or reaction.id
        external_id = reaction.id
        external_url = (
            "https://metabolicatlas.org/explore/Human-GEM/gem-browser/reaction/"
            f"{quote(external_id)}"
        )

        return {
            "raw_id": raw_id,
            "primary_id": reaction.id,
            "display_label": raw_id,
            # What a figure labels the reaction with (reaction_names).
            "short_name": short_reaction_name(reaction.name, reaction.equation) or raw_id,
            "external_url": external_url,
            "external_source": "Metabolic Atlas",
            "external_id": external_id,
        }

    def get_subsystem_reference_data(
        self, subsystem_id_or_name: str
    ) -> Dict[str, str]:
        """Get display and external reference data for a subsystem.

        The external URL is the Metabolic Atlas Human-GEM gem-browser subsystem
        page. Its slug is the model's canonical subsystem ``id``, which is the
        display name run through ``_clean_subsystem_name`` at load time. That
        cleaning is the Metabolic Atlas subsystem slug (A-17: deterministically
        derived, validated against the full Human-GEM subsystem list with zero
        collisions and only URL-safe characters), so no maintained
        name-to-slug map is required.

        The argument accepts either the canonical subsystem id or its display
        name, mirroring :meth:`get_subsystem_category`.
        """
        self._ensure_data_loaded("subsystems")

        subsystem = self.subsystems.get(subsystem_id_or_name)
        if not subsystem:
            subsystem = self.find_subsystem_by_name(subsystem_id_or_name)

        if not subsystem:
            return {
                "raw_id": subsystem_id_or_name,
                "primary_id": subsystem_id_or_name,
                "display_label": subsystem_id_or_name,
                "external_url": "",
                "external_source": "",
                "external_id": "",
            }

        slug = subsystem.id
        # Metabolic Atlas has no page for the parts of the divided transport
        # (loaders.transport_part): they link to its Transport reactions page.
        if slug in TRANSPORT_PART_IDS:
            slug = _clean_subsystem_name(TRANSPORT_SUBSYSTEM)
        external_url = (
            "https://metabolicatlas.org/explore/Human-GEM/gem-browser/subsystem/"
            f"{quote(slug)}"
        )

        return {
            "raw_id": subsystem.id,
            "primary_id": subsystem.id,
            "display_label": subsystem.name,
            "external_url": external_url,
            "external_source": "Metabolic Atlas",
            "external_id": slug,
        }

    # ============================================================================
    # Interface Implementation - Metadata Methods
    # ============================================================================

    def get_gene_metadata(self, gene_id: str = None) -> Dict[str, any]:
        """Get metadata for genes.

        Args:
            gene_id: Specific gene ID to get metadata for (None for all genes)

        Returns:
            For single gene: flat metadata dictionary
            For all genes: nested dictionary with gene IDs as keys
        """
        if gene_id is not None:
            # Get metadata for a specific gene - return flat dictionary
            # Handle legacy format mapping
            gene = self.genes.get(gene_id)

            if gene:
                gene_data = {
                    **self.get_gene_reference_data(gene.id),
                    "ensmble_id": gene.id,  # Match legacy typo
                    "geneSymbols": getattr(gene, "symbol", ""),
                    "geneNames": getattr(gene, "name", ""),
                    "geneAliases": getattr(gene, "aliases", ""),
                    "geneENSTID": gene.id_mappings.get("geneENSTID", ""),
                    "geneENSPID": gene.id_mappings.get("geneENSPID", ""),
                    "geneUniProtID": gene.id_mappings.get("geneUniProtID", ""),
                    "geneEntrezID": gene.id_mappings.get("geneEntrezID", ""),
                    "id_mappings": dict(gene.id_mappings),
                    "compartments": getattr(gene, "compartments", ""),
                    "compDataSource": gene.id_mappings.get("compDataSource", ""),
                    "description": getattr(gene, "description", ""),
                    "organism": getattr(gene, "organism", ""),
                    "chromosome": getattr(gene, "chromosome", ""),
                    "start_position": str(getattr(gene, "start_position", "")),
                    "end_position": str(getattr(gene, "end_position", "")),
                    "strand": getattr(gene, "strand", ""),
                }

                # Add subsystems information
                if hasattr(gene, "associated_reactions") and gene.associated_reactions:
                    subsystems = set()
                    for reaction_id in gene.associated_reactions:
                        reaction = self.reactions.get(reaction_id, "primary")
                        if (
                            reaction
                            and hasattr(reaction, "subsystem_id")
                            and reaction.subsystem_id
                        ):
                            subsystem = self.subsystems.get(
                                reaction.subsystem_id, "primary"
                            )
                            if subsystem:
                                subsystems.add(subsystem.name)
                    gene_data["subsystems"] = list(subsystems)
                else:
                    gene_data["subsystems"] = []

                return gene_data
            return {}
        else:
            # Get metadata for all genes - return nested dictionary
            metadata = {}
            for gene in self.genes.get_all():
                # Handle legacy format mapping and special formats
                gene_display_id = gene.id  # Primary format is already ENSG

                # Create gene data for this gene
                gene_data = {
                    **self.get_gene_reference_data(gene.id),
                    "ensmble_id": gene.id,  # Match legacy typo
                    "geneSymbols": getattr(gene, "symbol", ""),
                    "geneNames": getattr(gene, "name", ""),
                    "geneAliases": getattr(gene, "aliases", ""),
                    "geneENSTID": gene.id_mappings.get("geneENSTID", ""),
                    "geneENSPID": gene.id_mappings.get("geneENSPID", ""),
                    "geneUniProtID": gene.id_mappings.get("geneUniProtID", ""),
                    "geneEntrezID": gene.id_mappings.get("geneEntrezID", ""),
                    "id_mappings": dict(gene.id_mappings),
                    "compartments": getattr(gene, "compartments", ""),
                    "compDataSource": gene.id_mappings.get("compDataSource", ""),
                    "description": getattr(gene, "description", ""),
                    "organism": getattr(gene, "organism", ""),
                    "chromosome": getattr(gene, "chromosome", ""),
                    "start_position": str(getattr(gene, "start_position", "")),
                    "end_position": str(getattr(gene, "end_position", "")),
                    "strand": getattr(gene, "strand", ""),
                }

                # Add subsystems information
                if hasattr(gene, "associated_reactions") and gene.associated_reactions:
                    subsystems = set()
                    for reaction_id in gene.associated_reactions:
                        reaction = self.reactions.get(reaction_id, "primary")
                        if (
                            reaction
                            and hasattr(reaction, "subsystem_id")
                            and reaction.subsystem_id
                        ):
                            subsystem = self.subsystems.get(
                                reaction.subsystem_id, "primary"
                            )
                            if subsystem:
                                subsystems.add(subsystem.name)
                    gene_data["subsystems"] = list(subsystems)
                else:
                    gene_data["subsystems"] = []

                metadata[gene_display_id] = gene_data

            return metadata

    def get_reaction_metadata(self, reaction_id: str = None) -> Dict[str, any]:
        """Get metadata for reactions.

        Args:
            reaction_id: Specific reaction ID to get metadata for
                (None for all reactions)

        Returns:
            For single reaction: flat metadata dictionary
            For all reactions: nested dictionary with reaction IDs as keys
        """
        if reaction_id is not None:
            # Get metadata for a specific reaction - return flat dictionary
            # Handle legacy format mapping
            reaction, _ = self._resolve_reaction_entity(reaction_id)

            if reaction:
                reaction_data = {
                    **self.get_reaction_reference_data(reaction_id),
                    "name": getattr(reaction, "name", ""),
                    "equation": getattr(reaction, "equation", ""),
                    "formula": getattr(reaction, "formula", ""),
                    "gpr": getattr(reaction, "gene_rule_symbol", ""),
                    "gpr_ensmbl": getattr(reaction, "gene_rule", ""),
                    "compartment": getattr(reaction, "compartment_id", ""),
                    "is_reversible": getattr(reaction, "is_reversible", False),
                    "kegg_id": self._extract_reaction_kegg_id(reaction),
                    "id_mappings": dict(reaction.id_mappings),
                }

                # Get subsystem information
                if hasattr(reaction, "subsystem_id") and reaction.subsystem_id:
                    subsystem = self.subsystems.get(reaction.subsystem_id, "primary")
                    if subsystem:
                        reaction_data["subsystems"] = [subsystem.name]
                        reaction_data["subsystem_category"] = getattr(
                            subsystem, "category", None
                        )
                    else:
                        reaction_data["subsystems"] = []
                        reaction_data["subsystem_category"] = None
                else:
                    reaction_data["subsystems"] = []
                    reaction_data["subsystem_category"] = None

                return reaction_data
            return {}
        else:
            # Get metadata for all reactions - return nested dictionary
            metadata = {}
            for reaction in self.reactions.get_all():
                reaction_display_id = reaction.id  # Primary is already MAR

                if reaction_display_id:
                    reaction_data = {
                        **self.get_reaction_reference_data(reaction.id),
                        "name": getattr(reaction, "name", ""),
                        "equation": getattr(reaction, "equation", ""),
                        "formula": getattr(reaction, "formula", ""),
                        "gpr_ensmbl": getattr(reaction, "gene_rule", ""),
                        "gpr": getattr(reaction, "gene_rule_symbol", ""),
                        "compartment": getattr(reaction, "compartment_id", ""),
                        "is_reversible": getattr(reaction, "is_reversible", False),
                        "kegg_id": self._extract_reaction_kegg_id(reaction),
                        "id_mappings": dict(reaction.id_mappings),
                    }

                    # Get subsystem information
                    if hasattr(reaction, "subsystem_id") and reaction.subsystem_id:
                        subsystem = self.subsystems.get(
                            reaction.subsystem_id, "primary"
                        )
                        if subsystem:
                            reaction_data["subsystems"] = [subsystem.name]
                            reaction_data["subsystem_category"] = getattr(
                                subsystem, "category", None
                            )
                        else:
                            reaction_data["subsystems"] = []
                            reaction_data["subsystem_category"] = None
                    else:
                        reaction_data["subsystems"] = []
                        reaction_data["subsystem_category"] = None

                    metadata[reaction_display_id] = reaction_data

            return metadata

    def _reaction_subsystem_names(self) -> Dict[str, str]:
        """Reaction ID -> subsystem name, built once per model.

        The fan-out below asks this question once per incident reaction, and a
        hub metabolite is incident to thousands of them. Resolving each through
        ``self.reactions.get`` and then ``self.subsystems.get`` pays two indexed
        entity lookups every time; this pays them once for the whole model.
        """
        cached = self._cache.get("reaction_subsystem_names")
        if cached is not None:
            return cached

        names: Dict[str, str] = {}
        for reaction in self.reactions.get_all():
            subsystem_id = getattr(reaction, "subsystem_id", None)
            if not subsystem_id:
                continue
            subsystem = self.subsystems.get(subsystem_id, "primary")
            name = getattr(subsystem, "name", None) if subsystem else None
            if name:
                names[reaction.id] = name

        self._cache["reaction_subsystem_names"] = names
        return names

    def _metabolite_subsystem_fanout(
        self, metabolite: any
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """Subsystems producing and consuming one metabolite, memoized.

        This is a pure function of model structure, but it is O(degree) and the
        degree distribution is brutally skewed: cytosolic water is incident to
        3,352 reactions in Human1. Recomputing it per call made
        ``get_metabolite_data`` 36x more expensive per call than its gene
        sibling (48.9 ms versus 1.2 ms over comparable call counts) and drove
        225,294 entity lookups while drawing a single 348-node subsystem graph,
        because hover asks for the same metabolite up to 36 times per render.

        Memoizing is safe precisely because AM8 made the model replace-only: a
        filter, reload, or category change installs a *fresh* model object
        through ``AnalysisWorkspace.apply_model_transition``, which clears this
        cache (``managers/workspace.py``). Nothing mutates an installed model in
        place -- ``test_installed_model_immutability.py`` forbids it -- so a structural memo
        cannot go stale under the caller.
        """
        cache = self._cache.setdefault("metabolite_subsystem_fanout", {})
        key = getattr(metabolite, "id", None)
        if key is not None and key in cache:
            return cache[key]

        subsystem_names = self._reaction_subsystem_names()
        produced_by_subsystems = set()
        consumed_by_subsystems = set()
        try:
            for r_id in getattr(metabolite, "produced_by", []) or []:
                name = subsystem_names.get(r_id)
                if name:
                    produced_by_subsystems.add(name)

            for r_id in getattr(metabolite, "consumed_by", []) or []:
                name = subsystem_names.get(r_id)
                if name:
                    consumed_by_subsystems.add(name)
        except Exception:
            # Be robust to partial data, as the per-call implementation was.
            pass

        # Tuples, not sets: the value is shared with every caller, and a caller
        # that mutated it would silently repaint every later hover.
        result = (
            tuple(sorted(produced_by_subsystems)),
            tuple(sorted(consumed_by_subsystems)),
        )
        if key is not None:
            cache[key] = result
        return result

    def get_metabolite_metadata(self, metabolite_id: str = None) -> Dict[str, any]:
        """Get metadata for metabolites.

        Args:
            metabolite_id: Specific metabolite ID to get metadata for
                (None for all metabolites)

        Returns:
            Metadata dictionary for the metabolite(s)
        """
        # For single metabolite requests, return a flat dictionary (robust API)
        if metabolite_id is not None:
            # Try to resolve the provided ID across all known formats
            metabolite = self.metabolites.get(metabolite_id, "primary")
            if metabolite is None:
                for fmt in self.metabolites.get_available_formats():
                    if fmt == "primary":
                        continue
                    metabolite = self.metabolites.get(metabolite_id, fmt)
                    if metabolite is not None:
                        break

            if not metabolite:
                return {}

            # Derive compartment string from available compartment IDs
            compartment_str = ""
            try:
                comp_ids = getattr(metabolite, "compartment_ids", set()) or set()
                if comp_ids:
                    compartment_str = ", ".join(sorted(list(comp_ids)))
            except Exception:
                compartment_str = ""

            produced_by_subsystems, consumed_by_subsystems = (
                self._metabolite_subsystem_fanout(metabolite)
            )

            return {
                "primary_id": metabolite.id,
                "name": getattr(metabolite, "name", ""),
                "formula": getattr(metabolite, "formula", ""),
                "charge": str(getattr(metabolite, "charge", "")),
                "compartment": compartment_str,
                "inchi": getattr(metabolite, "inchi", ""),
                "smiles": getattr(metabolite, "smiles", ""),
                "molecular_weight": str(getattr(metabolite, "molecular_weight", "")),
                "produced_by": list(getattr(metabolite, "produced_by", [])),
                "consumed_by": list(getattr(metabolite, "consumed_by", [])),
                "produced_by_subsystems": sorted(produced_by_subsystems),
                "consumed_by_subsystems": sorted(consumed_by_subsystems),
            }
        else:
            # Get metadata for all metabolites
            metadata = {}
            for metabolite in self.metabolites.get_all():
                produced, consumed = self._metabolite_subsystem_fanout(metabolite)
                subsystems_set = set(produced) | set(consumed)

                metadata[metabolite.id] = {
                    "primary_id": metabolite.id,
                    "name": getattr(metabolite, "name", ""),
                    "formula": getattr(metabolite, "formula", ""),
                    "charge": str(getattr(metabolite, "charge", "")),
                    "compartment": ", ".join(
                        sorted(
                            list(getattr(metabolite, "compartment_ids", set()) or set())
                        )
                    ),
                    "inchi": getattr(metabolite, "inchi", ""),
                    "smiles": getattr(metabolite, "smiles", ""),
                    "molecular_weight": str(
                        getattr(metabolite, "molecular_weight", "")
                    ),
                    "produced_by": list(getattr(metabolite, "produced_by", [])),
                    "consumed_by": list(getattr(metabolite, "consumed_by", [])),
                    "subsystems": sorted(subsystems_set),
                }
            return metadata

            return metadata

    def get_subsystem_metadata(
        self, subsystem_id: str = None
    ) -> Dict[str, Dict[str, str]]:
        """Get metadata for subsystems.

        Args:
            subsystem_id: Specific subsystem ID to get metadata for
                (None for all subsystems)

        Returns:
            Metadata dictionary for the subsystem(s)
        """
        metadata = {}

        if subsystem_id is not None:
            # Get metadata for a specific subsystem
            subsystem = self.subsystems.get(subsystem_id, "primary")
            if subsystem:
                metadata[subsystem_id] = {
                    "primary_id": subsystem.id,
                    "name": getattr(subsystem, "name", ""),
                    "description": getattr(subsystem, "description", ""),
                    "category": getattr(subsystem, "category", ""),
                }
        else:
            # Get metadata for all subsystems
            for subsystem in self.subsystems.get_all():
                metadata[subsystem.name] = {
                    "primary_id": subsystem.id,
                    "name": getattr(subsystem, "name", ""),
                    "description": getattr(subsystem, "description", ""),
                    "category": getattr(subsystem, "category", ""),
                }

        return metadata

    # ============================================================================
    # Entity Management Methods
    # ============================================================================

    def add_reaction(self, reaction: Reaction) -> None:
        """Add a reaction to the model."""
        self.reactions.add(reaction)

    def add_metabolite(self, metabolite: Metabolite) -> None:
        """Add a metabolite to the model."""
        self.metabolites.add(metabolite)

    def add_gene(self, gene: Gene) -> None:
        """Add a gene to the model."""
        self.genes.add(gene)

    def add_subsystem(self, subsystem: Subsystem) -> None:
        """Add a subsystem to the model."""
        self.subsystems.add(subsystem)

    def add_compartment(self, compartment: Compartment) -> None:
        """Add a compartment to the model."""
        self.compartments.add(compartment)

    def get_reaction(self, reaction_id: str) -> Optional[Reaction]:
        """Get a reaction by ID in any supported format."""
        return self.reactions.get(reaction_id)

    def get_metabolite(self, metabolite_id: str) -> Optional[Metabolite]:
        """Get a metabolite by ID in any supported format."""
        return self.metabolites.get(metabolite_id)

    def get_gene(self, gene_id: str) -> Optional[Gene]:
        """Get a gene by ID in any supported format."""
        return self.genes.get(gene_id)

    def get_subsystem(self, subsystem_id: str) -> Optional[Subsystem]:
        """Get a subsystem by ID in any supported format."""
        return self.subsystems.get(subsystem_id)

    def get_compartment(self, compartment_id: str) -> Optional[Compartment]:
        """Get a compartment by ID in any supported format."""
        return self.compartments.get(compartment_id)

    def get_entity_count(self, entity_type: str) -> int:
        """Get the count of entities of a specific type."""
        entity_collections = {
            "reactions": self.reactions,
            "metabolites": self.metabolites,
            "genes": self.genes,
            "subsystems": self.subsystems,
            "compartments": self.compartments,
        }

        collection = entity_collections.get(entity_type)
        if collection is None:
            raise ValueError(f"Unknown entity type: {entity_type}")

        return collection.count()

    def get_available_id_formats(self, entity_type: str) -> List[str]:
        """Get available ID formats for a specific entity type."""
        entity_collections = {
            "reactions": self.reactions,
            "metabolites": self.metabolites,
            "genes": self.genes,
            "subsystems": self.subsystems,
            "compartments": self.compartments,
        }

        collection = entity_collections.get(entity_type)
        if collection is None:
            raise ValueError(f"Unknown entity type: {entity_type}")

        return collection.get_available_formats()

    # ============================================================================
    # Cache Management Methods
    # ============================================================================

    def _get_cache_dir(self) -> str:
        """Get the cache directory path, creating it if it doesn't exist."""
        return self.resource_registry.get_runtime_cache_dir("model")

    def _get_source_files(self) -> List[str]:
        """Get list of all source files used for model loading."""
        return list(self.resource_registry.get_reference_model_file_paths().values())

    def _get_source_files_hash(self) -> str:
        """Generate hash based on source file modification times and sizes.

        :data:`MODEL_BUILD_REVISION` names how the model is built from those
        files, so a cache built another way is never read back.
        """
        hash_input = [MODEL_BUILD_REVISION]

        for file_path in self._get_source_files():
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                # Include file path, modification time, and size
                hash_input.append(f"{file_path}:{stat.st_mtime}:{stat.st_size}")
            else:
                # If file doesn't exist, include that info too
                hash_input.append(f"{file_path}:missing")

        # Create hash from all file info
        hash_string = "|".join(hash_input)
        return hashlib.md5(hash_string.encode()).hexdigest()

    def _get_cache_path(self) -> str:
        """Get the full path to the cache file."""
        source_hash = self._get_source_files_hash()
        cache_filename = f"model_cache_{source_hash}.pkl"
        return self.resource_registry.get_model_cache_paths(cache_filename)[0]

    def _get_cache_candidates(self) -> List[str]:
        """Get all compatible filesystem locations for the current cache file."""
        source_hash = self._get_source_files_hash()
        cache_filename = f"model_cache_{source_hash}.pkl"
        return list(self.resource_registry.get_model_cache_paths(cache_filename))

    def _is_cache_valid(self) -> bool:
        """Check if a valid cache file exists."""
        if not self.use_cache:
            return False

        cache_path = next(
            (path for path in self._get_cache_candidates() if os.path.exists(path)),
            None,
        )
        if cache_path is None:
            return False

        # Check if all source files exist
        for file_path in self._get_source_files():
            if not os.path.exists(file_path):
                return False

        # If we get here, cache file exists and source files exist
        return True

    def _load_from_cache(self) -> bool:
        """Load model data from cache file.

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        try:
            cache_path = next(
                (path for path in self._get_cache_candidates() if os.path.exists(path)),
                None,
            )
            if cache_path is None:
                return False

            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)

            # Validate cache format
            if not isinstance(cache_data, dict) or "version" not in cache_data:
                print("Warning: Invalid model cache format, rebuilding.")
                return False

            # Load entity collections
            collections = cache_data.get("entity_collections", {})
            self.reactions = collections.get("reactions", EntityCollection())
            self.metabolites = collections.get("metabolites", EntityCollection())
            self.genes = collections.get("genes", EntityCollection())
            self.subsystems = collections.get("subsystems", EntityCollection())
            self.compartments = collections.get("compartments", EntityCollection())

            # Load subsystem metabolite classification
            self.subsystem_metabolite_classification = cache_data.get(
                "subsystem_metabolite_classification", {}
            )

            # Load subsystem neighbor relationships
            self.subsystem_neighbor_relationships = cache_data.get(
                "subsystem_neighbor_relationships", []
            )
            # Initialize CategoryManager and load categories (not cached)
            # Categories are always loaded fresh to allow easy reloading
            self._category_manager = CategoryManager(self)
            if self.use_subsystem_classification:
                category_file = self.subsystem_classification_file_path  # May be None
                with redirect_stdout(io.StringIO()):
                    self._category_manager.load_categories(category_file)

            return True

        except Exception as e:
            print(f"Warning: Failed to load model cache: {e}")
            return False

    def _save_to_cache(self) -> None:
        """Save current model data to cache file."""
        if not self.use_cache:
            return

        try:
            cache_path = self._get_cache_path()

            # Prepare cache data
            cache_data = {
                "version": "1.0",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_files_hash": self._get_source_files_hash(),
                "entity_collections": {
                    "reactions": self.reactions,
                    "metabolites": self.metabolites,
                    "genes": self.genes,
                    "subsystems": self.subsystems,
                    "compartments": self.compartments,
                },
                "subsystem_metabolite_classification": (
                    self.subsystem_metabolite_classification
                ),
                "subsystem_neighbor_relationships": (
                    self.subsystem_neighbor_relationships
                ),
                "metadata": {
                    "reaction_count": self.reactions.count(),
                    "metabolite_count": self.metabolites.count(),
                    "gene_count": self.genes.count(),
                    "subsystem_count": self.subsystems.count(),
                    "compartment_count": self.compartments.count(),
                },
            }

            # Save to cache file
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            print(f"Warning: Failed to save model cache: {e}")

    def clear_cache(self) -> None:
        """Clear all cache files in the cache directory."""
        try:
            cache_dir = self._get_cache_dir()
            legacy_cache_dir = self.resource_registry.get_legacy_model_cache_dir()
            cache_directories = [cache_dir]
            if legacy_cache_dir not in cache_directories:
                cache_directories.append(legacy_cache_dir)

            removed_count = 0
            for candidate_cache_dir in cache_directories:
                if not os.path.isdir(candidate_cache_dir):
                    continue

                cache_files = [
                    filename
                    for filename in os.listdir(candidate_cache_dir)
                    if filename.startswith("model_cache_")
                    and filename.endswith(".pkl")
                ]

                for cache_file in cache_files:
                    cache_path = os.path.join(candidate_cache_dir, cache_file)
                    os.remove(cache_path)
                    removed_count += 1
                    print(f"🗑️  Removed cache file: {cache_file}")

            if removed_count:
                print(f"✅ Cleared {removed_count} cache file(s)")
            else:
                print("ℹ️  No cache files found to clear")

        except Exception as e:
            print(f"⚠️  Failed to clear cache: {e}")

    def get_cache_info(self) -> Dict[str, any]:
        """Get information about the current cache state."""
        cache_candidates = self._get_cache_candidates()
        active_cache_path = next(
            (path for path in cache_candidates if os.path.exists(path)),
            cache_candidates[0],
        )
        cache_info = {
            "use_cache": self.use_cache,
            "cache_dir": self._get_cache_dir(),
            "legacy_cache_dir": self.resource_registry.get_legacy_model_cache_dir(),
            "cache_candidates": cache_candidates,
            "current_cache_path": active_cache_path,
            "cache_valid": self._is_cache_valid(),
            "source_files_hash": self._get_source_files_hash(),
            "cache_size": len(self._cache),
            "cache_keys": list(self._cache.keys()),
        }

        # Add cache file info if it exists
        cache_path = active_cache_path
        if os.path.exists(cache_path):
            stat = os.stat(cache_path)
            cache_info.update(
                {
                    "cache_file_size_mb": stat.st_size / (1024 * 1024),
                    "cache_file_modified": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
                    ),
                }
            )

        return cache_info

    # ============================================================================
    # Additional Utility Methods
    # ============================================================================

    def clear_memory_cache(self) -> None:
        """Clear the internal memory cache."""
        self._cache.clear()

    def export_summary(self) -> Dict[str, any]:
        """Export a summary of the entire model for inspection or debugging."""
        return {
            "model_info": {
                "base_path": self.base_path,
                "lazy_loading": self.lazy_loading,
                "initialized": not self.lazy_loading,
            },
            "entity_counts": {
                "reactions": self.reactions.count(),
                "metabolites": self.metabolites.count(),
                "genes": self.genes.count(),
                "subsystems": self.subsystems.count(),
                "compartments": self.compartments.count(),
            },
            "available_formats": {
                "reactions": self.reactions.get_available_formats(),
                "metabolites": self.metabolites.get_available_formats(),
                "genes": self.genes.get_available_formats(),
                "subsystems": self.subsystems.get_available_formats(),
                "compartments": self.compartments.get_available_formats(),
            },
            "relationships_built": {
                "gene_to_subsystems": self._gene_to_subsystems is not None,
                "subsystem_to_genes": self._subsystem_to_genes is not None,
                "reaction_to_metabolites": self._reaction_to_metabolites is not None,
                "metabolite_to_reactions": self._metabolite_to_reactions is not None,
            },
        }

    # ============================================================================
    # Alignment Correction Methods
    # ============================================================================

    # Available alignment models and their corresponding file names
    ALIGNMENT_MODEL_FILES = {
        "Human1": None,
        "HMR1.0": "alignment_HMR1.0.txt",
        "HMR2.0": "alignment_HMR2.0.txt",
        "RECON1": "alignment_RECON1.txt",
        "RECON2": "alignment_RECON2.txt",
        "RECON2.2": "alignment_RECON2.2.txt",
        "RECON3D": "alignment_RECON3D.txt",
    }

    @classmethod
    def get_available_alignment_models(cls) -> List[str]:
        """Get list of available alignment model names.

        Returns:
            List of model names that can be used for alignment
        """
        return list(cls.ALIGNMENT_MODEL_FILES.keys())

    def get_alignment_file_path(self, model_name: str) -> Optional[str]:
        """Get the alignment file path for a given model name.

        Args:
            model_name: One of the available alignment model names
                (e.g., "Human1", "HMR1.0", "HMR2.0", "RECON1",
                "RECON2", "RECON2.2", "RECON3D")

        Returns:
            Full path to alignment file, or None if no alignment needed
            (i.e., "Human1" selected)

        Raises:
            ValueError: If model_name is not recognized
        """
        if model_name not in self.ALIGNMENT_MODEL_FILES:
            available = ", ".join(self.ALIGNMENT_MODEL_FILES.keys())
            raise ValueError(
                f"Unknown alignment model: '{model_name}'. "
                f"Available models: {available}"
            )

        filename = self.ALIGNMENT_MODEL_FILES[model_name]
        if filename is None:
            return None

        alignment_path = self.resource_registry.get_alignment_file(filename)

        if not os.path.exists(alignment_path):
            raise FileNotFoundError(f"Alignment file not found: {alignment_path}")

        return alignment_path

    def _load_alignment_file(self, alignment_file_path: str) -> Dict[str, int]:
        """Load alignment file mapping reaction IDs to alignment values.

        Args:
            alignment_file_path: Path to alignment file

        Returns:
            Dictionary mapping reaction_id -> alignment_value (0, 1, or -1)

        Format example:
            MAR03905 : 1
            MAR03907 : 1
            MAR04281 : -1

        Raises:
            FileNotFoundError: If alignment file doesn't exist
            ValueError: If alignment file has invalid format or values
        """
        alignment_mapping = {}

        # Check if file exists
        if not os.path.exists(alignment_file_path):
            raise FileNotFoundError(f"Alignment file not found: {alignment_file_path}")

        try:
            with open(alignment_file_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue

                    # Parse line: "reaction_id : value"
                    if ":" not in line:
                        raise ValueError(
                            f"Invalid format at line {line_num}: missing ':' separator"
                        )

                    parts = line.split(":")
                    if len(parts) != 2:
                        raise ValueError(
                            f"Invalid format at line {line_num}: expected 'reaction_id : value'"
                        )

                    reaction_id = parts[0].strip()
                    value_str = parts[1].strip()

                    # Convert value to int
                    try:
                        value = int(value_str)
                    except ValueError:
                        raise ValueError(
                            f"Invalid alignment value at line {line_num}: "
                            f"'{value_str}' is not an integer"
                        )

                    # Validate value is -1, 0, or 1
                    if value not in [-1, 0, 1]:
                        raise ValueError(
                            f"Invalid alignment value at line {line_num}: "
                            f"{value} (must be -1, 0, or 1)"
                        )

                    alignment_mapping[reaction_id] = value

        except (FileNotFoundError, ValueError):
            raise  # Re-raise these exceptions
        except Exception as e:
            raise ValueError(f"Error loading alignment file: {e}")

        return alignment_mapping

    def apply_alignment_corrections(self, alignment_file_path: str) -> Dict[str, any]:
        """Apply alignment corrections to reaction definitions in the model.

        For reactions marked with alignment value -1, this method swaps the
        reactants and products lists, effectively reversing the reaction
        stoichiometry. If a reaction is not reversible, it will be made
        reversible before applying the correction.

        This is useful when the reaction direction in the model doesn't match
        the expected biological direction for a specific analysis context.

        Note:
            Metabolite `produced_by` and `consumed_by` relationships are NOT
            updated by this method. This may cause inconsistencies between
            reaction definitions and metabolite relationships.

        Args:
            alignment_file_path: Path to alignment file with format "reaction_id : value"
                where value is:
                - 1: Reaction is correctly aligned (no change)
                - 0: Reaction is missing/not applicable (no change)
                - -1: Reaction should be reversed (swap reactants/products)

        Returns:
            Dict with correction statistics:
                - 'corrections_applied': Number of reactions corrected
                - 'reactions_not_found': List of reaction IDs not found in model
                - 'made_reversible': List of reactions that were made reversible
                - 'corrected_reactions': List of reaction IDs that were corrected

        Raises:
            FileNotFoundError: If alignment file doesn't exist
            ValueError: If alignment file has invalid format

        Example:
            >>> model = MetabolicModel()
            >>> results = model.apply_alignment_corrections("alignment.txt")
            >>> print(f"Corrected {results['corrections_applied']} reactions")
        """
        # Load alignment file
        alignment_mapping = self._load_alignment_file(alignment_file_path)

        # Track statistics
        corrections_applied = 0
        reactions_not_found = []
        made_reversible = []
        corrected_reactions = []

        for reaction_id, alignment_value in alignment_mapping.items():
            # Skip if alignment is 0 or 1 (no correction needed)
            if alignment_value in [0, 1]:
                continue

            # alignment_value is -1, need to reverse the reaction
            reaction = self.reactions.get(reaction_id)

            if reaction is None:
                reactions_not_found.append(reaction_id)
                continue

            # If reaction is not reversible, make it reversible first
            if not reaction.is_reversible:
                reaction.is_reversible = True
                if "<=>" not in reaction.equation:
                    if "=>" in reaction.equation:
                        reaction.equation = reaction.equation.replace("=>", "<=>")
                    elif "<=" in reaction.equation:
                        reaction.equation = reaction.equation.replace("<=", "<=>")
                    elif "->" in reaction.equation:
                        reaction.equation = reaction.equation.replace("->", "<=>")
                    elif "<-" in reaction.equation:
                        reaction.equation = reaction.equation.replace("<-", "<=>")

                made_reversible.append(reaction_id)

            # Swap reactants and products
            original_reactants = reaction.reactants.copy()
            original_products = reaction.products.copy()

            reaction.reactants = original_products
            reaction.products = original_reactants

            # Swap sides of the equation and formula strings
            if reaction.equation:
                if "<=>" in reaction.equation:
                    sides = reaction.equation.split("<=>")
                    reaction.equation = f"{sides[1].strip()} <=> {sides[0].strip()}"
                elif "=>" in reaction.equation:
                    sides = reaction.equation.split("=>")
                    reaction.equation = f"{sides[1].strip()} => {sides[0].strip()}"
                elif "<=" in reaction.equation:
                    sides = reaction.equation.split("<=")
                    reaction.equation = f"{sides[1].strip()} <= {sides[0].strip()}"
                elif "->" in reaction.equation:
                    sides = reaction.equation.split("->")
                    reaction.equation = f"{sides[1].strip()} -> {sides[0].strip()}"
                elif "<-" in reaction.equation:
                    sides = reaction.equation.split("<-")
                    reaction.equation = f"{sides[1].strip()} <- {sides[0].strip()}"

            if reaction.formula:
                if "<=>" in reaction.formula:
                    sides = reaction.formula.split("<=>")
                    reaction.formula = f"{sides[1].strip()} <=> {sides[0].strip()}"
                elif "=>" in reaction.formula:
                    sides = reaction.formula.split("=>")
                    reaction.formula = f"{sides[1].strip()} => {sides[0].strip()}"
                elif "<=" in reaction.formula:
                    sides = reaction.formula.split("<=")
                    reaction.formula = f"{sides[1].strip()} <= {sides[0].strip()}"
                elif "->" in reaction.formula:
                    sides = reaction.formula.split("->")
                    reaction.formula = f"{sides[1].strip()} -> {sides[0].strip()}"
                elif "<-" in reaction.formula:
                    sides = reaction.formula.split("<-")
                    reaction.formula = f"{sides[1].strip()} <- {sides[0].strip()}"

            # Negate stoichiometry coefficients if present
            if reaction.stoichiometry:
                reaction.stoichiometry = {
                    met_id: -coeff for met_id, coeff in reaction.stoichiometry.items()
                }

            corrections_applied += 1
            corrected_reactions.append(reaction_id)

        if reactions_not_found:
            print(
                f"Warning: {len(reactions_not_found)} reactions in the alignment "
                f"file were not found in the model."
            )

        return {
            "corrections_applied": corrections_applied,
            "reactions_not_found": reactions_not_found,
            "made_reversible": made_reversible,
            "corrected_reactions": corrected_reactions,
        }

    def get_subsystem_metabolite_classification(
        self, subsystem_name: str = None
    ) -> Dict[str, Dict[str, List[str]]]:
        """Get per-subsystem metabolite classification from Human1 maps.

        Args:
            subsystem_name: Specific subsystem to get classification for.
                            If None, returns all subsystem classifications.

        Returns:
            Dictionary mapping subsystem names to classification dictionaries
            containing "main" and "intermediate" metabolite lists
        """
        self._ensure_data_loaded("subsystem_metabolite_classification")

        if subsystem_name is None:
            return self.subsystem_metabolite_classification
        else:
            return {
                subsystem_name: self.subsystem_metabolite_classification.get(
                    subsystem_name, {"main": [], "intermediate": []}
                )
            }

    def get_subsystem_neighbors(
        self, subsystem_id: str = None
    ) -> List[SubsystemNeighborRelationship]:
        """Get neighbor relationships for a subsystem or all relationships.

        Args:
            subsystem_id: Specific subsystem to get neighbors for.
                         If None, returns all neighbor relationships.

        Returns:
            List of SubsystemNeighborRelationship objects
        """
        if subsystem_id is None:
            return self.subsystem_neighbor_relationships

        # Filter relationships where this subsystem is either producer or consumer
        return [
            rel
            for rel in self.subsystem_neighbor_relationships
            if (
                rel.producing_subsystem_id == subsystem_id
                or rel.consuming_subsystem_id == subsystem_id
            )
        ]

    def get_subsystem_producers(
        self, subsystem_id: str
    ) -> List[SubsystemNeighborRelationship]:
        """Get subsystems that produce metabolites consumed by the given subsystem.

        Args:
            subsystem_id: Subsystem to find producers for

        Returns:
            List of relationships where this subsystem is the consumer
        """
        return [
            rel
            for rel in self.subsystem_neighbor_relationships
            if rel.consuming_subsystem_id == subsystem_id
        ]

    def get_subsystem_consumers(
        self, subsystem_id: str
    ) -> List[SubsystemNeighborRelationship]:
        """Get subsystems that consume metabolites produced by the given subsystem.

        Args:
            subsystem_id: Subsystem to find consumers for

        Returns:
            List of relationships where this subsystem is the producer
        """
        return [
            rel
            for rel in self.subsystem_neighbor_relationships
            if rel.producing_subsystem_id == subsystem_id
        ]

    def get_neighbor_relationship(
        self,
        producing_subsystem_id: str,
        consuming_subsystem_id: str,
        metabolite_id: str = None,
    ) -> List[SubsystemNeighborRelationship]:
        """Get specific neighbor relationship between two subsystems.

        Args:
            producing_subsystem_id: Producer subsystem ID
            consuming_subsystem_id: Consumer subsystem ID
            metabolite_id: Optional specific metabolite ID to filter by

        Returns:
            List of matching relationships (may be multiple metabolites)
        """
        relationships = [
            rel
            for rel in self.subsystem_neighbor_relationships
            if rel.producing_subsystem_id == producing_subsystem_id
            and rel.consuming_subsystem_id == consuming_subsystem_id
        ]

        if metabolite_id is not None:
            relationships = [
                rel for rel in relationships if rel.metabolite_id == metabolite_id
            ]

        return relationships

    def retrieve_element_id(
        self, element_name: str, graph_type: str
    ) -> Tuple[str, str]:
        """Retrieve element ID and display title from element name.

        Args:
            element_name: Human-readable name (e.g., "Glycolysis / Gluconeogenesis")
            graph_type: One of "subsystem", "compartment", "metabolic_network"

        Returns:
            Tuple of (element_id, display_title)

        Raises:
            ValueError: If element not found or graph_type invalid
        """
        # Validate graph type
        valid_graph_types = ["subsystem", "compartment", "metabolic_network"]
        if graph_type not in valid_graph_types:
            raise ValueError(
                f"Invalid graph_type '{graph_type}'. "
                f"Must be one of: {valid_graph_types}"
            )

        # Handle each graph type
        if graph_type == "subsystem":
            return self._retrieve_subsystem_id(element_name)
        elif graph_type == "compartment":
            return self._retrieve_compartment_id(element_name)
        elif graph_type == "metabolic_network":
            return self._retrieve_metabolic_network_filter(element_name)
        else:
            # This should never happen due to validation above
            raise ValueError(f"Unsupported graph_type: {graph_type}")

    def _retrieve_subsystem_id(self, element_name: str) -> Tuple[str, str]:
        """Retrieve subsystem ID from name.

        Args:
            element_name: Subsystem name to look up

        Returns:
            Tuple of (subsystem_id, display_title)

        Raises:
            ValueError: If subsystem not found
        """
        self._ensure_data_loaded("subsystems")

        # First try exact match by name
        for subsystem in self.subsystems.get_all():
            if subsystem.name == element_name:
                return subsystem.id, element_name

        # Try by canonical subsystem ID
        for subsystem in self.subsystems.get_all():
            if subsystem.id == element_name:
                return subsystem.id, subsystem.name

        # If no exact match, try case-insensitive match
        element_name_lower = element_name.lower()
        for subsystem in self.subsystems.get_all():
            if subsystem.name.lower() == element_name_lower:
                return subsystem.id, subsystem.name

        # No match found
        available_subsystems = [s.name for s in self.subsystems.get_all()]
        if len(available_subsystems) > 10:
            error_msg = (
                f"Subsystem '{element_name}' not found. "
                f"Available subsystems: {available_subsystems[:10]}..."
            )
        else:
            error_msg = (
                f"Subsystem '{element_name}' not found. "
                f"Available subsystems: {available_subsystems}"
            )
        raise ValueError(error_msg)

    def _retrieve_compartment_id(self, element_name: str) -> Tuple[str, str]:
        """Retrieve compartment ID from name.

        Args:
            element_name: Compartment name to look up

        Returns:
            Tuple of (compartment_id, display_title)

        Raises:
            ValueError: If compartment not found
        """
        self._ensure_data_loaded("compartments")

        # First try exact match by name
        for compartment in self.compartments.get_all():
            if compartment.name == element_name:
                return compartment.id, f"Compartment: {compartment.name}"

        # Try by ID
        for compartment in self.compartments.get_all():
            if compartment.id == element_name:
                return compartment.id, f"Compartment: {compartment.name}"

        # If no exact match, try case-insensitive match
        element_name_lower = element_name.lower()
        for compartment in self.compartments.get_all():
            if compartment.name.lower() == element_name_lower:
                return compartment.id, f"Compartment: {compartment.name}"

        # No match found
        available_compartments = [(c.id, c.name) for c in self.compartments.get_all()]
        raise ValueError(
            f"Compartment '{element_name}' not found. "
            f"Available compartments: {available_compartments}"
        )

    def _retrieve_metabolic_network_filter(self, element_name: str) -> Tuple[str, str]:
        """Retrieve metabolic network filter configuration.

        Args:
            element_name: Filter specification (subsystem name, list, or None)

        Returns:
            Tuple of (filter_config, display_title)
        """
        if not element_name or element_name.strip() == "":
            # No filter - full metabolic network
            return "", "Complete Metabolic Network"

        # Handle subsystem filter
        if isinstance(element_name, str):
            # Single subsystem filter - validate it exists
            try:
                subsystem_id, _ = self._retrieve_subsystem_id(element_name.strip())
                display_title = f"Metabolic Network - {element_name.strip()}"
                return element_name.strip(), display_title
            except ValueError:
                # If subsystem doesn't exist, still allow it as a filter
                # This maintains backward compatibility
                display_title = f"Metabolic Network - {element_name.strip()}"
                return element_name.strip(), display_title
        elif isinstance(element_name, list):
            # Multiple subsystem filters
            count = len(element_name)
            display_title = f"Metabolic Network - {count} subsystems"
            return str(element_name), display_title
        else:
            # Invalid filter type
            return str(element_name), "Metabolic Network - Custom Filter"

    # ============================================================================
    # Interface Implementation - Core Data Access (continued)
    # ============================================================================
