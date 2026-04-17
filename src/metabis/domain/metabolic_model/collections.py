"""Generic entity collection for fast storage and ID translation.

This module provides the EntityCollection class, a generic container that offers:
- O(1) entity storage and retrieval by primary ID
- ID format translation between different naming conventions
- Support for alternative ID mappings
- Fast lookups and entity management

The EntityCollection is designed to work with any entity type that has an 'id'
attribute and optionally an 'id_mappings' dictionary for alternative formats.
"""

from typing import Dict, Generic, List, Optional, Set, TypeVar

T = TypeVar("T")


class EntityCollection(Generic[T]):
    """Generic container for fast entity access with O(1) lookups.

    Provides primary ID access and alternative ID format translations.

    This collection supports entities that have:
    - An 'id' attribute for the primary identifier
    - An optional 'id_mappings' attribute (Dict[str, str]) for alternative IDs

    Example:
        reactions = EntityCollection[Reaction]()
        reactions.add(some_reaction)
        reaction = reactions.get("rxn_001", "primary")
        bigg_id = reactions.translate_id("rxn_001", "primary", "bigg_id")
    """

    def __init__(self):
        """Initialize an empty entity collection."""
        self._entities: Dict[str, T] = {}  # primary_id -> entity
        # format -> {alt_id -> primary_id}
        self._id_index: Dict[str, Dict[str, str]] = {}
        self._formats: Set[str] = {"primary"}  # Available ID formats

    def add(self, entity: T) -> None:
        """Add entity and build ID indexes.

        Args:
            entity: Entity to add. Must have an 'id' attribute.

        Raises:
            ValueError: If entity doesn't have an 'id' attribute.
        """
        if not hasattr(entity, "id"):
            raise ValueError("Entity must have an 'id' attribute")

        primary_id = entity.id
        self._entities[primary_id] = entity

        # Build indexes for alternative ID formats if entity has id_mappings
        if hasattr(entity, "id_mappings") and entity.id_mappings:
            for format_name, alt_id in entity.id_mappings.items():
                if format_name not in self._id_index:
                    self._id_index[format_name] = {}
                    self._formats.add(format_name)
                self._id_index[format_name][alt_id] = primary_id

    def get(self, entity_id: str, id_format: str = "primary") -> Optional[T]:
        """Get entity by ID in any supported format.

        Args:
            entity_id: The entity ID to look up
            id_format: The format of the provided ID (default: "primary")

        Returns:
            The entity if found, None otherwise
        """
        if id_format == "primary":
            return self._entities.get(entity_id)

        if id_format not in self._id_index:
            return None

        primary_id = self._id_index[id_format].get(entity_id)
        if primary_id is None:
            return None

        return self._entities.get(primary_id)

    def get_all(self) -> List[T]:
        """Get all entities.

        Returns:
            List of all entities in the collection
        """
        return list(self._entities.values())

    def get_ids(self, id_format: str = "primary") -> List[str]:
        """Get all entity IDs in specified format.

        Args:
            id_format: The desired ID format (default: "primary")

        Returns:
            List of entity IDs in the specified format
        """
        if id_format == "primary":
            return list(self._entities.keys())

        if id_format not in self._id_index:
            return []

        return list(self._id_index[id_format].keys())

    def get_available_formats(self) -> List[str]:
        """Get all available ID formats for this entity type.

        Returns:
            List of available ID format names
        """
        return list(self._formats)

    def translate_id(
        self,
        entity_id: str,
        input_format: str = "primary",
        output_format: str = "primary",
    ) -> Optional[str]:
        """Translate an ID from one format to another.

        Args:
            entity_id: The ID to translate
            input_format: Format of the input ID (default: "primary")
            output_format: Desired output format (default: "primary")

        Returns:
            Translated ID if successful, None if translation fails
        """
        if input_format == output_format:
            return entity_id

        # Get the entity first
        entity = self.get(entity_id, input_format)
        if entity is None:
            return None

        # If output format is primary, return the primary ID
        if output_format == "primary":
            return entity.id

        # Otherwise, look up in id_mappings
        if hasattr(entity, "id_mappings") and output_format in entity.id_mappings:
            return entity.id_mappings[output_format]

        return None

    def count(self) -> int:
        """Get the number of entities in the collection.

        Returns:
            Number of entities stored
        """
        return len(self._entities)

    def contains(self, entity_id: str, id_format: str = "primary") -> bool:
        """Check if an entity exists with the given ID.

        Args:
            entity_id: The entity ID to check
            id_format: Format of the provided ID (default: "primary")

        Returns:
            True if entity exists, False otherwise
        """
        return self.get(entity_id, id_format) is not None
