"""Domain module for metabolic modeling.

This module contains the core domain objects and interfaces for working with
metabolic models and their associated data structures.

The domain is organized into several modules:
- entities: Core data structures (Reaction, Metabolite, Gene, etc.)
- collections: EntityCollection for O(1) entity storage and ID translation
- model: ModernMetabolicModel implementation
- metabolic_model: Legacy interface definition
- loaders: Data loading utilities (future)
- validation: Data validation and quality control (future)
- compatibility: Backward compatibility layer (future)
"""

from .collections import EntityCollection
from .entities import Compartment, Gene, Metabolite, Reaction, Subsystem
from .model import MetabolicModel

# Legacy support
# NOTE: Keep importing from legacy files for backward compatibility
# TODO: Add compatibility imports in future phases

__all__ = [
    # Core entities
    "Reaction",
    "Metabolite",
    "Gene",
    "Subsystem",
    "Compartment",
    # Collections
    "EntityCollection",
    # Models
    "MetabolicModel",
]
