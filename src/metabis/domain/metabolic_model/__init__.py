"""Domain module for metabolic modeling.

This module contains the core domain objects and interfaces for working with
metabolic models and their associated data structures.

The domain is organized into several modules:
- entities: Core data structures (Reaction, Metabolite, Gene, etc.)
- collections: EntityCollection for O(1) entity storage and ID translation
- model: MetabolicModel implementation
- loaders: Data loading utilities
"""

from .collections import EntityCollection
from .entities import Compartment, Gene, Metabolite, Reaction, Subsystem
from .model import MetabolicModel

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
