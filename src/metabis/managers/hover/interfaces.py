from abc import ABC, abstractmethod
from typing import Any, Dict

from .models import HoverContext, HoverData

class HoverSection(ABC):
    """Abstract base class for hover text sections."""

    @abstractmethod
    def build(self, data: HoverData, context: HoverContext) -> str:
        """Build the HTML content for this section.

        Args:
            data: The data dictionary for the element (gene/reaction/etc.)
            context: Additional context (data_manager, settings, etc.)

        Returns:
            str: HTML content for this section
        """
        pass

    @abstractmethod
    def should_include(self, data: HoverData, context: HoverContext) -> bool:
        """Determine if this section should be included in the hover text.

        Args:
            data: The data dictionary for the element
            context: Additional context

        Returns:
            bool: True if section should be included
        """
        pass
