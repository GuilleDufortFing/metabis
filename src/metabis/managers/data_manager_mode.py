"""Data manager modes enumeration.

This module defines the operation modes for the DataManager to avoid
circular import issues between data_manager.py and hover_manager.py.
"""

from enum import Enum


class DataManagerMode(Enum):
    """Enumeration of different DataManager operation modes."""

    GROUP_COMPARISON = "group_comparison"
    EXPERIMENT_ANALYSIS = "experiment_analysis"
    NO_EXPERIMENT = "no_experiment"
