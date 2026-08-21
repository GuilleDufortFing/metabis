"""Analysis Workspace operation-mode enumeration.

This module defines the operation modes for the AnalysisWorkspace to avoid
circular import issues between workspace.py and hover_manager.py.
"""

from enum import Enum


class AnalysisWorkspaceMode(Enum):
    """Enumeration of different AnalysisWorkspace operation modes."""

    GROUP_COMPARISON = "group_comparison"
    EXPERIMENT_ANALYSIS = "experiment_analysis"
    NO_EXPERIMENT = "no_experiment"
