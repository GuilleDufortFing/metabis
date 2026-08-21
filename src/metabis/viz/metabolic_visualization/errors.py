"""Failures raised by graph analysis and rendering.

Kept apart from the pipeline so the rendering adapter can report a failed render
without importing the pipeline that called it.
"""

from __future__ import annotations

from typing import Literal

GraphPhaseName = Literal["resolve", "generate", "enrich", "render", "serialize"]


class GraphPipelineError(RuntimeError):
    """A graph run failed, and names the phase and resource it failed on."""

    def __init__(
        self,
        message: str,
        *,
        phase: GraphPhaseName,
        resource_path: str | None = None,
        element_id: str | None = None,
    ):
        super().__init__(message)
        self.phase = phase
        self.resource_path = resource_path
        self.element_id = element_id


__all__ = ["GraphPhaseName", "GraphPipelineError"]
