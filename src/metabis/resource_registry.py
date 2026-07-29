"""Centralized runtime resource resolution for metabis.

This module defines semantic resource aliases plus compatibility fallbacks so
runtime code can target stable domain meanings without hardcoding the current
on-disk resource layout.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESOURCES_ROOT = os.path.join(PACKAGE_ROOT, "resources")


def _default_cache_root() -> str:
    """Return the default writable cache root for runtime-generated files."""
    configured_cache_dir = os.getenv("METABIS_CACHE_DIR")
    if configured_cache_dir:
        return os.path.abspath(os.path.expanduser(configured_cache_dir))

    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    if xdg_cache_home:
        return os.path.join(
            os.path.abspath(os.path.expanduser(xdg_cache_home)),
            "metabis",
        )

    return os.path.join(str(Path.home()), ".cache", "metabis")


def _fallback_cache_root() -> str:
    """Return a last-resort writable cache root."""
    return os.path.join(tempfile.gettempdir(), "metabis")


@dataclass(frozen=True)
class ResourceLocation:
    """Semantic resource location with compatibility fallbacks."""

    canonical_relpath: str
    legacy_relpaths: Tuple[str, ...] = ()


class ResourceRegistry:
    """Resolve runtime resource paths from semantic aliases.

    The registry prefers semantically named canonical locations and falls back to
    legacy paths when needed for compatibility.
    """

    _ALIASES = {
        "reaction_catalog": ResourceLocation(
            "reference_model/reactions/reaction_catalog.csv",
            ("Human-GEM_meta.csv",),
        ),
        "reaction_formulas": ResourceLocation(
            "reference_model/reactions/reaction_formulas.txt",
            ("Human-GEM.txt",),
        ),
        "reaction_identifier_crosswalk": ResourceLocation(
            "reference_model/reactions/identifier_crosswalk.tsv",
            ("reactions.tsv",),
        ),
        "metabolite_identifier_crosswalk": ResourceLocation(
            "reference_model/metabolites/identifier_crosswalk.tsv",
            ("metabolites.tsv",),
        ),
        "gene_identifier_crosswalk": ResourceLocation(
            "reference_model/genes/identifier_crosswalk.tsv",
            ("genes.tsv",),
        ),
        "subsystem_category_taxonomy": ResourceLocation(
            "reference_model/subsystems/category_taxonomy.txt",
            ("subsystem_categories.txt",),
        ),
        "bigg_legacy_identifier_mapping": ResourceLocation(
            "model_interoperability/reaction_mappings/bigg_legacy_to_mar.csv",
            ("BIGG_rxn_old_identifiers_mapping.csv",),
        ),
        "hmr_to_mar_mapping": ResourceLocation(
            "model_interoperability/reaction_mappings/hmr_to_mar.tsv",
            ("HMR_con_MAR.tsv",),
        ),
    }

    _ALIGNMENT_DIRECTORY = ResourceLocation(
        "model_interoperability/model_alignments",
        ("Metabolic Models/AlignmentFiles",),
    )

    _GRAPH_ROOTS = {
        "subsystem": ResourceLocation(
            "visualization/subsystem_layouts",
            ("graphs/subsystem",),
        ),
        "compartment": ResourceLocation(
            "visualization/compartment_layouts",
            ("graphs/compartment",),
        ),
        "metabolic_network": ResourceLocation(
            "visualization/network_layouts",
            ("graphs/metabolic_network",),
        ),
    }

    def __init__(self, resources_root: str = None, cache_root: str = None):
        """Initialize the registry.

        Args:
            resources_root: Optional override for the package resource root.
            cache_root: Optional override for the writable runtime cache root.
        """
        self.resources_root = os.path.abspath(resources_root or DEFAULT_RESOURCES_ROOT)
        self.cache_root = os.path.abspath(cache_root or _default_cache_root())
        self._active_cache_root = None

    def _candidate_paths(self, location: ResourceLocation) -> Tuple[str, ...]:
        """Return canonical and legacy filesystem candidates for a location."""
        return (
            os.path.join(self.resources_root, location.canonical_relpath),
            *[
                os.path.join(self.resources_root, relpath)
                for relpath in location.legacy_relpaths
            ],
        )

    def _resolve_location(
        self,
        location: ResourceLocation,
        *,
        required: bool = True,
    ) -> str:
        """Resolve a semantic location to the first existing path."""
        candidate_paths = self._candidate_paths(location)
        for candidate_path in candidate_paths:
            if os.path.exists(candidate_path):
                return candidate_path

        if required:
            tried_paths = ", ".join(candidate_paths)
            raise FileNotFoundError(f"Metabis resource not found. Tried: {tried_paths}")

        return candidate_paths[0]

    def _resolve_alias(self, alias_name: str, *, required: bool = True) -> str:
        """Resolve a named semantic alias."""
        return self._resolve_location(self._ALIASES[alias_name], required=required)

    def get_reference_model_file_paths(self, *, required: bool = True) -> Dict[str, str]:
        """Return resolved file paths for the core reference model tables."""
        return {
            "reaction_catalog": self._resolve_alias(
                "reaction_catalog",
                required=required,
            ),
            "reaction_formulas": self._resolve_alias(
                "reaction_formulas",
                required=required,
            ),
            "reaction_identifier_crosswalk": self._resolve_alias(
                "reaction_identifier_crosswalk",
                required=required,
            ),
            "metabolite_identifier_crosswalk": self._resolve_alias(
                "metabolite_identifier_crosswalk",
                required=required,
            ),
            "gene_identifier_crosswalk": self._resolve_alias(
                "gene_identifier_crosswalk",
                required=required,
            ),
        }

    def get_subsystem_category_taxonomy_path(self, *, required: bool = True) -> str:
        """Return the subsystem category taxonomy file path."""
        return self._resolve_alias("subsystem_category_taxonomy", required=required)

    def get_bigg_legacy_identifier_mapping_path(self, *, required: bool = True) -> str:
        """Return the BiGG legacy identifier mapping file path."""
        return self._resolve_alias(
            "bigg_legacy_identifier_mapping",
            required=required,
        )

    def get_hmr_to_mar_mapping_path(self, *, required: bool = True) -> str:
        """Return the HMR-to-MAR mapping file path."""
        return self._resolve_alias("hmr_to_mar_mapping", required=required)

    def get_alignment_directory(self, *, required: bool = True) -> str:
        """Return the resolved alignment directory path."""
        return self._resolve_location(self._ALIGNMENT_DIRECTORY, required=required)

    def get_alignment_file(self, filename: str, *, required: bool = True) -> str:
        """Return the resolved path for an alignment file inside the alignment directory."""
        alignment_directory = self.get_alignment_directory(required=required)
        alignment_file = os.path.join(alignment_directory, filename)

        if os.path.exists(alignment_file) or not required:
            return alignment_file

        raise FileNotFoundError(f"Metabis alignment file not found: {alignment_file}")

    def get_graph_base_paths(self, *, required: bool = False) -> Dict[str, str]:
        """Return resolved graph resource roots keyed by graph type."""
        return {
            graph_type: self._resolve_location(location, required=required)
            for graph_type, location in self._GRAPH_ROOTS.items()
        }

    def get_runtime_cache_dir(self, namespace: str) -> str:
        """Return a writable runtime cache directory for the given namespace."""
        if self._active_cache_root is None:
            for candidate_root in (self.cache_root, _fallback_cache_root()):
                try:
                    os.makedirs(candidate_root, exist_ok=True)
                    self._active_cache_root = candidate_root
                    break
                except OSError:
                    continue

            if self._active_cache_root is None:
                raise OSError("Unable to create a writable metabis cache directory")

        cache_directory = os.path.join(self._active_cache_root, namespace)
        os.makedirs(cache_directory, exist_ok=True)
        return cache_directory

    def get_legacy_model_cache_dir(self) -> str:
        """Return the legacy in-package model cache directory path."""
        return os.path.join(self.resources_root, ".cache")

    def get_model_cache_paths(self, cache_filename: str) -> Tuple[str, str]:
        """Return preferred and legacy model cache locations for a cache file."""
        return (
            os.path.join(self.get_runtime_cache_dir("model"), cache_filename),
            os.path.join(self.get_legacy_model_cache_dir(), cache_filename),
        )
