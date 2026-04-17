"""Group management functionality for DataManager.

This module handles all group-related operations including group creation,
validation, querying, and color assignment.
"""

import re
from typing import Dict, List, Optional


class GroupManager:
    """Manages sample grouping operations and group-related queries."""

    def __init__(self):
        """Initialize GroupManager with empty state."""
        self.groups: Optional[Dict[str, List[str]]] = None
        self.active_group: Optional[str] = None

    def load_groups(
        self,
        sample_ids: List[str],
        criteria: str = "sample_prefix",
        columns: List[str] = None,
        fields: List[int] = None,
        metabolic_model=None,
        expression_data=None,
    ) -> Dict[str, List[str]]:
        """Organize samples into groups based on specified criteria.

        Args:
            sample_ids: List of sample identifiers
            criteria: Method to use for grouping samples. Supports:
                - "sample_prefix": Groups samples by prefix before first underscore
                - "adata_column": Groups samples by values in specified AnnData columns
                - "underscore_split": Groups samples by selected fields after splitting
            columns: List of column names to use when criteria is "adata_column"
            fields: List of field indices (0-based) to use when criteria is
                "underscore_split"
            metabolic_model: Metabolic model instance for gene translation (optional)
            expression_data: Expression data instance for adata_column criteria

        Returns:
            Dictionary containing grouping results and status information

        Raises:
            ValueError: If invalid criteria or missing required parameters
        """
        grouping_results = {"status": "success", "messages": []}
        grouped_samples: Dict[str, List[str]] = {}

        if criteria == "sample_prefix":
            grouped_samples = self._group_by_sample_prefix(sample_ids, metabolic_model)

        elif criteria == "underscore_split":
            if not fields or len(fields) == 0:
                raise ValueError(
                    "At least one field must be specified for underscore_split criteria"
                )
            grouped_samples = self._group_by_underscore_split(sample_ids, fields)

        elif criteria == "adata_column":
            if not columns:
                raise ValueError(
                    "At least one column must be specified for adata_column criteria"
                )
            grouped_samples = self._group_by_adata_column(
                sample_ids, columns, expression_data
            )

        else:
            raise ValueError(f"Unsupported grouping criteria: {criteria}")

        # Store the groups
        self.groups = grouped_samples

        grouping_results["messages"].append(f"Created {len(self.groups)} groups")

        return grouping_results

    def _group_by_sample_prefix(
        self, sample_ids: List[str], metabolic_model=None
    ) -> Dict[str, List[str]]:
        """Group samples by prefix before first underscore.

        Args:
            sample_ids: List of sample identifiers
            metabolic_model: Metabolic model for gene translation (optional)

        Returns:
            Dictionary mapping group IDs to sample lists
        """
        grouped_samples: Dict[str, List[str]] = {}
        pattern = r"([^_]+)_.*"

        for sample_id in sample_ids:
            match = re.match(pattern, sample_id)
            if match:
                group_id = match.group(1)

                # Handle Ensembl IDs if metabolic model is available
                if group_id.startswith("ENS") and metabolic_model:
                    genes = metabolic_model.get_genes()
                    if group_id in genes:
                        translated_id = metabolic_model.translate_gene(
                            group_id,
                            input_format="ensmbl_id",
                            output_format="geneSymbols",
                        )
                        group_id = translated_id or group_id

                if group_id not in grouped_samples:
                    grouped_samples[group_id] = []
                grouped_samples[group_id].append(sample_id)

        return grouped_samples

    def _group_by_underscore_split(
        self, sample_ids: List[str], fields: List[int]
    ) -> Dict[str, List[str]]:
        """Group samples based on selected underscore-separated fields.

        Args:
            sample_ids: List of sample identifiers
            fields: List of field indices (0-based) to use for grouping

        Returns:
            Dictionary mapping group IDs to sample lists
        """
        grouped_samples: Dict[str, List[str]] = {}
        print(f"Grouping samples by fields: {fields}")

        for sample_id in sample_ids:
            # Split sample ID by underscore
            split_parts = sample_id.split("_")

            # Extract the selected fields (handle index out of range safely)
            selected_values = []
            for field_idx in fields:
                if 0 <= field_idx < len(split_parts):
                    selected_values.append(split_parts[field_idx])
                else:
                    # Use empty string for out-of-range fields
                    selected_values.append("")

            # Create group ID by joining selected fields
            group_id = "_".join(selected_values)

            # Add to the appropriate group
            if group_id not in grouped_samples:
                grouped_samples[group_id] = []
            grouped_samples[group_id].append(sample_id)

        return grouped_samples

    def _group_by_adata_column(
        self, sample_ids: List[str], columns: List[str], expression_data
    ) -> Dict[str, List[str]]:
        """Group samples by values in AnnData observation columns.

        Args:
            sample_ids: List of sample identifiers
            columns: List of column names to use for grouping
            expression_data: Expression data instance with adata attribute

        Returns:
            Dictionary mapping group IDs to sample lists

        Raises:
            ValueError: If expression data is invalid or columns don't exist
        """
        # Check if expression data is available and in h5ad format
        if (
            not expression_data
            or not hasattr(expression_data, "adata")
            or expression_data.adata is None
        ):
            raise ValueError(
                "adata_column criteria requires h5ad format expression data"
            )

        print(f"Grouping samples by columns: {columns}")

        # Verify all columns exist in the AnnData object
        missing_cols = [
            col for col in columns if col not in expression_data.adata.obs.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Columns not found in AnnData observations: {missing_cols}"
            )

        grouped_samples: Dict[str, List[str]] = {}

        # Group samples by combination of column values
        for sample_id in sample_ids:
            if sample_id in expression_data.adata.obs.index:
                # Create group ID by joining values with underscores
                group_values = [
                    str(expression_data.adata.obs.loc[sample_id, col])
                    for col in columns
                ]
                group_id = "_".join(group_values)

                if group_id not in grouped_samples:
                    grouped_samples[group_id] = []
                grouped_samples[group_id].append(sample_id)

        return grouped_samples

    def set_active_group(self, group_id: str) -> None:
        """Set the current analysis group by ID.

        Args:
            group_id: ID of the group to set as active

        Raises:
            ValueError: If the group ID does not exist
        """
        if not self.groups or group_id not in self.groups:
            raise ValueError(f"Group '{group_id}' does not exist in experiment groups")

        self.active_group = group_id

    def get_group_samples(
        self, group_id: str, exclude_group: str = None, all_sample_ids: List[str] = None
    ) -> List[str]:
        """Get samples associated with a group.

        Args:
            group_id: ID of the group. If "The rest", returns all samples not in
                any other group.
            exclude_group: Group to exclude when getting "The rest" samples
            all_sample_ids: All available sample IDs (required for "The rest"
                functionality)

        Returns:
            List of sample IDs in the group

        Raises:
            ValueError: If requested group doesn't exist and isn't "The rest"
        """
        if not self.groups:
            raise ValueError("No groups have been loaded")

        # Special handling for "The rest" group
        if group_id == "The rest":
            if not all_sample_ids:
                raise ValueError(
                    "all_sample_ids parameter is required for 'The rest' group"
                )
            if not exclude_group or exclude_group not in self.groups:
                raise ValueError("Valid exclude_group is required for 'The rest' group")

            exclude_group_samples = self.groups[exclude_group]
            return [
                sample_id
                for sample_id in all_sample_ids
                if sample_id not in exclude_group_samples
            ]

        # Standard case - return the samples for the specified group
        if group_id not in self.groups:
            raise ValueError(f"Group '{group_id}' not found in experiment groups")

        return self.groups[group_id]

    def get_group_ids(self) -> List[str]:
        """Return the list of group IDs.

        Returns:
            List of group IDs, or empty list if no groups loaded
        """
        if not self.groups:
            return []
        return list(self.groups.keys())

    def get_active_group(self) -> Optional[str]:
        """Get the currently active group ID.

        Returns:
            Active group ID or None if no active group set
        """
        return self.active_group

    def has_groups(self) -> bool:
        """Check if groups have been loaded.

        Returns:
            True if groups are available, False otherwise
        """
        return self.groups is not None and len(self.groups) > 0

    def get_group_count(self) -> int:
        """Get the number of loaded groups.

        Returns:
            Number of groups, or 0 if no groups loaded
        """
        return len(self.groups) if self.groups else 0

    def validate_group_id(self, group_id: str) -> bool:
        """Validate that a group ID exists.

        Args:
            group_id: Group ID to validate

        Returns:
            True if group exists, False otherwise
        """
        return self.groups is not None and group_id in self.groups

    def clear_groups(self) -> None:
        """Clear all group data."""
        self.groups = None
        self.group_colors = None
        self.active_group = None

    def get_group_summary(self) -> Dict[str, int]:
        """Get a summary of group sizes.

        Returns:
            Dictionary mapping group IDs to sample counts
        """
        if not self.groups:
            return {}

        return {group_id: len(samples) for group_id, samples in self.groups.items()}
