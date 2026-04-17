from typing import Any, Dict, List, Optional, Tuple
import os

import numpy as np
import pandas as pd

from .metabolic_model import MetabolicModel


class ReactionData:
    @classmethod
    def create_with_validation(
        cls,
        path: str,
        data_mode: str,
        metabolic_model: MetabolicModel,
        reaction_name_format: str = "mar_id",
        csv_separator: str = "\t",
        samples_in: str = "rows",
        check_old_identifiers: bool = False,
        normalization_options: Optional[List[str]] = None,
    ) -> tuple["ReactionData", Dict[str, Any]]:
        """Create ReactionData instance with validation results.

        Returns:
            tuple: (ReactionData instance, validation_results dict)
        """
        validation_results = {"status": "success", "messages": []}

        try:
            instance = cls(
                path=path,
                data_mode=data_mode,
                metabolic_model=metabolic_model,
                reaction_name_format=reaction_name_format,
                csv_separator=csv_separator,
                samples_in=samples_in,
                check_old_identifiers=check_old_identifiers,
                normalization_options=normalization_options,
            )

            # Perform validation after successful creation
            if (
                hasattr(instance, "df_reaction_data")
                and instance.df_reaction_data is not None
            ):
                # Columns should be mar_id after processing in ReactionData
                loaded_reactions_df = instance.get_reaction_df()
                loaded_reactions_count = len(loaded_reactions_df.columns)

                # Get model reactions in mar_id format for comparison
                model_reactions_mar_id = metabolic_model.get_reactions()
                model_reactions_mar_id_count = len(model_reactions_mar_id)

                # Generate alignment report
                alignment_report = instance.get_alignment_report()
                validation_results["alignment_report"] = alignment_report

                # Determine status based on mapping results
                if alignment_report["mapped_count"] == 0:
                    validation_results["status"] = "error"
                    error_msg = (
                        f"No reactions found in the CSV file ('{path}') "
                        f"match the reactions in the metabolic model "
                        f"(using input format '{reaction_name_format}' "
                        f"with '{data_mode}' mode processing and "
                        f"translated to 'mar_id'). Please check the reaction "
                        f"format, CSV separator ('{csv_separator}'), sample "
                        f"orientation ('{samples_in}'), and file content."
                    )
                    validation_results["messages"].append(error_msg)
                else:
                    # At least one reaction mapped = success
                    validation_results["status"] = "success"
                    validation_results["messages"].append(
                        "Reaction data loaded successfully"
                    )
            else:
                validation_results["status"] = "error"
                validation_results["messages"].append(
                    "Failed to initialize or process reaction data object. "
                    "Check logs for details."
                )

        except FileNotFoundError:
            validation_results["status"] = "error"
            validation_results["messages"].append(
                f"Reaction data CSV file not found at {path}"
            )
            # Create a minimal instance for consistency
            instance = cls.__new__(cls)
            instance.metabolic_model = metabolic_model
            instance.df_reaction_data = None
        except Exception as e:
            validation_results["status"] = "error"
            error_msg = (
                f"Error loading reaction data from CSV '{path}' "
                f"(sep='{csv_separator}', samples='{samples_in}', "
                f"mode='{data_mode}'): {str(e)}"
            )
            validation_results["messages"].append(error_msg)
            # Create a minimal instance for consistency
            instance = cls.__new__(cls)
            instance.metabolic_model = metabolic_model
            instance.df_reaction_data = None

        return instance, validation_results

    def _load_old_identifiers_mapping(self) -> Dict[str, str]:
        """
        Load BiGG old identifiers mapping from CSV file.

        Returns:
            Dictionary mapping old_identifier -> rxn_id (MAR ID)
            Example: {"ALCD2x": "MAR03905", "ADH_c": "MAR03907", ...}

        Note:
            The CSV file has columns: rxn_id, old_identifiers
            Where old_identifiers is a comma-separated string.
        """
        mapping = {}

        csv_path = (
            self.metabolic_model.resource_registry.get_bigg_legacy_identifier_mapping_path(
                required=False
            )
        )

        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"Warning: Old identifiers mapping file not found: {csv_path}")
            return mapping

        try:
            # Load CSV
            df = pd.read_csv(csv_path)

            # Check required columns exist
            if "rxn_id" not in df.columns or "old_identifiers" not in df.columns:
                print("Warning: Old identifiers CSV missing required columns")
                return mapping

            # Build mapping: old_identifier -> rxn_id
            for _, row in df.iterrows():
                rxn_id = row["rxn_id"]
                old_ids_str = row["old_identifiers"]

                # Skip if either is NaN or empty
                if pd.isna(rxn_id) or pd.isna(old_ids_str):
                    continue

                # Split comma-separated old identifiers and clean whitespace
                old_ids = [oid.strip() for oid in old_ids_str.split(",")]

                # Map each old identifier to the rxn_id
                for old_id in old_ids:
                    if old_id:  # Skip empty strings
                        mapping[old_id] = rxn_id

        except Exception as e:
            print(f"Warning: Error loading old identifiers mapping: {e}")

        return mapping

    def _map_base_reactions_to_model(
        self,
        reaction_columns: List[str],
        reaction_name_format: str,
        data_mode: str,
        check_old_identifiers: bool = False,
    ) -> Tuple[Dict[str, str], List[str], Dict[str, str]]:
        """
        Map base reaction IDs (without suffixes) to mar_id format.

        This is done BEFORE mode-specific processing to avoid redundant translations.

        Args:
            reaction_columns: Raw column names from CSV
            reaction_name_format: Format of input reaction IDs
            data_mode: 'dual' or 'single'
            check_old_identifiers: If True, attempt high-effort mapping via old identifiers

        Returns:
            Tuple of:
                - Dict mapping original_base_id -> mar_id (only successful mappings)
                - List of unmapped base IDs
                - Dict of high-effort mappings (original_base_id -> mar_id)
        """
        # 1. Extract base IDs by stripping mode-specific suffixes
        base_ids = set()
        for col in reaction_columns:
            if data_mode.lower() == "dual":
                # Strip _pos/_neg suffixes
                if col.endswith("_pos") or col.endswith("_neg"):
                    base_ids.add(col[:-4])
                else:
                    base_ids.add(col)
            else:
                # Single mode: columns are already base IDs
                base_ids.add(col)

        # 2. Get all Human1 reactions in the input format for quick lookup
        model_reactions_in_input_format = set(
            self.metabolic_model.get_reactions(format=reaction_name_format)
        )

        # 3. Load old identifiers mapping if high-effort mode enabled
        old_identifiers_mapping = {}
        if check_old_identifiers:
            old_identifiers_mapping = self._load_old_identifiers_mapping()

        # 4. Map each base ID to mar_id
        mapping = {}
        unmapped = []
        high_effort_mappings = {}

        for base_id in base_ids:
            # Low effort: Check if base_id exists in model reactions (input format)
            if base_id in model_reactions_in_input_format:
                mar_id = self.metabolic_model.translate_reaction(
                    base_id,
                    input_format=reaction_name_format,
                    output_format="mar_id",
                )
                if mar_id:
                    mapping[base_id] = mar_id
                else:
                    unmapped.append(base_id)
            else:
                # High effort: Try old identifiers mapping (if enabled)
                if check_old_identifiers and base_id in old_identifiers_mapping:
                    mar_id = old_identifiers_mapping[base_id]
                    mapping[base_id] = mar_id
                    high_effort_mappings[base_id] = mar_id
                else:
                    unmapped.append(base_id)

        return mapping, unmapped, high_effort_mappings

    def __init__(
        self,
        path: str,
        data_mode: str,
        metabolic_model: MetabolicModel,
        reaction_name_format: str = "mar_id",
        csv_separator: str = "\t",
        samples_in: str = "rows",
        check_old_identifiers: bool = False,
        normalization_options: Optional[List[str]] = None,
    ) -> None:
        """Load reaction data from a CSV file specified by the path.

        Args:
            path (str): Path to the reaction data CSV file.
            data_mode (str): Data mode for processing ('single' or 'dual').
                - 'single': Single column per reaction with positive/negative values.
                - 'dual': Separate columns for forward/backward (e.g., _pos/_neg suffixes).
            metabolic_model: The metabolic model to use for reaction information and translation.
            reaction_name_format (str): Format of reaction IDs in the data file.
            csv_separator (str): Separator used in the CSV file. Default is '\t'.
            samples_in (str): Location of sample names in the CSV ('rows' or 'columns'). Default is 'rows'.
            check_old_identifiers (bool): If True, attempt high-effort mapping via old identifiers. Default is False.
            normalization_options (List[str], optional): List of normalization methods to apply.
                Valid options: 'neg_log', 'cbrt', 'none'.
                If None, uses mode-specific defaults: 'neg_log' for dual, 'cbrt' for single.
                If empty list or ['none'], no normalization is applied.

        Returns:
            None

        Raises:
            ValueError: If data_mode is not 'single' or 'dual', or parameters invalid.
            FileNotFoundError: If the specified path does not exist.
        """
        # Validate data_mode and parameters
        if data_mode.lower() not in ["single", "dual"]:
            raise ValueError("data_mode must be 'single' or 'dual'")
        # Validate samples_in for CSV format
        if samples_in not in ["rows", "columns"]:
            raise ValueError("samples_in must be 'rows' or 'columns'")

        self.format = data_mode.lower()  # Refers to single/dual processing mode
        self.input_data_format = "csv"  # Explicitly set as csv
        self.metabolic_model = metabolic_model
        self.df_reaction_data = None  # Initialize
        self.applied_normalizations = []  # Track which normalizations were applied
        
        # Determine actual normalizations to apply
        if normalization_options is None:
            # Use mode-specific defaults
            if data_mode.lower() == "dual":
                actual_normalizations = ["neg_log"]
            else:
                actual_normalizations = ["cbrt"]
        elif normalization_options == [] or (len(normalization_options) == 1 and normalization_options[0].lower() == "none"):
            # Explicit no normalization
            actual_normalizations = []
        else:
            # Use provided options
            actual_normalizations = [n.lower() for n in normalization_options]

        # Load reaction data - Only CSV logic remains
        try:
            # Load CSV data
            df = pd.read_csv(path, sep=csv_separator, index_col=0)

            # Ensure samples are in rows and reactions are in columns for internal df_reaction_data
            if samples_in == "rows":
                # Input: rows=samples, columns=reactions -> Use directly
                original_df = df
            elif samples_in == "columns":
                # Input: rows=reactions, columns=samples -> Transpose needed
                original_df = df.T

            # Check for duplicate columns (reactions) before any processing
            if original_df.columns.duplicated().any():
                print(
                    "Warning: Found duplicate columns (reactions) in loaded reaction data. Keeping first instance."
                )
                original_df = original_df.loc[
                    :, ~original_df.columns.duplicated(keep="first")
                ]

            # --- Step 1: Map base reactions to Human1 model FIRST ---
            # This is done before software-specific processing to avoid redundant translations
            base_reaction_mapping, unmapped_reactions, high_effort_mappings = (
                self._map_base_reactions_to_model(
                    list(original_df.columns),
                    reaction_name_format,
                    data_mode,
                    check_old_identifiers,
                )
            )

            # Store mapping statistics for alignment report
            # Count unique base reactions from input (before suffix processing)
            base_ids = set()
            for col in original_df.columns:
                if data_mode.lower() == "dual":
                    if col.endswith("_pos") or col.endswith("_neg"):
                        base_ids.add(col[:-4])
                    else:
                        base_ids.add(col)
                else:
                    base_ids.add(col)
            
            self.mapping_stats = {
                "total_input_base_reactions": len(base_ids),
                "mapped_count": len(base_reaction_mapping),
                "unmapped_count": len(unmapped_reactions),
            }

            # --- Step 2: Mode-specific processing ---
            if data_mode.lower() == "dual":
                # Dual mode processing expects samples x reactions_with_suffixes
                # Process suffixes (_pos and _neg) and handle bidirectional reactions
                # Output directly in mar_id format using the pre-computed mapping
                processed_data = {}
                reaction_bases = {}

                # Identify base names from columns which might have _pos/_neg
                for col in original_df.columns:
                    base_name = col
                    if col.endswith("_pos") or col.endswith("_neg"):
                        base_name = col[:-4]
                    if base_name not in reaction_bases:
                        reaction_bases[base_name] = []
                    reaction_bases[base_name].append(col)

                # Process each base reaction (only if it maps to the model)
                for base_name, suffixed_reactions in reaction_bases.items():
                    # Skip if base_name doesn't map to the model
                    if base_name not in base_reaction_mapping:
                        continue

                    mar_id = base_reaction_mapping[base_name]
                    pos_reaction = f"{base_name}_pos"
                    neg_reaction = f"{base_name}_neg"

                    pos_exists = pos_reaction in original_df.columns
                    neg_exists = neg_reaction in original_df.columns

                    if pos_exists and neg_exists:
                        processed_data[mar_id] = original_df[pos_reaction]
                        processed_data[f"{mar_id}_back"] = original_df[
                            neg_reaction
                        ].abs()

                    elif pos_exists:
                        processed_data[mar_id] = original_df[pos_reaction]

                # Create DataFrame with mar_id columns directly
                self.df_reaction_data = pd.DataFrame(
                    processed_data, index=original_df.index
                )

            else:
                # Single mode processing: split bidirectional reactions
                # Forward: positive values, Backward: absolute value of negative values
                # Output directly in mar_id format using the pre-computed mapping
                processed_data = {}

                for col in original_df.columns:
                    # Skip if column doesn't map to the model
                    if col not in base_reaction_mapping:
                        continue

                    mar_id = base_reaction_mapping[col]

                    # Forward: Positive values (clip negatives to 0)
                    pos_vals = original_df[col].clip(lower=0)
                    # Backward: Absolute value of negative values
                    neg_vals = (-original_df[col].clip(upper=0)).abs()

                    processed_data[mar_id] = pos_vals
                    # Check if reaction is reversible using the already-known mar_id or if neg_vals has any non-zero entries
                    is_reversible = self.metabolic_model.is_reaction_reversible(
                        mar_id, input_format="mar_id"
                    ) or (neg_vals > 0).any()

                    if is_reversible:
                        processed_data[f"{mar_id}_back"] = neg_vals

                self.df_reaction_data = pd.DataFrame(
                    processed_data, index=original_df.index
                )

            # --- Step 3: Apply normalizations ---
            self._apply_normalizations(actual_normalizations)

        except FileNotFoundError:
            raise FileNotFoundError(f"Reaction data file not found: {path}")
        except Exception as e:
            # Add more context to the error message
            raise ValueError(
                f"Error loading or processing reaction data from CSV '{path}': {str(e)}"
            )

    def _apply_normalizations(self, normalization_options: List[str]) -> None:
        """Apply normalization methods to the reaction data.
        
        Args:
            normalization_options: List of normalization options to apply in order.
                Valid options: 'neg_log', 'cbrt'
        
        Modifies self.df_reaction_data in place and updates self.applied_normalizations.
        """
        if not normalization_options:
            return
        
        for normalization in normalization_options:
            norm_lower = normalization.lower()
            if norm_lower == "neg_log":
                self.df_reaction_data = apply_neg_log_normalization(
                    self.df_reaction_data, min_range=0.0
                )
                self.applied_normalizations.append("neg_log")
            elif norm_lower == "cbrt":
                self.df_reaction_data = apply_cbrt_normalization(self.df_reaction_data)
                self.applied_normalizations.append("cbrt")
            elif norm_lower == "none":
                # Explicitly skip normalization
                pass
            else:
                print(f"Warning: Unknown normalization option: {normalization}")

    def get_reaction_df(self) -> pd.DataFrame:
        """Get the reaction data as a DataFrame (samples x reactions).

        Returns:
            pd.DataFrame: Reaction data.
        """
        if self.df_reaction_data is None:
            raise ValueError(
                "Reaction data not loaded or processed correctly. Call __init__ first."
            )
        return self.df_reaction_data

    def get_reaction_row(
        self, reaction_id: str, reaction_format: str = "mar_id"
    ) -> Optional[pd.Series]:
        """Get a row (series) of values for a specific reaction across all samples.

        Args:
            reaction_id: Reaction identifier
            reaction_format: Format of the reaction ID (currently only 'mar_id' supported internally)

        Returns:
            pd.Series or None: Series with sample IDs as index and reaction values, or None if not found
        """
        if self.df_reaction_data is None:
            return None

        # Since our internal data is already in mar_id format, we expect reaction_format to be 'mar_id'
        # If other formats are needed, we'd need to add translation here
        if reaction_format != "mar_id":
            # Try to translate to mar_id format
            translated_id = self.metabolic_model.translate_reaction(
                reaction_id, input_format=reaction_format, output_format="mar_id"
            )
            if translated_id:
                reaction_id = translated_id
            # If translation fails, continue with original ID (might still work)

        if reaction_id not in self.df_reaction_data.columns:
            return None

        # Return the column as a Series (samples x values)
        return self.df_reaction_data[reaction_id]

    def get_sample_ids(self) -> List[str]:
        """Get list of all sample IDs.

        Returns:
            List[str]: List of sample IDs
        """
        if self.df_reaction_data is None:
            return []  # Return empty list if data not loaded
        return list(self.df_reaction_data.index)

    def get_alignment_report(self) -> Dict[str, Any]:
        """Generate an alignment report showing coverage statistics.

        Returns:
            Dict with:
                - mapped_count: reactions from input file that were mapped
                - unmapped_count: reactions from input that could not be mapped
                - base_reactions_mapped: unique base reactions mapped (no _back suffix)
                - total_model_reactions: total reactions in Human1 model
                - model_coverage_percent: percentage of model reactions mapped
                - subsystems_with_reactions: number of subsystems with mapped reactions
                - total_subsystems: total subsystems in the model
        """
        report = {
            "mapped_count": 0,
            "unmapped_count": 0,
            "base_reactions_mapped": 0,
            "total_model_reactions": 0,
            "model_coverage_percent": 0.0,
            "subsystems_with_reactions": 0,
            "total_subsystems": 0,
        }

        # Get mapping stats if available
        if hasattr(self, "mapping_stats") and self.mapping_stats:
            report["mapped_count"] = self.mapping_stats.get("mapped_count", 0)
            report["unmapped_count"] = self.mapping_stats.get("unmapped_count", 0)

        if self.df_reaction_data is None or self.df_reaction_data.empty:
            return report

        # Get all loaded reaction columns and extract unique base reactions (strip _back suffix)
        loaded_columns = list(self.df_reaction_data.columns)
        base_reactions_mapped = set()
        for col in loaded_columns:
            if col.endswith("_back"):
                base_reactions_mapped.add(col[:-5])  # Remove "_back" suffix
            else:
                base_reactions_mapped.add(col)

        report["base_reactions_mapped"] = len(base_reactions_mapped)

        # Get total model reactions (unique base reactions in mar_id format)
        model_reactions = self.metabolic_model.get_reactions(format="mar_id")
        report["total_model_reactions"] = len(model_reactions)

        # Calculate model coverage percentage
        if report["total_model_reactions"] > 0:
            report["model_coverage_percent"] = (
                report["base_reactions_mapped"] / report["total_model_reactions"]
            ) * 100

        # Count subsystems with at least one mapped reaction
        all_subsystems = self.metabolic_model.get_subsystems()
        report["total_subsystems"] = len(all_subsystems)

        subsystems_with_reactions = 0
        for subsystem in all_subsystems:
            subsystem_reactions = self.metabolic_model.get_reactions_for_subsystem(subsystem)
            # Check if any reaction in this subsystem is mapped
            for rxn in subsystem_reactions:
                if rxn in base_reactions_mapped:
                    subsystems_with_reactions += 1
                    break

        report["subsystems_with_reactions"] = subsystems_with_reactions

        return report


def apply_neg_log_normalization(reaction_penalties, min_range=1e-3):
    """Apply negative log normalization to convert penalty values into activity scores.
    
    Higher output values indicate more activity. This transformation is commonly
    used for penalty-based outputs where lower penalties indicate higher activity.
    
    Args:
        reaction_penalties: DataFrame with reaction penalty values
        min_range: Minimum range threshold for filtering low-variance reactions
    
    Returns:
        DataFrame with normalized activity scores
    """
    # Apply monotonic transform from penalties -> activity-like scores
    df = -np.log(reaction_penalties + 1)

    # Remove rows with negligible variation across samples
    df = df[df.max(axis=1) - df.min(axis=1) >= min_range]

    # Compute global minimum (scalar). If it's NaN or frame is empty, return as-is.
    global_min = df.min().min()
    if pd.isna(global_min) or df.shape[0] == 0 or df.shape[1] == 0:
        return df

    # If global_min is zero, nothing to shift.
    if global_min == 0:
        return df

    # Subtract global_min only for non-zero cells; keep exact zeros unchanged.
    # This preserves zero entries while shifting other values so the global
    # minimum becomes 0 (except for pre-existing zeros which stay zero).
    df = df.where(df == 0, df - global_min)

    return df


def apply_cbrt_normalization(reaction_data: pd.DataFrame) -> pd.DataFrame:
    """Apply cubic root normalization preserving sign.
    
    Formula: sign(x) * |x|^(1/3)
    
    This transformation compresses large values while preserving the sign of the data.
    It's useful for flux data where values can span several orders of magnitude.
    
    Args:
        reaction_data: DataFrame with reaction flux values
    
    Returns:
        DataFrame with cubic root normalized values
    """
    return np.sign(reaction_data) * np.abs(reaction_data) ** (1/3)
