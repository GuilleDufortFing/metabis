import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _strip_compass_suffixes(name: str) -> str:
    """Removes _pos or _neg suffix from a string if present."""
    if isinstance(name, str):
        if name.endswith("_pos") or name.endswith("_neg"):
            return name[:-4]
    return name


def _detect_csv_separator(path: str, n_lines: int = 10) -> Optional[str]:
    """Detect the most likely CSV separator by reading the first few lines."""
    if not os.path.exists(path):
        warnings.warn(f"File not found for separator detection: {path}")
        return None

    separators = {",": 0, "\t": 0, ";": 0}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            line_counts = []
            for i, line in enumerate(f):
                if i >= n_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                counts_in_line = {sep: line.count(sep) for sep in separators}
                line_counts.append(counts_in_line)

        if not line_counts:
            warnings.warn(
                f"Could not read any non-empty lines from {path} for separator detection."
            )
            return None

        total_counts = {sep: sum(lc[sep] for lc in line_counts) for sep in separators}

        if not any(total_counts.values()):
            warnings.warn(f"Could not detect any common separators in {path}")
            return None

        likely_sep = max(total_counts, key=total_counts.get)

        counts_for_likely_sep = [
            lc[likely_sep] for lc in line_counts if lc[likely_sep] > 0
        ]
        if len(counts_for_likely_sep) > 1:
            if max(counts_for_likely_sep) > 2 * min(counts_for_likely_sep) + 1:
                warnings.warn(
                    f"Separator '{likely_sep}' count varies significantly between lines in {path}. Detection might be unreliable."
                )
        elif not counts_for_likely_sep:
            warnings.warn(
                f"Separator '{likely_sep}' chosen but had zero counts in lines? Check logic."
            )
            return None
        return likely_sep
    except Exception as e:
        warnings.warn(f"Error reading file {path} for separator detection: {e}")
        return None


def _detect_compass_software_and_orientation(
    path: str,
    separator: Optional[str],
    n_check: int = 50,
    compass_threshold_ratio: float = 0.5,
) -> Dict[str, any]:
    """Detects if the software is Compass and infers orientation based on _pos/_neg suffixes.
    Returns: {"is_compass": bool, "orientation_if_compass": Optional[str]}
    """
    if not separator or not os.path.exists(path):
        return {"is_compass": False, "orientation_if_compass": None}

    is_compass = False
    orientation_if_compass = None
    compass_ids_in_header = 0
    compass_ids_in_index = 0
    total_header_candidates = 0
    total_index_candidates = 0

    try:
        # Check headers (potential reaction IDs if samples are rows)
        df_for_header = pd.read_csv(
            path, sep=separator, nrows=1
        )  # Read only header line
        if not df_for_header.empty:
            header_candidates = [str(col) for col in df_for_header.columns]
            total_header_candidates = len(header_candidates)
            for header_id in header_candidates:
                if header_id.endswith("_pos") or header_id.endswith("_neg"):
                    compass_ids_in_header += 1

        # Check first column/index (potential reaction IDs if samples are columns)
        df_for_index = pd.read_csv(path, sep=separator, nrows=n_check, index_col=0)
        if not df_for_index.empty:
            index_candidates = [str(idx) for idx in df_for_index.index]
            total_index_candidates = len(index_candidates)
            for index_id in index_candidates:
                if index_id.endswith("_pos") or index_id.endswith("_neg"):
                    compass_ids_in_index += 1

        header_compass_ratio = (
            compass_ids_in_header / total_header_candidates
            if total_header_candidates > 0
            else 0
        )
        index_compass_ratio = (
            compass_ids_in_index / total_index_candidates
            if total_index_candidates > 0
            else 0
        )

        # Prioritize if one clearly indicates Compass structure
        if (
            header_compass_ratio >= compass_threshold_ratio
            and header_compass_ratio > index_compass_ratio
        ):
            is_compass = True
            orientation_if_compass = "rows"  # Reactions in columns -> Samples in rows
        elif index_compass_ratio >= compass_threshold_ratio:
            is_compass = True
            orientation_if_compass = (
                "columns"  # Reactions in index -> Samples in columns
            )

    except pd.errors.EmptyDataError:
        warnings.warn(f"File {path} is empty for Compass detection.")
    except Exception as e:
        warnings.warn(f"Error during Compass software detection for {path}: {e}")
        # traceback.print_exc() # For debugging

    return {"is_compass": is_compass, "orientation_if_compass": orientation_if_compass}


def _detect_data_mode(
    path: str,
    separator: Optional[str],
    n_check: int = 100,
) -> Optional[str]:
    """Detects if the file is in 'single' or 'dual' data mode.
    
    Single mode: Values can be negative (indicating reverse flow), reactions are not repeated.
    Dual mode: Reversible reactions have paired _pos/_neg columns, all values are positive.
    
    Args:
        path: Path to the CSV file.
        separator: The CSV separator to use.
        n_check: Number of rows to check for detection.
    
    Returns: 
        "single", "dual", or None if detection is inconclusive.
    """
    if not separator or not os.path.exists(path):
        return None
    
    try:
        df = pd.read_csv(path, sep=separator, nrows=n_check, index_col=0)
        if df.empty:
            return None
        
        # Collect all potential reaction IDs (from both headers and index)
        header_ids = [str(col) for col in df.columns]
        index_ids = [str(idx) for idx in df.index]
        
        # Analyze both orientations
        for reaction_ids, numeric_axis in [(header_ids, 0), (index_ids, 1)]:
            # Count _pos/_neg pairs
            base_reactions = {}
            for rid in reaction_ids:
                if rid.endswith("_pos"):
                    base = rid[:-4]
                    base_reactions.setdefault(base, {"pos": False, "neg": False})["pos"] = True
                elif rid.endswith("_neg"):
                    base = rid[:-4]
                    base_reactions.setdefault(base, {"pos": False, "neg": False})["neg"] = True
            
            # Count paired reactions (have both _pos and _neg)
            paired_count = sum(1 for r in base_reactions.values() if r["pos"] and r["neg"])
            total_with_suffix = len(base_reactions)
            
            # Check for dual mode: significant proportion of paired _pos/_neg reactions
            # At least 30% paired and at least 5 pairs
            if total_with_suffix > 0:
                paired_ratio = paired_count / total_with_suffix
                if paired_ratio >= 0.3 and paired_count >= 5:
                    return "dual"
            
            # Check for single mode: presence of negative values
            try:
                if numeric_axis == 0:  # Numeric data in columns (after index)
                    numeric_data = df.select_dtypes(include=[np.number])
                else:  # Numeric data in rows
                    numeric_data = df.T.select_dtypes(include=[np.number])
                
                if not numeric_data.empty:
                    # If any negative values exist and no paired suffixes → single mode
                    has_negative = (numeric_data < 0).any().any()
                    if has_negative and total_with_suffix == 0:
                        return "single"
            except Exception:
                pass
        
        # Fallback: if we found _pos/_neg suffixes but not enough pairs, still likely dual
        all_ids = header_ids + index_ids
        has_any_suffix = any(rid.endswith("_pos") or rid.endswith("_neg") for rid in all_ids)
        
        if has_any_suffix:
            return "dual"  # Has suffixes, assume dual even if detection was weak
        
        return None  # Inconclusive
        
    except Exception as e:
        warnings.warn(f"Error during data mode detection for {path}: {e}")
        return None

def _detect_id_format_and_orientation(
    data_manager,
    path: str,
    separator: Optional[str],
    id_fetch_func,  # e.g., data_manager.metabolic_model.get_genes or .get_reactions
    prioritized_formats: List[str],
    all_formats: List[str],
    n_check: int = 100,  # Max items to read from file for candidacy
    threshold: float = 0.25,  # Threshold for the calculated score
    is_compass_file: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    if not data_manager.metabolic_model:
        warnings.warn("Metabolic model not loaded, cannot detect ID format.")
        return None, None
    if not separator or not os.path.exists(path):
        return None, None

    header_candidates = set()
    index_candidates = set()
    first_col_candidates = set()  # When index_col=0 is not appropriate

    # Read candidates from file (up to n_check items for headers/index/first_col)
    try:
        # Try reading with index_col=0 first
        # Limit rows read to n_check to get index candidates, header is from first row
        df = pd.read_csv(
            path, sep=separator, index_col=0, header=0, low_memory=False
        )  # Added header=0
        if not df.empty:
            # Limit header candidates read from file to n_check
            header_candidates = {str(col) for col in df.columns}
            # df.index is already limited by nrows=n_check
            index_candidates = {str(idx) for idx in df.index}
    except (
        pd.errors.EmptyDataError,
        ValueError,
        IndexError,
        TypeError,
    ):  # Added TypeError
        # Fallback or if index_col=0 is not suitable
        try:
            df_no_index = pd.read_csv(path, sep=separator, low_memory=False)
            if not df_no_index.empty:
                # Limit header candidates read from file to n_check
                header_candidates = df_no_index.columns
                if df_no_index.shape[1] > 0:
                    # For first_col_candidates, we take from the first n_check rows
                    # df_no_index is already limited by nrows
                    first_col_candidates = {str(val) for val in df_no_index.iloc[:, 0]}
        except pd.errors.EmptyDataError:
            warnings.warn(
                f"File {path} is empty or could not be parsed for ID detection."
            )
            return None, None
        except Exception as e:
            warnings.warn(f"Error reading CSV for ID detection: {path} - {e}")
            return None, None

    if not header_candidates and not index_candidates and not first_col_candidates:
        warnings.warn(
            f"No ID candidates found in file {path} (checked up to {n_check} rows/columns)."
        )
        return None, None

    best_match = {"format": None, "orientation": None, "score": -1.0}
    formats_to_check = prioritized_formats + [
        f for f in all_formats if f not in prioritized_formats
    ]

    def process_candidates_set(candidates_set: set, strip_suffixes: bool) -> set:
        if not strip_suffixes:
            return {str(c) for c in candidates_set}  # Ensure string conversion
        return {_strip_compass_suffixes(str(c)) for c in candidates_set}

    for fmt in formats_to_check:
        try:
            # known_ids_from_model are all IDs of this format from the (filtered) metabolic model
            known_ids_from_model = set(id_fetch_func(format=fmt)[:n_check])
        except Exception as e:
            warnings.warn(
                f"Error fetching IDs for format '{fmt}': {e}. Skipping format."
            )
            continue
        print(f"Checking format '{fmt}' with {len(known_ids_from_model)} known IDs.")
        if not known_ids_from_model:
            continue  # Cannot score if model has no IDs for this format

        num_known_ids_in_model = len(known_ids_from_model)
        if (
            num_known_ids_in_model == 0
        ):  # Should be caught by 'if not known_ids_from_model'
            continue

        # Process file candidates (these are already limited by n_check implicitly or explicitly)
        processed_headers = process_candidates_set(header_candidates, is_compass_file)
        processed_indices = process_candidates_set(index_candidates, is_compass_file)
        processed_first_cols = process_candidates_set(
            first_col_candidates, is_compass_file
        )

        # Number of matches: model IDs (for this format) found in the file's candidates.
        # This is limited by n_check because file_candidates are from n_check items.
        header_matches = len(known_ids_from_model.intersection(processed_headers))
        index_matches = len(known_ids_from_model.intersection(processed_indices))
        first_col_matches = len(known_ids_from_model.intersection(processed_first_cols))

        # Score: % of model's IDs (for this format) found in the file's (n_check limited) candidates.
        # If n_check is small and num_known_ids_in_model is large, scores will be small.
        header_score = header_matches / num_known_ids_in_model
        index_score = index_matches / num_known_ids_in_model
        first_col_score = first_col_matches / num_known_ids_in_model

        current_best_score_for_fmt = -1.0
        current_orientation_for_fmt = None

        if header_score > current_best_score_for_fmt:
            current_best_score_for_fmt = header_score
            current_orientation_for_fmt = (
                "rows"  # Features in columns -> samples in rows
            )
        if index_score > current_best_score_for_fmt:  # Check if index_score is better
            current_best_score_for_fmt = index_score
            current_orientation_for_fmt = (
                "columns"  # Features in index -> samples in columns
            )
        if (
            first_col_score > current_best_score_for_fmt
        ):  # Check if first_col_score is even better
            current_best_score_for_fmt = first_col_score
            current_orientation_for_fmt = (
                "columns"  # Features in first col -> samples in columns
            )

        if current_best_score_for_fmt > best_match["score"]:
            best_match = {
                "format": fmt,
                "orientation": current_orientation_for_fmt,
                "score": current_best_score_for_fmt,
            }

        # If a prioritized format achieves the primary threshold, we might stop early.
        # Note: This threshold might be hard to reach with the new scoring if n_check is restrictive
        # relative to the total number of model IDs for this format.
        if fmt in prioritized_formats and best_match["score"] >= threshold:
            return best_match["format"], best_match["orientation"]

    # After checking all formats, evaluate the best_match
    # Reconstruct the number of matches for the best found format to use in heuristic
    final_fmt_actual_matches = 0
    if best_match["format"]:
        try:
            _final_known_ids = set(id_fetch_func(format=best_match["format"]))
            _is_compass = (
                is_compass_file  # Use the same flag as during detection for consistency
            )
            if best_match["orientation"] == "rows":  # headers
                _final_proc_headers = process_candidates_set(
                    header_candidates, _is_compass
                )
                final_fmt_actual_matches = len(
                    _final_proc_headers.intersection(_final_known_ids)
                )
            elif best_match["orientation"] == "columns":  # index or first_col
                _final_proc_indices = process_candidates_set(
                    index_candidates, _is_compass
                )
                _final_proc_first_cols = process_candidates_set(
                    first_col_candidates, _is_compass
                )
                # Take the max of matches from index or first_col, as both map to "columns" orientation
                final_fmt_actual_matches = max(
                    len(_final_proc_indices.intersection(_final_known_ids)),
                    len(_final_proc_first_cols.intersection(_final_known_ids)),
                )
        except Exception:
            pass  # Error fetching IDs, final_fmt_actual_matches remains 0

    # Primary threshold check
    if best_match["format"] and best_match["score"] >= threshold:
        return best_match["format"], best_match["orientation"]

    # Secondary heuristic: if score is low but a reasonable number of matches were found
    # (e.g., > 5 matches and these constitute > 0.1% of model IDs for that format)
    # This helps when n_check is small compared to the total model IDs.
    # The min_matches_heuristic could be a fraction of n_check or a small absolute number.
    min_matches_heuristic = min(
        max(5, n_check // 10), 20
    )  # e.g., at least 5, or 10% of n_check, capped at 20
    # The score_heuristic_floor ensures we don't pick a format with trivial coverage of model IDs.
    score_heuristic_floor = 0.001  # Corresponds to 0.1% of model IDs matched

    if (
        best_match["format"]
        and best_match["score"] > score_heuristic_floor
        and final_fmt_actual_matches >= min_matches_heuristic
    ):
        warnings.warn(
            f"Detected format '{best_match['format']}' with orientation '{best_match['orientation']}' for {os.path.basename(path)} "
            f"based on a score of {best_match['score']:.4f} ({final_fmt_actual_matches} matches of model IDs found in file's first {n_check} candidates). "
            f"This score is below the primary threshold of {threshold} but met secondary heuristic criteria. Please verify."
        )
        return best_match["format"], best_match["orientation"]
    else:
        warnings.warn(
            f"Could not reliably detect ID format/orientation in {os.path.basename(path)}. "
            f"Best format '{best_match['format']}' had score {best_match['score']:.4f} ({final_fmt_actual_matches} matches of model IDs in file's first {n_check} candidates). "
            f"This may be due to `n_check` ({n_check}) limiting file candidates relative to the number of model IDs, "
            f"a poor match between file IDs and model IDs for any format, or the model lacking IDs for the file's true format. "
            f"Consider increasing `n_check` if the file is large and diverse, or verifying file content and model definitions."
        )
        return None, None


def detect_reaction_csv_params(data_manager, path: str) -> Dict[str, Optional[str]]:
    """Auto-detect separator, reaction ID format, sample orientation, and data mode for a reaction CSV."""
    separator = _detect_csv_separator(path)
    if separator is None:
        warnings.warn(
            f"Failed to detect separator for {path}. Cannot proceed with full detection."
        )
        return {"separator": None, "format": None, "samples_in": None, "data_mode": None}

    if not data_manager.metabolic_model:
        warnings.warn(
            "Metabolic model not loaded, cannot detect reaction format or data mode robustly."
        )
        return {
            "separator": separator,
            "format": None,
            "samples_in": None,
            "data_mode": None,
        }

    # Detect data mode (single vs dual)
    detected_data_mode = _detect_data_mode(path, separator)
    
    # Use compass detection for orientation hints when in dual mode
    compass_info = _detect_compass_software_and_orientation(path, separator)
    
    detected_orientation = None
    detected_format = None

    is_dual_mode = detected_data_mode == "dual" or compass_info["is_compass"]
    
    # If data_mode detection was inconclusive, use compass detection as fallback
    if detected_data_mode is None:
        detected_data_mode = "dual" if compass_info["is_compass"] else "single"

    if is_dual_mode and compass_info["orientation_if_compass"]:
        detected_orientation = compass_info["orientation_if_compass"]
        # Now detect base format, knowing it's dual mode and the orientation
        # _detect_id_format_and_orientation will strip suffixes.
        temp_format, temp_orientation_from_id_detect = (
            _detect_id_format_and_orientation(
                data_manager,
                path,
                separator,
                data_manager.metabolic_model.get_reactions,
                ["mar_id", "rxnHMR2ID", "rxnRecon3DID"],  # Prioritized
                data_manager.metabolic_model.get_available_reaction_formats(),  # All
                is_compass_file=True,
            )
        )
        detected_format = temp_format
        if (
            temp_orientation_from_id_detect
            and temp_orientation_from_id_detect != detected_orientation
        ):
            warnings.warn(
                f"Dual mode orientation ('{detected_orientation}') differs from ID-based orientation ('{temp_orientation_from_id_detect}') for {path}. "
                f"Using dual mode orientation. Detected base format: {detected_format}."
            )
    else:  # Single mode, or dual mode detection was inconclusive about orientation
        detected_format, detected_orientation = _detect_id_format_and_orientation(
            data_manager,
            path,
            separator,
            data_manager.metabolic_model.get_reactions,
            ["mar_id", "rxnHMR2ID", "rxnRecon3DID"],  # Prioritized
            data_manager.metabolic_model.get_available_reaction_formats(),  # All
            is_compass_file=is_dual_mode,
        )

    return {
        "separator": separator,
        "format": detected_format,
        "samples_in": detected_orientation,
        "data_mode": detected_data_mode,
    }


def detect_expression_csv_params(data_manager, path: str) -> Dict[str, Optional[str]]:
    """Auto-detect separator, gene ID format, and sample orientation for an expression CSV."""
    if path and path.lower().endswith(".h5ad"):
        warnings.warn("Auto-detection is only applicable to CSV/TSV files, not h5ad.")
        return {"separator": None, "format": None, "samples_in": None}

    separator = _detect_csv_separator(path)
    if separator is None:
        warnings.warn(
            f"Failed to detect separator for {path}. Cannot proceed with format detection."
        )
        return {"separator": None, "format": None, "samples_in": None}

    if not data_manager.metabolic_model:
        warnings.warn("Metabolic model not loaded, cannot detect gene format.")
        return {"separator": separator, "format": None, "samples_in": None}

    prioritized = ["ensmbl_id", "geneSymbols"]
    try:
        all_formats = data_manager.metabolic_model.get_available_gene_formats()
    except Exception as e:
        warnings.warn(f"Could not get available gene formats from model: {e}")
        all_formats = ["ensmbl_id", "geneSymbols"]  # Fallback

    # Expression files are not expected to be "Compass" format with _pos/_neg
    detected_format, detected_orientation = _detect_id_format_and_orientation(
        data_manager,
        path,
        separator,
        data_manager.metabolic_model.get_genes,
        prioritized,
        all_formats,
        is_compass_file=False,  # Expression data doesn't use _pos/_neg suffixes
    )

    return {
        "separator": separator,
        "format": detected_format,
        "samples_in": detected_orientation,
    }
