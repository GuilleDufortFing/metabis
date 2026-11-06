import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from .metabolic_model import MetabolicModel
from .preprocessing import (
    ReactionPreprocessingConfig,
    ReactionTransform,
    preprocess_reactions,
    signed_cuberoot,
)
from .table_values import (
    SEPARATOR_NAMES,
    describe_non_finite,
    describe_repeated_samples,
    describe_repeats,
    describe_text_cell,
    non_finite_cells,
    renamed_repeats,
    repeated_header_names,
    text_cell,
)

# How a loading message names the file it read.
READ_WHAT = "The reaction file"


def _check_reaction_values(frame: pd.DataFrame) -> None:
    """Raise a plain ValueError naming the first cell of ``frame`` (samples x
    reactions, as read) that is not a number."""
    found = text_cell(frame)
    if found is not None:
        sample, reaction, value = found
        raise ValueError(describe_text_cell(READ_WHAT, "reaction", reaction, sample, value))
    problem = non_finite_cells(frame.to_numpy(dtype=float))
    if problem is not None:
        count, (row, column) = problem
        raise ValueError(
            describe_non_finite(READ_WHAT, count, "reaction", frame.columns[column], frame.index[row])
        )


def no_reaction_matched_message(
    path: str, reaction_name_format: str, data_mode: str, csv_separator: str, samples_in: str
) -> str:
    """Say that no reaction of the file at ``path`` was found in Human1, and how it was read."""
    separator = SEPARATOR_NAMES.get(csv_separator, repr(csv_separator))
    mode = "dual direction (_pos/_neg)" if str(data_mode).lower() == "dual" else "single direction"
    return (
        f"No reaction in '{path}' matches a Human1 reaction with its identifiers read as "
        f"'{reaction_name_format}', in {mode} mode, separated by {separator}, with the samples in "
        f"{samples_in}. Check the identifier format, the separator, which way the samples run "
        "and the data mode."
    )


# Recon1 and Recon2 write a reaction's compartment in parentheses, ``EX_glc(e)``;
# Recon3D - and so Human1's Recon3D annotation - writes it in brackets,
# ``EX_glc[e]``.
_PARENTHESIZED_COMPARTMENT = re.compile(r"\(([A-Za-z0-9]+)\)$")

# An SBML ID carries a type prefix and escapes what an ID may not hold, and
# Compass keeps them: Recon 2.2's EX_glc(e) reaches a file as
# R_EX_glc_LPAREN_e_RPAREN_.
_SBML_ESCAPES = {
    "_LPAREN_": "(", "_RPAREN_": ")", "_LSQBKT_": "[", "_RSQBKT_": "]", "_DASH_": "-",
    "_COMMA_": ",", "_SLASH_": "/", "_PERIOD_": ".", "_COLON_": ":", "_APOS_": "'",
}


def _identifier_spellings(source_id: str) -> List[str]:
    """Return a source reaction ID, then the same ID unescaped and as Recon3D spells it."""
    spellings = [source_id]
    decoded = source_id.removeprefix("R_")
    for escaped, character in _SBML_ESCAPES.items():
        decoded = decoded.replace(escaped, character)
    if decoded != source_id:
        spellings.append(decoded)
    for spelling in list(spellings):
        match = _PARENTHESIZED_COMPARTMENT.search(spelling)
        if match:
            spellings.append(f"{spelling[: match.start()]}[{match.group(1)}]")
    return spellings


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
        preprocessing: ReactionPreprocessingConfig | None = None,
        source_reactions: Mapping[str, Tuple[str, int]] | None = None,
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
                preprocessing=preprocessing,
                source_reactions=source_reactions,
            )

            # Perform validation after successful creation
            if (
                hasattr(instance, "df_reaction_data")
                and instance.df_reaction_data is not None
            ):
                # Generate alignment report
                alignment_report = instance.get_alignment_report()
                validation_results["alignment_report"] = alignment_report
                validation_results["preprocessing_report"] = (
                    instance.preprocessing_report
                )

                # Determine status based on mapping results
                if alignment_report["mapped_count"] == 0:
                    validation_results["status"] = "error"
                    error_msg = no_reaction_matched_message(
                        path, reaction_name_format, data_mode, csv_separator, samples_in
                    )
                    validation_results["messages"].append(error_msg)
                else:
                    # At least one reaction mapped = success
                    validation_results["status"] = "success"
                    validation_results["messages"].append(
                        "Reaction data loaded successfully"
                    )
                    validation_results["warnings"] = list(instance.load_warnings)
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
            validation_results["messages"].append(str(e))
            # Create a minimal instance for consistency
            instance = cls.__new__(cls)
            instance.metabolic_model = metabolic_model
            instance.df_reaction_data = None

        return instance, validation_results

    def _load_old_identifier_claims(self) -> Dict[str, List[str]]:
        """Load which Human1 reactions each legacy BiGG identifier is claimed by.

        The crosswalk is many-to-**many**: one legacy identifier can be listed by
        several Human1 reactions, and one Human1 reaction can list several legacy
        identifiers. Keeping the full claim list lets
        :meth:`_map_base_reactions_to_model` measure how *specific* a source
        identifier is before letting it represent a Human1 reaction.

        Returns:
            Dictionary mapping old_identifier -> list of rxn_ids (MAR IDs) that
            claim it, in file order.
            Example: {"ALCD2x": ["MAR03905"], "FAOXC80": ["MAR03160", ...]}

        Note:
            The CSV file has columns: rxn_id, old_identifiers
            Where old_identifiers is a comma-separated string.
        """
        claims: Dict[str, List[str]] = {}

        csv_path = self.metabolic_model.resource_registry.get_bigg_legacy_identifier_mapping_path(
            required=False
        )

        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"Warning: Old identifiers mapping file not found: {csv_path}")
            return claims

        try:
            # Load CSV
            df = pd.read_csv(csv_path)

            # Check required columns exist
            if "rxn_id" not in df.columns or "old_identifiers" not in df.columns:
                print("Warning: Old identifiers CSV missing required columns")
                return claims

            # Build claims: old_identifier -> [rxn_id, ...]
            for _, row in df.iterrows():
                rxn_id = row["rxn_id"]
                old_ids_str = row["old_identifiers"]

                # Skip if either is NaN or empty
                if pd.isna(rxn_id) or pd.isna(old_ids_str):
                    continue

                # Split comma-separated old identifiers and clean whitespace
                old_ids = [oid.strip() for oid in old_ids_str.split(",")]

                for old_id in old_ids:
                    if old_id:  # Skip empty strings
                        claims.setdefault(old_id, []).append(rxn_id)

        except Exception as e:
            print(f"Warning: Error loading old identifiers mapping: {e}")

        return claims

    def _load_old_identifiers_mapping(self) -> Dict[str, str]:
        """Load BiGG old identifiers mapping from CSV file.

        Returns:
            Dictionary mapping old_identifier -> rxn_id (MAR ID)
            Example: {"ALCD2x": "MAR03905", "ADH_c": "MAR03907", ...}
        """
        return {
            old_id: claiming[-1]
            for old_id, claiming in self._load_old_identifier_claims().items()
        }

    def _load_annotation_claims(self, reaction_name_format: str) -> Dict[str, List[str]]:
        """Load which Human1 reactions list each identifier of one format.

        Human1 lists several identifiers for some reactions, joined by ``;`` -
        ``HMR_8572;RE0944C`` - and each of them names that reaction, so each is
        indexed on its own. Claims are in model order, the order in which the
        model's own index resolves an identifier that several reactions share.

        Returns:
            Dictionary mapping identifier -> list of rxn_ids (MAR IDs) listing it.
        """
        claims: Dict[str, List[str]] = {}
        for mar_id, annotation in zip(
            self.metabolic_model.get_reactions(format="mar_id"),
            self.metabolic_model.get_reactions(format=reaction_name_format),
        ):
            if not isinstance(annotation, str):
                continue
            for identifier in annotation.split(";"):
                identifier = identifier.strip()
                if identifier:
                    claims.setdefault(identifier, []).append(mar_id)
        return claims

    def _map_base_reactions_to_model(
        self,
        reaction_columns: List[str],
        reaction_name_format: str,
        data_mode: str,
        check_old_identifiers: bool = False,
        source_reactions: Mapping[str, Tuple[str, int]] | None = None,
    ) -> Tuple[Dict[str, str], List[str], Dict[str, str], List[Dict[str, Any]]]:
        """Map base reaction IDs (without suffixes) to mar_id format.

        This is done BEFORE mode-specific processing to avoid redundant translations.

        The returned mapping is **injective**: at most one source reaction backs any
        Human1 reaction. Several source reactions can legitimately claim the same
        Human1 reaction - the crosswalk is many-to-many - and only one of them can
        supply that reaction's values, so the choice is made here, once, by an
        explicit policy rather than by whichever column happened to come last in the
        file. See :meth:`_resolve_reaction_id_collisions` for the policy.

        A source ID the declared source model's table lists maps where the table
        says: the table was read from both models' chemistry, so it knows which
        reaction that model means by an ID another release reuses, and it catches
        the identifiers Human1 misannotates. Any other ID is looked up in three
        tiers, the first that matches deciding its Human1 reaction: a Human1
        reaction's whole annotation in the input format; then one identifier of a
        listed annotation, or the ID unescaped or with its compartment respelled
        (see :func:`_identifier_spellings`); then, with ``check_old_identifiers``,
        the legacy BiGG table. Only the last is a high-effort match. The legacy
        table is read under the source's own spelling alone: respelled, Recon2's
        exchanges land in a block of it copied from misannotated Human1 reactions
        (``EX_Rtotal[e]`` names the exchange of ribose there).

        Args:
            reaction_columns: Raw column names from CSV
            reaction_name_format: Format of input reaction IDs
            data_mode: 'dual' or 'single'
            check_old_identifiers: If True, attempt high-effort mapping via old identifiers
            source_reactions: The declared source model's table, source reaction
                ID -> (MAR ID, orientation), from ``MetabolicModel.get_source_reactions``

        Returns:
            Tuple of:
                - Dict mapping original_base_id -> mar_id (only successful mappings)
                - List of unmapped base IDs
                - Dict of high-effort mappings (original_base_id -> mar_id)
                - List of collision records, one per contested mar_id
        """
        # 1. Extract base IDs by stripping mode-specific suffixes. Sorted, so every
        #    step below is reproducible: a set's iteration order over strings varies
        #    between processes.
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

        # 2. Get all Human1 reactions in the input format for quick lookup, as whole
        #    annotations and as the identifiers a listed annotation names
        model_reactions_in_input_format = set(
            self.metabolic_model.get_reactions(format=reaction_name_format)
        )
        annotation_claims = self._load_annotation_claims(reaction_name_format)

        # 3. Load old identifiers mapping if high-effort mode enabled
        old_identifier_claims: Dict[str, List[str]] = {}
        if check_old_identifiers:
            old_identifier_claims = self._load_old_identifier_claims()

        # 4. Map each base ID to mar_id, collecting every claim on each mar_id
        candidates: Dict[str, List[Dict[str, Any]]] = {}
        unmapped = []

        for base_id in sorted(base_ids):
            listed = next(
                (s for s in _identifier_spellings(base_id) if s in annotation_claims),
                None,
            )
            # How many Human1 reactions does this source claim? A source listed by
            # one reaction is specific to it; a lumped source listed by twelve is a
            # poor representative of any single one.
            claimed = set(old_identifier_claims.get(base_id, ()))

            if source_reactions and base_id in source_reactions:
                # The declared source model's own pairing, read from its chemistry.
                # Human1's annotation still says how lumped the source is: Recon2's
                # FAOXC80 is listed under twelve reactions, the thiolase step under one.
                mar_id = source_reactions[base_id][0]
                if listed is not None:
                    claimed.update(annotation_claims[listed])
                high_effort = False
            # Low effort: Check if base_id exists in model reactions (input format)
            elif base_id in model_reactions_in_input_format:
                mar_id = self.metabolic_model.translate_reaction(
                    base_id,
                    input_format=reaction_name_format,
                    output_format="mar_id",
                )
                high_effort = False
            elif listed is not None:
                # Still Human1's own identifier, listed beside others or spelled
                # as another model spells it. It outranks the legacy table, which
                # inherits BiGG's errors: there EX_ac(e), acetate exchange, is also
                # listed under the exchange of heptaglutamyl-folate.
                mar_id = annotation_claims[listed][-1]
                claimed.update(annotation_claims[listed])
                high_effort = False
            elif base_id in old_identifier_claims:
                # High effort: Try old identifiers mapping (if enabled)
                mar_id = old_identifier_claims[base_id][-1]
                high_effort = True
            else:
                unmapped.append(base_id)
                continue

            if not mar_id:
                unmapped.append(base_id)
                continue

            claimed.add(mar_id)
            candidates.setdefault(mar_id, []).append(
                {
                    "source_id": base_id,
                    "mapping_degree": len(claimed),
                    "high_effort": high_effort,
                }
            )

        # 5. Resolve any mar_id claimed by more than one source reaction
        mapping, collisions = self._resolve_reaction_id_collisions(candidates)
        high_effort_mappings = {
            record["source_id"]: mar_id
            for mar_id, entries in candidates.items()
            for record in entries
            if record["high_effort"] and mapping.get(record["source_id"]) == mar_id
        }

        return mapping, unmapped, high_effort_mappings, collisions

    @staticmethod
    def _read_orientations(
        mapping: Mapping[str, str],
        source_reactions: Mapping[str, Tuple[str, int]],
    ) -> Tuple[frozenset, List[str]]:
        """Which kept Human1 reactions to reverse, and the sources no table orients.

        A source the table lists carries its own orientation. One it does not list,
        mapped by identifier instead, takes its Human1 reaction's orientation when
        every source the table pairs with that reaction agrees on it; otherwise it
        stays as Human1 writes it and is counted as unread.

        Returns:
            The MAR IDs to reverse, and the source IDs left unread.
        """
        agreed = defaultdict(set)
        for mar_id, orientation in source_reactions.values():
            if orientation:
                agreed[mar_id].add(orientation)
        reversed_ids, unread = set(), []
        for source_id, mar_id in mapping.items():
            row = source_reactions.get(source_id)
            if row is not None and row[0] == mar_id:
                orientation = row[1]
            elif len(agreed.get(mar_id, ())) == 1:
                orientation = next(iter(agreed[mar_id]))
            else:
                orientation = 0
            if orientation == -1:
                reversed_ids.add(mar_id)
            elif orientation == 0:
                unread.append(source_id)
        return frozenset(reversed_ids), sorted(unread)

    @staticmethod
    def _resolve_reaction_id_collisions(
        candidates: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """Pick one source reaction per Human1 reaction, most specific first.

        Policy (A-19, approved 2026-07-24): among source reactions competing for one
        Human1 reaction, keep the one with the lowest **mapping degree** - the fewest
        Human1 reactions it claims. Ties break on an exact identifier match ahead of a
        legacy high-effort match, then lexicographically, so the result depends only
        on the identifiers themselves and never on column order in the file.

        Args:
            candidates: mar_id -> list of claim records

        Returns:
            Tuple of the injective source_id -> mar_id mapping and one record per
            contested mar_id naming the source kept and those discarded.
        """
        mapping: Dict[str, str] = {}
        collisions: List[Dict[str, Any]] = []

        def specificity(record: Dict[str, Any]) -> Tuple[int, int, str]:
            return (
                record["mapping_degree"],
                1 if record["high_effort"] else 0,
                record["source_id"],
            )

        for mar_id in sorted(candidates):
            ranked = sorted(candidates[mar_id], key=specificity)
            kept, discarded = ranked[0], ranked[1:]
            mapping[kept["source_id"]] = mar_id
            if discarded:
                collisions.append(
                    {
                        "mar_id": mar_id,
                        "kept": kept["source_id"],
                        "kept_degree": kept["mapping_degree"],
                        "discarded": [
                            (record["source_id"], record["mapping_degree"])
                            for record in discarded
                        ],
                    }
                )

        return mapping, collisions

    def __init__(
        self,
        path: str,
        data_mode: str,
        metabolic_model: MetabolicModel,
        reaction_name_format: str = "mar_id",
        csv_separator: str = "\t",
        samples_in: str = "rows",
        check_old_identifiers: bool = False,
        preprocessing: ReactionPreprocessingConfig | None = None,
        source_reactions: Mapping[str, Tuple[str, int]] | None = None,
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
            preprocessing: Typed preprocessing configuration. When omitted, dual
                inputs use Compass negative-log preprocessing and single inputs use
                the signed cube-root transform, preserving the mode defaults.
            source_reactions: The declared source model's table, from
                ``MetabolicModel.get_source_reactions``. It maps the IDs it lists,
                and ``reversed_reactions`` names the Human1 reactions whose kept
                source it writes the other way; the caller reverses them.

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
        self.preprocessing_report = None
        # What the load left out or found odd, as sentences for the reader.
        self.load_warnings: List[str] = []
        # Source base ID -> the Human1 reaction it backs, and the Human1 reactions
        # whose backing source the declared model writes the other way.
        self.reaction_mapping: Dict[str, str] = {}
        self.reversed_reactions: frozenset = frozenset()

        if preprocessing is None:
            preprocessing = (
                ReactionPreprocessingConfig(compass_activity=True)
                if data_mode.lower() == "dual"
                else ReactionPreprocessingConfig(
                    transform=ReactionTransform.SIGNED_CUBEROOT
                )
            )

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
            repeated = (
                repeated_header_names(path, csv_separator)
                if samples_in == "columns"
                else list(dict.fromkeys(original_df.index[original_df.index.duplicated()]))
            )
            if repeated:
                raise ValueError(describe_repeated_samples(READ_WHAT, repeated))

            # A reaction named twice keeps its first column. Along the header,
            # pandas renamed the second (R1.1), so the header itself says which.
            repeated_reactions = (
                repeated_header_names(path, csv_separator)
                if samples_in == "rows"
                else list(dict.fromkeys(original_df.columns[original_df.columns.duplicated()]))
            )
            if repeated_reactions:
                original_df = original_df.drop(columns=renamed_repeats(original_df.columns, repeated_reactions))
                original_df = original_df.loc[:, ~original_df.columns.duplicated(keep="first")]
                self.load_warnings.append(describe_repeats("reaction", repeated_reactions))

            # --- Step 1: Map base reactions to Human1 model FIRST ---
            # This is done before software-specific processing to avoid redundant translations
            (
                base_reaction_mapping,
                unmapped_reactions,
                high_effort_mappings,
                id_collisions,
            ) = self._map_base_reactions_to_model(
                list(original_df.columns),
                reaction_name_format,
                data_mode,
                check_old_identifiers,
                source_reactions=source_reactions,
            )
            self.reaction_mapping = dict(base_reaction_mapping)
            self.reversed_reactions, unread_orientations = self._read_orientations(
                base_reaction_mapping, source_reactions or {}
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
                "contested_reaction_count": len(id_collisions),
                "contesting_source_count": sum(
                    1 + len(record["discarded"]) for record in id_collisions
                ),
                "id_collisions": id_collisions,
                "reversed_count": len(self.reversed_reactions),
                "unread_orientation_count": (
                    len(unread_orientations) if source_reactions else 0
                ),
            }

            # Only the columns that become reactions must hold numbers: one the
            # model does not know, such as a description, is left alone.
            dual = data_mode.lower() == "dual"
            used = [
                column
                for column in original_df.columns
                if (
                    str(column)[:-4] in base_reaction_mapping
                    if dual and str(column).endswith(("_pos", "_neg"))
                    else not dual and column in base_reaction_mapping
                )
            ]
            _check_reaction_values(original_df.loc[:, used])
            if used and not original_df.loc[:, used].to_numpy(dtype=float).any():
                self.load_warnings.append(
                    "Every value of the reactions read is zero, so no reaction can differ between samples."
                )

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

                    elif neg_exists:
                        # A source reaction measured only in reverse. Dropping it
                        # would discard a real measurement, and upstream Compass
                        # keeps these.
                        processed_data[f"{mar_id}_back"] = original_df[
                            neg_reaction
                        ].abs()

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
                    # Check if reaction is reversible using the already-known mar_id or if neg_vals has any non-zero entries.
                    # A reaction to be reversed is made reversible when it is.
                    is_reversible = (
                        self.metabolic_model.is_reaction_reversible(
                            mar_id, input_format="mar_id"
                        )
                        or mar_id in self.reversed_reactions
                        or (neg_vals > 0).any()
                    )

                    if is_reversible:
                        processed_data[f"{mar_id}_back"] = neg_vals

                self.df_reaction_data = pd.DataFrame(
                    processed_data, index=original_df.index
                )

            # --- Step 3: Apply preprocessing ---
            self.df_reaction_data, self.preprocessing_report = preprocess_reactions(
                self.df_reaction_data, preprocessing
            )
            if preprocessing.transform is not ReactionTransform.NONE:
                self.applied_normalizations.append(preprocessing.transform.value)

        except FileNotFoundError:
            raise FileNotFoundError(f"Reaction data file not found: {path}")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"{READ_WHAT} is not text. Save it as CSV or TSV; a gzipped file's name must end in .gz."
            ) from error
        except pd.errors.EmptyDataError as error:
            raise ValueError(f"{READ_WHAT} is empty.") from error
        except pd.errors.ParserError as error:
            raise ValueError(f"{READ_WHAT} could not be read as a table: {error}") from error
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"{READ_WHAT} could not be read: {error}") from error

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
            "contested_reaction_count": 0,
            "contesting_source_count": 0,
            "unread_orientation_count": 0,
        }

        # Get mapping stats if available
        if hasattr(self, "mapping_stats") and self.mapping_stats:
            report["mapped_count"] = self.mapping_stats.get("mapped_count", 0)
            report["unmapped_count"] = self.mapping_stats.get("unmapped_count", 0)
            report["contested_reaction_count"] = self.mapping_stats.get(
                "contested_reaction_count", 0
            )
            report["contesting_source_count"] = self.mapping_stats.get(
                "contesting_source_count", 0
            )
            report["unread_orientation_count"] = self.mapping_stats.get(
                "unread_orientation_count", 0
            )

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
            subsystem_reactions = self.metabolic_model.get_reactions_for_subsystem(
                subsystem
            )
            # Check if any reaction in this subsystem is mapped
            for rxn in subsystem_reactions:
                if rxn in base_reactions_mapped:
                    subsystems_with_reactions += 1
                    break

        report["subsystems_with_reactions"] = subsystems_with_reactions

        return report


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
    return signed_cuberoot(reaction_data)
