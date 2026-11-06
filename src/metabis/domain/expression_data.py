import warnings
from typing import Any, Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd

from .expression_matrix import (
    DenseExpressionMatrix,
    DenseMaterializationError,
    ExpressionMatrix,
    SparseExpressionMatrix,
    freeze_matrix_buffers,
    is_sparse,
)
from .h5ad_embedding_input import capture_h5ad_embedding_input
from .preprocessing import (
    ExpressionInputState,
    ExpressionPreprocessingConfig,
    PreprocessingReport,
    estimate_dense_bytes,
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
READ_WHAT = "The expression file"


def no_gene_matched_message(path: str, gene_format: str, data_format: str, csv_separator: str, samples_in: str) -> str:
    """Say that no gene of the file at ``path`` was found in Human1, and how it was read."""
    if data_format != "csv":
        return (
            f"No gene in '{path}' matches a Human1 metabolic gene with its identifiers read as "
            f"'{gene_format}'. Check the identifier format."
        )
    separator = SEPARATOR_NAMES.get(csv_separator, repr(csv_separator))
    return (
        f"No gene in '{path}' matches a Human1 metabolic gene with its identifiers read as "
        f"'{gene_format}', separated by {separator}, with the samples in {samples_in}. Check the "
        "identifier format, the separator and which way the samples run."
    )


class ExpressionData:
    """Aligned expression input, loaded from a CSV/TSV table or an h5ad file.

    Storage and sparsity (AM6)
    --------------------------
    An h5ad whose ``X`` is sparse is kept sparse through the load and through
    every supported analysis. Reads go through the :meth:`matrix` seam, which
    serves sample/feature identity, a single feature column, and the bounded
    dense blocks (a group's samples, the top-N variable features) that group
    summaries, the two-group comparison, the supported reductions
    (PCA/t-SNE/UMAP), and the activity z-score need -- so the whole matrix is
    never densified for a supported workflow. A dense CSV/TSV has no sparsity to
    preserve and is materialized at load exactly as before.

    The full dense frame is built only by the explicit escape hatch
    (:meth:`get_expression_df` / :meth:`ExpressionMatrix.to_dense_frame`). When a
    ``dense_memory_budget_bytes`` is set, an over-budget full densification is
    rejected before allocation from the shape/dtype estimate. A dense load is
    likewise rejected at load if over budget.

    Measured: at 5% nonzero the dense frame is ~10x the sparse storage; a
    sparse-first load plus a bounded reduction peaks several times below one
    resident dense frame, which the pre-AM6 load allocated and kept for the whole
    session. See ``MetabisReview/architecture-migration/am6-inventory.md``.
    """

    @classmethod
    def create_with_validation(
        cls,
        path: str,
        gene_list: List[str],
        gene_format: str = "ensmbl_id",
        metabolic_model=None,
        data_format: str = "csv",
        csv_separator: str = ",",
        samples_in: str = "rows",
        preprocessing: ExpressionPreprocessingConfig | None = None,
        dense_memory_budget_bytes: int | None = None,
    ) -> tuple["ExpressionData", Dict[str, Any]]:
        """Create ExpressionData instance with validation results.

        Returns:
            tuple: (ExpressionData instance, validation_results dict)
        """
        validation_results = {"status": "success", "messages": []}

        try:
            instance = cls(
                path=path,
                gene_list=gene_list,
                gene_format=gene_format,
                metabolic_model=metabolic_model,
                data_format=data_format,
                csv_separator=csv_separator,
                samples_in=samples_in,
                preprocessing=preprocessing,
                dense_memory_budget_bytes=dense_memory_budget_bytes,
            )

            # Perform validation after successful creation
            if hasattr(instance, "adata") and instance.adata is not None:
                # Generate alignment report
                alignment_report = instance.get_alignment_report()
                validation_results["alignment_report"] = alignment_report
                validation_results["preprocessing_report"] = instance.preprocessing_report

                # Determine status based on mapping results
                if alignment_report["mapped_count"] == 0:
                    validation_results["status"] = "error"
                    error_msg = no_gene_matched_message(path, gene_format, data_format, csv_separator, samples_in)
                    validation_results["messages"].append(error_msg)
                else:
                    # At least one gene mapped = success
                    validation_results["status"] = "success"
                    validation_results["messages"].append(
                        "Expression data loaded successfully"
                    )
                    validation_results["warnings"] = list(instance.load_warnings)
            else:
                validation_results["status"] = "error"
                validation_results["messages"].append(
                    "Failed to initialize or process expression data object. "
                    "Check logs for details."
                )

        except FileNotFoundError:
            validation_results["status"] = "error"
            validation_results["messages"].append(
                f"Expression data file not found at {path}"
            )
            # Create a minimal instance for consistency
            instance = cls.__new__(cls)
            instance.adata = None
            instance.df_expression_data = None
            instance.metabolic_model = metabolic_model
        except Exception as e:
            validation_results["status"] = "error"
            validation_results["messages"].append(str(e))
            # Create a minimal instance for consistency
            instance = cls.__new__(cls)
            instance.adata = None
            instance.df_expression_data = None
            instance.metabolic_model = metabolic_model

        return instance, validation_results

    def __init__(
        self,
        path: str,
        gene_list: List[str],
        gene_format: str = "ensmbl_id",
        metabolic_model=None,
        data_format: str = "csv",  # Added
        csv_separator: str = ",",  # Added
        samples_in: str = "rows",  # Added
        preprocessing: ExpressionPreprocessingConfig | None = None,
        dense_memory_budget_bytes: int | None = None,
    ):
        """Initialize the ExpressionData class.

        Args:
            path: Path to expression data file
            gene_list: List of valid genes from the metabolic model
            gene_format: Format of gene IDs in the input file
            metabolic_model: MetabolicModel instance for gene translation
            data_format (str): Format of the input data ('h5ad' or 'csv'). Default is 'csv'.
            csv_separator (str): Separator used if format is 'csv'. Default is ','.
            samples_in (str): Location of sample names in 'csv' format ('rows' or 'columns'). Default is 'rows'.
            dense_memory_budget_bytes: Optional preflight budget (AM6 slice 6). When
                set, a load whose dense materialization would exceed it is rejected
                *before* allocation with ``DenseMaterializationError``. ``None`` (the
                default) means no limit, so library-standalone use is unaffected; the
                server injects its configured budget.
        """
        # Data storage
        self.adata = None
        self.df_expression_data = None
        # The file's own name of each gene translated on loading, for messages.
        self._file_gene_names: Dict[str, str] = {}
        # What the load left out or found odd, as sentences for the reader.
        self.load_warnings: List[str] = []
        self.embedding_input = None
        self.format = None  # Will be set in build based on data_format
        self.gene_list = []
        self.metabolic_model = metabolic_model

        # File info
        self.source_path = None
        self.preprocessing_report = None
        preprocessing = preprocessing or ExpressionPreprocessingConfig()

        self.dense_memory_budget_bytes = dense_memory_budget_bytes

        # Call build method with all parameters
        self.build(
            path,
            gene_list,
            gene_format,
            data_format,
            csv_separator,
            samples_in,
            preprocessing,
            dense_memory_budget_bytes,
        )

    def build(
        self,
        path: str,
        gene_list: List[str],
        gene_format: str = "ensmbl_id",
        data_format: str = "csv",  # Added
        csv_separator: str = ",",  # Added
        samples_in: str = "rows",  # Added
        preprocessing: ExpressionPreprocessingConfig | None = None,
        dense_memory_budget_bytes: int | None = None,
    ):
        """Build the expression data object from the specified path.

        Args:
            path (str): Path to the expression data file.
            gene_list (List[str]): List of genes from the metabolic model (expected in ensmbl_id format).
            gene_format (str): Format of gene IDs in the input file. Default is "ensmbl_id".
            data_format (str): Format of the input data ('h5ad' or 'csv'). Default is 'csv'.
            csv_separator (str): Separator used if format is 'csv'. Default is ','.
            samples_in (str): Location of sample names in 'csv' format ('rows' or 'columns'). Default is 'rows'.

        Returns:
            None

        Raises:
            ValueError: If the format of the expression data is not supported or parameters are invalid.
        """
        self.source_path = path
        self.gene_list = (
            gene_list  # This is the target list (ensmbl_id) for filtering later
        )
        self.format = data_format  # Store the format used

        if data_format == "csv" and samples_in not in ["rows", "columns"]:
            raise ValueError(
                "If data_format is 'csv', samples_in must be 'rows' or 'columns'"
            )

        try:
            # --- Load Data ---
            if data_format == "h5ad":
                adata_raw = ad.read_h5ad(path)
                # AnnData standard: samples (obs) x genes (var)
                # Check for duplicate var_names (genes)
                if adata_raw.var_names.duplicated().any():
                    repeats = list(dict.fromkeys(adata_raw.var_names[adata_raw.var_names.duplicated()]))
                    self.load_warnings.append(describe_repeats("gene", repeats))
                    # Keep only the first occurrence of each gene
                    self.adata = adata_raw[
                        :, ~adata_raw.var_names.duplicated(keep="first")
                    ].copy()
                else:
                    self.adata = adata_raw

            elif data_format == "csv":
                # Load tabular data
                df_raw = pd.read_csv(path, sep=csv_separator, index_col=0)
                repeated = (
                    repeated_header_names(path, csv_separator)
                    if samples_in == "columns"
                    else list(dict.fromkeys(df_raw.index[df_raw.index.duplicated()]))
                )
                if repeated:
                    raise ValueError(describe_repeated_samples(READ_WHAT, repeated))
                found = text_cell(df_raw)
                if found is not None:
                    row, column, value = found
                    gene, sample = (column, row) if samples_in == "rows" else (row, column)
                    raise ValueError(describe_text_cell(READ_WHAT, "gene", gene, sample, value))

                # --- Handle Duplicates Before AnnData Creation ---
                if samples_in == "rows":
                    # Input: rows=samples, columns=genes. pandas renamed a gene the
                    # header names twice, so the header itself says which.
                    repeats = repeated_header_names(path, csv_separator)
                    df = df_raw
                    if repeats:
                        df = df_raw.drop(columns=renamed_repeats(df_raw.columns, repeats))
                        self.load_warnings.append(describe_repeats("gene", repeats))
                    # Create AnnData object
                    self.adata = ad.AnnData(
                        X=df.values,
                        obs=pd.DataFrame(index=df.index),  # Samples are in index
                        var=pd.DataFrame(index=df.columns),  # Genes are in columns
                    )
                elif samples_in == "columns":
                    # Input: rows=genes, columns=samples
                    if df_raw.index.duplicated().any():
                        repeats = list(dict.fromkeys(df_raw.index[df_raw.index.duplicated()]))
                        self.load_warnings.append(describe_repeats("gene", repeats))
                        df = df_raw[~df_raw.index.duplicated(keep="first")]
                    else:
                        df = df_raw
                    # Create AnnData object (transpose needed)
                    self.adata = ad.AnnData(
                        X=df.T.values,  # Transpose data matrix
                        obs=pd.DataFrame(index=df.columns),  # Samples are in columns
                        var=pd.DataFrame(index=df.index),  # Genes are in index
                    )

            else:
                raise ValueError(
                    f"Unsupported expression data format: {data_format}. Choose 'h5ad' or 'csv'."
                )

            # --- Gene Translation ---
            # Translate gene IDs (adata.var_names) to ensmbl_id format if necessary
            if gene_format != "ensmbl_id" and self.metabolic_model is not None:
                original_var_names = list(self.adata.var_names)
                translated_var_names = []
                genes_not_translated = 0
                for gene_id in original_var_names:
                    translated_id = self.metabolic_model.translate_gene(
                        gene_id, input_format=gene_format, output_format="ensmbl_id"
                    )
                    if translated_id:
                        translated_var_names.append(translated_id)
                    else:
                        # Keep original ID if translation fails, but log it
                        translated_var_names.append(gene_id)
                        genes_not_translated += 1

                if genes_not_translated > 0:
                    print(
                        f"Warning: Could not translate {genes_not_translated} gene IDs from {gene_format} to ensmbl_id. Keeping original IDs for these."
                    )

                # Update var_names in AnnData object
                self.adata.var_names = translated_var_names
                self._file_gene_names = dict(zip(translated_var_names, original_var_names))

                # --- Handle Duplicates *After* Translation ---
                if self.adata.var_names.duplicated().any():
                    names = self.adata.var_names
                    shared = list(dict.fromkeys(self._file_gene_names.get(gene, gene) for gene in names[names.duplicated()]))
                    self.load_warnings.append(
                        f"{len(shared):,} gene{'s' if len(shared) != 1 else ''} of the file name a Human1 gene another "
                        f"already names ({', '.join(map(str, shared[:3]))}{', ...' if len(shared) > 3 else ''}): "
                        "the first of each was kept."
                    )
                    # Re-filter adata based on non-duplicated translated names
                    self.adata = self.adata[
                        :, ~self.adata.var_names.duplicated(keep="first")
                    ].copy()

            # --- Gene Filtering ---
            # Filter genes by the model gene list (gene_list should be ensmbl_id)
            # Ensure self.adata.var_names are in ensmbl_id format (or original if translation failed)
            # The gene_list provided to the constructor should be in ensmbl_id format

            # Count genes before filtering for mapping statistics
            total_input_genes = len(self.adata.var_names)
            gene_list_set = set(gene_list)

            valid_genes_in_model_list = [
                g for g in self.adata.var_names if g in gene_list_set
            ]

            # Store mapping statistics
            self.mapping_stats = {
                "total_input_genes": total_input_genes,
                "mapped_count": len(valid_genes_in_model_list),
                "unmapped_count": total_input_genes - len(valid_genes_in_model_list),
                "total_model_genes": len(gene_list),
            }

            if len(valid_genes_in_model_list) == 0:
                print(
                    "Warning: No genes remaining in the expression data match the provided metabolic model gene list (ensmbl_id format)."
                )
                # Decide what to do: keep empty adata or raise error? Keeping empty for now.
                self.adata = self.adata[
                    :, valid_genes_in_model_list
                ].copy()  # This will result in 0 genes
            else:
                # Filter adata to keep only genes present in the model list
                self.adata = self.adata[:, valid_genes_in_model_list].copy()

            # --- Normalization ---
            config = preprocessing or ExpressionPreprocessingConfig()
            before_shape = tuple(self.adata.shape)
            zeros_before, non_finite_before = self._matrix_zero_and_nonfinite_counts()
            if non_finite_before:
                gene, sample = self._first_non_finite_cell()
                raise ValueError(describe_non_finite(READ_WHAT, non_finite_before, "gene", gene, sample))
            if self.adata.shape[0] and self.adata.shape[1] and zeros_before == self.adata.shape[0] * self.adata.shape[1]:
                self.load_warnings.append(
                    "Every value of the Human1 genes read is zero, so no gene can differ between samples."
                )
            report_warnings = []
            if config.input_state is ExpressionInputState.NORMALIZED and (
                config.normalize_total or config.log1p
            ):
                report_warnings.append(
                    "Input was declared normalized; requested transforms may duplicate preprocessing."
                )
            self._apply_preprocessing(config)

            # Capture imported coordinates and PCA metadata while the h5ad load is
            # being written. Workers receive this detached, read-only value rather
            # than the live AnnData object, and source identity can include the
            # already-paid embedding fingerprint without hashing during analysis.
            self.embedding_input = (
                capture_h5ad_embedding_input(self.adata)
                if data_format == "h5ad"
                else None
            )

            # AM6 slice 7: sparsity is preserved through the load. A sparse input
            # is *not* densified here -- the matrix seam (``matrix()``) serves
            # identity and the bounded reads (a group's samples, the top-N variable
            # features) the supported analyses need, so the whole dense frame is
            # never built. A dense input (CSV) has no sparsity to preserve, so it
            # densifies now exactly as before, keeping the dense path byte-for-byte
            # unchanged. Storage-mode provenance records which arrived.
            input_storage = "sparse" if is_sparse(self.adata.X) else "dense"
            dense_bytes = estimate_dense_bytes(self.adata)
            if input_storage == "dense":
                # The dense path materializes the frame now, so an over-budget
                # densification is rejected *before* allocation from the shape/dtype
                # estimate. ``None`` means no limit (the library default).
                if (
                    dense_memory_budget_bytes is not None
                    and dense_bytes > dense_memory_budget_bytes
                ):
                    raise DenseMaterializationError(
                        dense_bytes, dense_memory_budget_bytes
                    )
                if dense_bytes >= 1_000_000_000:
                    message = (
                        f"Materializing expression data densely requires approximately "
                        f"{dense_bytes / 1_000_000_000:.2f} GB."
                    )
                    warnings.warn(message, ResourceWarning, stacklevel=2)
                    report_warnings.append(message)
                self.df_expression_data = self.adata.to_df()
            else:
                # Sparse-first: no dense frame at load. The dense-memory budget does
                # not bite here -- a large sparse dataset must *load* and be analysed
                # through its bounded reads; the budget guards the explicit escape
                # hatch (``get_expression_df`` / ``ExpressionMatrix.to_dense_frame``)
                # if a full densification is ever genuinely forced.
                self.df_expression_data = None

            # The load performs no full-matrix conversion now -- the dense path
            # arrived dense, the sparse path stays sparse -- so none is recorded.
            conversions: tuple[str, ...] = ()
            zeros_after, non_finite_after = self._matrix_zero_and_nonfinite_counts()
            if non_finite_after:
                gene, sample = self._first_non_finite_cell()
                cells = "1 cell" if non_finite_after == 1 else f"{non_finite_after:,} cells, the first"
                raise ValueError(
                    f"The normalization left {cells} without a finite value (gene {gene} in sample {sample}): "
                    "log1p cannot take values of -1 or below, and CPM needs a positive total in each sample. "
                    "If the file's values are already normalized or transformed, set the normalization to None."
                )
            methods = [
                method
                for enabled, method in (
                    (config.normalize_total, "normalize_total"),
                    (config.log1p, "log1p"),
                )
                if enabled
            ]
            self.preprocessing_report = PreprocessingReport(
                method="+".join(methods) or "none",
                parameters={
                    "input_state": config.input_state.value,
                    "target_sum": config.target_sum,
                },
                input_shape=before_shape,
                output_shape=tuple(self.adata.shape),
                zeros_before=zeros_before,
                zeros_after=zeros_after,
                non_finite_before=non_finite_before,
                non_finite_after=non_finite_after,
                warnings=tuple(report_warnings),
                input_storage=input_storage,
                conversions=conversions,
            )

        except FileNotFoundError:
            raise FileNotFoundError(f"Expression data file not found: {path}")
        except DenseMaterializationError:
            # The preflight rejection is a clean, actionable budget error; surface it
            # as-is rather than burying it in a load-failure traceback string.
            raise
        except UnicodeDecodeError as error:
            raise ValueError(
                f"{READ_WHAT} is not text. Save it as CSV or TSV, or as h5ad; a gzipped file's name must end in .gz."
            ) from error
        except pd.errors.EmptyDataError as error:
            raise ValueError(f"{READ_WHAT} is empty.") from error
        except pd.errors.ParserError as error:
            raise ValueError(f"{READ_WHAT} could not be read as a table: {error}") from error
        except ValueError:
            raise
        except Exception as error:
            if data_format == "h5ad":
                raise ValueError(f"{READ_WHAT} is not a readable h5ad file: {error}") from error
            raise ValueError(f"{READ_WHAT} could not be read: {error}") from error

    def _first_non_finite_cell(self) -> tuple[str, str]:
        """The gene, as the file names it, and the sample of the matrix's first non-finite cell."""
        matrix_x = self.adata.X
        if is_sparse(matrix_x):
            coo = matrix_x.tocoo()
            first = int(np.argmax(~np.isfinite(coo.data)))
            row, column = int(coo.row[first]), int(coo.col[first])
        else:
            _, (row, column) = non_finite_cells(np.asarray(matrix_x))
        gene = self.adata.var_names[column]
        return self._file_gene_names.get(gene, gene), self.adata.obs_names[row]

    def _matrix_zero_and_nonfinite_counts(self) -> tuple[int, int]:
        """``(zeros, non_finite)`` over the current matrix, sparse-aware, no densify.

        For sparse ``X`` the counts come from the stored nonzeros and ``nnz`` -- an
        implicit zero is an exact, finite zero -- so the whole matrix is never
        materialized. For dense ``X`` they come from the array. Used for both the
        pre- and post-preprocessing accounting the ``PreprocessingReport`` records
        (AM6 slice 7: the report no longer reads a densified frame).
        """
        matrix_x = self.adata.X
        if is_sparse(matrix_x):
            stored = np.asarray(matrix_x.data)
            zeros = int(np.prod(self.adata.shape) - matrix_x.nnz)
            non_finite = int((~np.isfinite(stored)).sum())
        else:
            values = np.asarray(matrix_x)
            zeros = int((values == 0).sum())
            non_finite = int((~np.isfinite(values)).sum())
        return zeros, non_finite

    def _reject_over_budget_densification(self) -> None:
        """Refuse an escape-hatch full densification that exceeds the load budget.

        AM6 slice 7 moves the dense-memory budget off the sparse load and onto the
        point where a full dense frame is genuinely forced. The estimate reads only
        shape and dtype, so an over-budget densification is rejected *before*
        allocation. ``None`` (the default) means no limit.
        """
        budget = getattr(self, "dense_memory_budget_bytes", None)
        if budget is None or self.adata is None:
            return
        estimated = estimate_dense_bytes(self.adata)
        if estimated > budget:
            raise DenseMaterializationError(estimated, budget)

    def _apply_preprocessing(self, config: ExpressionPreprocessingConfig) -> None:
        """Apply a typed expression preprocessing configuration in order."""
        # Scanpy costs about 200 MiB to import, so a process pays for it only once
        # some expression is actually normalized.
        import scanpy as sc

        if config.normalize_total:
            sc.pp.normalize_total(self.adata, target_sum=config.target_sum, inplace=True)
        if config.log1p:
            sc.pp.log1p(self.adata)

    def keep_samples(self, sample_ids: List[str]) -> None:
        """Keep only ``sample_ids`` (of the loaded samples, in their order).

        A loader calls it before the data is installed and frozen, to leave out the
        samples another dataset lacks; the dense frame, when built, follows.
        """
        keep = list(sample_ids)
        self.adata = self.adata[keep].copy()
        if self.df_expression_data is not None:
            self.df_expression_data = self.df_expression_data.loc[keep]

    def freeze(self) -> None:
        """Mark the canonical expression buffers read-only (M-11, AM9).

        The Analysis Workspace calls this when it installs this data, sealing the
        replace-only contract: after install the resident matrix must not be
        edited in place while a worker snapshot references it. ``adata.X`` is the
        canonical store the AM9 sparse snapshot reads through the ``matrix()``
        seam, so freezing its buffers makes a same-shape in-place edit raise at
        its source (prevention, the mechanism M-11 chose). ``df_expression_data``
        is a ``to_df`` copy no analysis mutates, and a pandas frame cannot be
        reliably frozen under Copy-on-Write, so it is left to the static boundary
        guard. Idempotent, so re-installing an unchanged matrix is harmless.
        """
        freeze_matrix_buffers(getattr(self.adata, "X", None))

    def get_expression_df(self) -> pd.DataFrame:
        """Get the expression data as a pandas DataFrame (samples x genes).

        This is the explicit full-densification escape hatch. AM6 slice 7 stops the
        load building this frame for a sparse input, so for a sparse load the first
        call materializes the whole matrix dense (and caches it). The supported
        analyses no longer reach here -- they read the bounded ``matrix()`` seam --
        so a caller landing here genuinely wants the whole dense frame. When a
        dense-memory budget is set, an over-budget materialization is rejected
        before allocation.

        Returns:
            pd.DataFrame: Expression data as a DataFrame.
        """
        if self.df_expression_data is None:
            # Check if adata exists but df is None (sparse-first load, or an error
            # during build). Materialize it now, honouring the budget first.
            if self.adata is not None:
                self._reject_over_budget_densification()
                self.df_expression_data = self.adata.to_df()
                if self.df_expression_data is not None:
                    return self.df_expression_data
            raise ValueError(
                "Expression data not loaded or processed correctly. Call build() first."
            )

        return self.df_expression_data

    def matrix(self) -> Optional[ExpressionMatrix]:
        """The expression-matrix seam over the loaded data (AM6 slices 2-3).

        Every value reader goes through this one seam. When ``X`` is sparse it
        returns a sparse adapter that serves identity and a single feature column
        without densifying the whole matrix; otherwise a dense adapter over the
        built frame. Either way the values are identical -- a caller cannot tell
        which adapter it holds by results, only by memory cost. Returns ``None``
        when no data is loaded, matching the readers that already tolerate an
        unloaded matrix. It does not build the dense frame -- ``get_expression_df``
        owns that lazy materialization -- so a caller that only wants identity or
        one column never forces the full densification here.

        The load's ``dense_memory_budget_bytes`` (``None`` for direct-library use)
        rides onto the adapter, so every bounded block and the escape hatch this
        seam serves are preflight-gated by the same budget (M-11, AM9). Threading
        it here means a worker reading through ``StableMatrixReference`` -> this
        ``matrix()`` inherits it without any per-call plumbing.
        """
        budget = getattr(self, "dense_memory_budget_bytes", None)
        if self.adata is not None and is_sparse(self.adata.X):
            return SparseExpressionMatrix(self.adata, max_bytes=budget)
        if self.df_expression_data is not None:
            return DenseExpressionMatrix(self.df_expression_data, max_bytes=budget)
        return None

    def get_gene_row(
        self, gene_id: str, gene_format: str = "ensmbl_id"
    ) -> Optional[pd.Series]:
        """Get a row (series) of values for a specific gene across all samples.

        Args:
            gene_id: Gene identifier
            gene_format: Format of the gene ID (currently only 'ensmbl_id' supported internally)

        Returns:
            pd.Series or None: Series with sample IDs as index and gene expression values, or None if not found
        """
        matrix = self.matrix()
        if matrix is None:
            return None

        # Since our internal data is already in ensmbl_id format, we expect gene_format to be 'ensmbl_id'
        # If other formats are needed, we'd need to add translation here
        if gene_format != "ensmbl_id":
            # Try to translate to ensmbl_id format
            if self.metabolic_model:
                translated_id = self.metabolic_model.translate_gene(
                    gene_id, input_format=gene_format, output_format="ensmbl_id"
                )
                if translated_id:
                    gene_id = translated_id
            # If translation fails, continue with original ID (might still work)

        # A single feature column never needs the whole matrix dense; read it
        # through the seam so a sparse adapter can serve it from one column.
        return matrix.feature_column(gene_id)

    def get_feature_ids(self) -> List[str]:
        """Feature (gene) identifiers, without materializing the dense matrix.

        A caller that needs only the column labels -- a subsystem membership
        check, a hover existence test -- should not force the whole matrix dense
        to read them. This reads ``var_names`` from the AnnData, which carries the
        feature identity whether ``X`` is sparse or dense, and equals the dense
        frame's columns exactly (``to_df`` builds its columns from ``var_names``).
        It falls back to the dense frame only when there is no AnnData.

        This is groundwork for the AM6 sparse path: today ``build`` densifies, so
        the labels are available either way, but these readers no longer depend on
        the densification and will not force it once ``build`` preserves sparsity.
        """
        if self.adata is not None:
            return [str(name) for name in self.adata.var_names]
        matrix = self.matrix()
        if matrix is not None:
            return [str(column) for column in matrix.feature_ids]
        return []

    def get_sample_ids(self) -> List[str]:
        """Get the list of sample IDs in the expression data.

        Returns:
            List[str]: List of sample IDs.
        """
        matrix = self.matrix()
        if matrix is None:
            return []
        return matrix.sample_ids

    def get_alignment_report(self) -> Dict[str, Any]:
        """Generate an alignment report showing coverage statistics.

        Returns:
            Dict with:
                - total_input_genes: genes in input file
                - mapped_count: genes that matched model
                - unmapped_count: genes that didn't match
                - total_model_genes: metabolic genes in Human1
                - model_coverage_percent: percentage of model genes covered
        """
        report = {
            "total_input_genes": 0,
            "mapped_count": 0,
            "unmapped_count": 0,
            "total_model_genes": 0,
            "model_coverage_percent": 0.0,
        }

        # Get mapping stats if available
        if hasattr(self, "mapping_stats") and self.mapping_stats:
            report["total_input_genes"] = self.mapping_stats.get("total_input_genes", 0)
            report["mapped_count"] = self.mapping_stats.get("mapped_count", 0)
            report["unmapped_count"] = self.mapping_stats.get("unmapped_count", 0)
            report["total_model_genes"] = self.mapping_stats.get("total_model_genes", 0)

        # Calculate model coverage percentage
        if report["total_model_genes"] > 0:
            report["model_coverage_percent"] = (
                report["mapped_count"] / report["total_model_genes"]
            ) * 100

        return report
