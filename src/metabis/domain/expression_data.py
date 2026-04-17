from typing import Any, Dict, List, Optional

import pandas as pd
import scanpy as sc


class ExpressionData:
    @classmethod
    def create_with_validation(
        cls,
        path: str,
        gene_list: List[str],
        normalization_options: List[str] = None,
        gene_format: str = "ensmbl_id",
        metabolic_model=None,
        data_format: str = "csv",
        csv_separator: str = ",",
        samples_in: str = "rows",
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
                normalization_options=normalization_options,
                gene_format=gene_format,
                metabolic_model=metabolic_model,
                data_format=data_format,
                csv_separator=csv_separator,
                samples_in=samples_in,
            )

            # Perform validation after successful creation
            if hasattr(instance, "adata") and instance.adata is not None:
                # Generate alignment report
                alignment_report = instance.get_alignment_report()
                validation_results["alignment_report"] = alignment_report

                # Determine status based on mapping results
                if alignment_report["mapped_count"] == 0:
                    validation_results["status"] = "error"
                    error_msg = (
                        f"No genes found in the data file ('{path}') "
                        f"match the genes in the metabolic model "
                        f"(using input format '{gene_format}' translated to "
                        f"'ensmbl_id'). Please check the gene format and "
                        f"file content."
                    )
                    validation_results["messages"].append(error_msg)
                else:
                    # At least one gene mapped = success
                    validation_results["status"] = "success"
                    validation_results["messages"].append(
                        "Expression data loaded successfully"
                    )
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
            validation_results["messages"].append(
                f"Error loading expression data: {str(e)}"
            )
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
        normalization_options: List[str] = None,
        gene_format: str = "ensmbl_id",
        metabolic_model=None,
        data_format: str = "csv",  # Added
        csv_separator: str = ",",  # Added
        samples_in: str = "rows",  # Added
    ):
        """Initialize the ExpressionData class.

        Args:
            path: Path to expression data file
            gene_list: List of valid genes from the metabolic model
            normalization_options: List of normalization options to apply
            gene_format: Format of gene IDs in the input file
            metabolic_model: MetabolicModel instance for gene translation
            data_format (str): Format of the input data ('h5ad' or 'csv'). Default is 'csv'.
            csv_separator (str): Separator used if format is 'csv'. Default is ','.
            samples_in (str): Location of sample names in 'csv' format ('rows' or 'columns'). Default is 'rows'.
        """
        # Data storage
        self.adata = None
        self.df_expression_data = None
        self.format = None  # Will be set in build based on data_format
        self.gene_list = []
        self.metabolic_model = metabolic_model

        # File info
        self.source_path = None

        # Call build method with all parameters
        self.build(
            path,
            gene_list,
            normalization_options,
            gene_format,
            data_format,
            csv_separator,
            samples_in,
        )

    def build(
        self,
        path: str,
        gene_list: List[str],
        normalization_options: List[str],
        gene_format: str = "ensmbl_id",
        data_format: str = "csv",  # Added
        csv_separator: str = ",",  # Added
        samples_in: str = "rows",  # Added
    ):
        """Build the expression data object from the specified path.

        Args:
            path (str): Path to the expression data file.
            gene_list (List[str]): List of genes from the metabolic model (expected in ensmbl_id format).
            normalization_options (List[str]): List of normalization options to apply.
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
                adata_raw = sc.read_h5ad(path)
                # AnnData standard: samples (obs) x genes (var)
                # Check for duplicate var_names (genes)
                if adata_raw.var_names.duplicated().any():
                    print(
                        "Warning: Duplicate gene names found in h5ad file. Keeping first occurrence."
                    )
                    # Keep only the first occurrence of each gene
                    self.adata = adata_raw[
                        :, ~adata_raw.var_names.duplicated(keep="first")
                    ].copy()
                else:
                    self.adata = adata_raw

            elif data_format == "csv":
                # Load tabular data
                df_raw = pd.read_csv(path, sep=csv_separator, index_col=0)

                # --- Handle Duplicates Before AnnData Creation ---
                if samples_in == "rows":
                    # Input: rows=samples, columns=genes
                    if df_raw.columns.duplicated().any():
                        print(
                            "Warning: Duplicate gene names found in CSV columns. Keeping first occurrence."
                        )
                        df = df_raw.loc[:, ~df_raw.columns.duplicated(keep="first")]
                    else:
                        df = df_raw
                    # Create AnnData object
                    self.adata = sc.AnnData(
                        X=df.values,
                        obs=pd.DataFrame(index=df.index),  # Samples are in index
                        var=pd.DataFrame(index=df.columns),  # Genes are in columns
                    )
                elif samples_in == "columns":
                    # Input: rows=genes, columns=samples
                    if df_raw.index.duplicated().any():
                        print(
                            "Warning: Duplicate gene names found in CSV index (rows). Keeping first occurrence."
                        )
                        df = df_raw[~df_raw.index.duplicated(keep="first")]
                    else:
                        df = df_raw
                    # Create AnnData object (transpose needed)
                    self.adata = sc.AnnData(
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

                # --- Handle Duplicates *After* Translation ---
                if self.adata.var_names.duplicated().any():
                    print(
                        "Warning: Duplicate gene names found *after* translation. Keeping first occurrence."
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

            # --- Create DataFrame View ---
            # Create dataframe view (samples x genes) from the final filtered/translated adata
            self.df_expression_data = self.adata.to_df()

            # --- Normalization ---
            # Apply normalizations (operates on self.adata inplace)
            self._apply_normalizations(normalization_options)
            # Update the dataframe view if normalization changed the data
            if normalization_options:
                self.df_expression_data = self.adata.to_df()

        except FileNotFoundError:
            raise FileNotFoundError(f"Expression data file not found: {path}")
        except Exception as e:
            # Improve error message context
            import traceback

            tb_str = traceback.format_exc()
            raise ValueError(
                f"Error loading/processing expression data from '{path}' (format: {data_format}, gene_fmt: {gene_format}): {str(e)}\nTraceback:\n{tb_str}"
            )

    def _apply_normalizations(self, normalization_options: List[str]) -> None:
        """Apply normalization methods to the expression data.

        Args:
            normalization_options (List[str]): List of normalization options to apply.

        Returns:
            None
        """
        if not normalization_options:
            return

        for normalization in normalization_options:
            if normalization.lower() == "cpm":
                # Counts per million normalization using scanpy
                sc.pp.normalize_total(self.adata, target_sum=1e6, inplace=True)
                # Update the transposed view
                self.df_expression_data = self.adata.to_df()

            elif normalization.lower() == "log":
                # Log transformation (log1p to handle zeros)
                sc.pp.log1p(self.adata)
                # Update the transposed view
                self.df_expression_data = self.adata.to_df()

            else:
                print(f"Warning: Unknown normalization option: {normalization}")

    def get_expression_df(self) -> pd.DataFrame:
        """Get the expression data as a pandas DataFrame (samples x genes).

        Returns:
            pd.DataFrame: Expression data as a DataFrame.
        """
        if self.df_expression_data is None:
            # Check if adata exists but df is None (e.g., error during build)
            if self.adata is not None:
                # Attempt to create it now
                self.df_expression_data = self.adata.to_df()
                if self.df_expression_data is not None:
                    return self.df_expression_data
            raise ValueError(
                "Expression data not loaded or processed correctly. Call build() first."
            )

        return self.df_expression_data

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
        if self.df_expression_data is None:
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

        if gene_id not in self.df_expression_data.columns:
            return None

        # Return the column as a Series (samples x values)
        return self.df_expression_data[gene_id]
    
    def get_sample_ids(self) -> List[str]:
        """Get the list of sample IDs in the expression data.

        Returns:
            List[str]: List of sample IDs.
        """
        if self.df_expression_data is None:
            return []
        return self.df_expression_data.index.tolist()

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
