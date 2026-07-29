#!/usr/bin/env python3
"""
Script to add cell type information to sample names in reactions.tsv and linear_gene_expression_matrix.tsv files.

This script reads cell_metadata.csv to create a mapping of cell_id to celltype_cell_id,
then updates the column headers in both TSV files to include cell type information.
"""

import sys
from pathlib import Path

import pandas as pd


def load_cell_metadata(metadata_file):
    """
    Load cell metadata and create mapping from cell_id to celltype_cell_id.

    Args:
        metadata_file (str): Path to cell_metadata.csv file

    Returns:
        dict: Mapping from original cell_id to new celltype_cell_id format
    """
    print(f"Loading cell metadata from: {metadata_file}")

    try:
        metadata_df = pd.read_csv(metadata_file)
        print(f"Loaded metadata for {len(metadata_df)} cells")

        # Create mapping dictionary
        cell_mapping = {}
        for _, row in metadata_df.iterrows():
            original_id = row['cell_id']
            cell_type = row['cell_type']
            new_id = f"{cell_type}_{original_id}"
            cell_mapping[original_id] = new_id

        print(f"Created mapping for {len(cell_mapping)} cell IDs")
        print(f"Example mapping: {list(cell_mapping.items())[0]}")

        return cell_mapping

    except Exception as e:
        print(f"Error loading metadata: {e}")
        sys.exit(1)

def process_reactions_file(reactions_file, cell_mapping, output_file):
    """
    Process reactions.tsv file and update column headers with cell type information.

    Args:
        reactions_file (str): Path to input reactions.tsv file
        cell_mapping (dict): Mapping from cell_id to celltype_cell_id
        output_file (str): Path to output file
    """
    print(f"\nProcessing reactions file: {reactions_file}")

    try:
        # Load the reactions file with tab delimiter
        reactions_df = pd.read_csv(reactions_file, sep='\t', index_col=0)
        print(f"Loaded reactions data with shape: {reactions_df.shape}")

        # Get current column names
        original_columns = reactions_df.columns.tolist()
        print(f"Original columns: {len(original_columns)} samples")

        # Create new column names using the mapping
        new_columns = []
        unmapped_columns = []

        for col in original_columns:
            if col in cell_mapping:
                new_columns.append(cell_mapping[col])
            else:
                new_columns.append(col)  # Keep original if not found in mapping
                unmapped_columns.append(col)

        if unmapped_columns:
            print(f"Warning: {len(unmapped_columns)} columns not found in metadata mapping")
            print(f"First few unmapped: {unmapped_columns[:5]}")

        # Update column names
        reactions_df.columns = new_columns

        # Save the updated file
        reactions_df.to_csv(output_file, sep='\t')
        print(f"Saved updated reactions file to: {output_file}")
        print(f"Example new column names: {new_columns[:3]}")

    except Exception as e:
        print(f"Error processing reactions file: {e}")
        sys.exit(1)

def process_expression_file(expression_file, cell_mapping, output_file):
    """
    Process linear_gene_expression_matrix.tsv file and update column headers with cell type information.

    Args:
        expression_file (str): Path to input expression matrix file
        cell_mapping (dict): Mapping from cell_id to celltype_cell_id
        output_file (str): Path to output file
    """
    print(f"\nProcessing expression file: {expression_file}")

    try:
        # Load the expression file with tab delimiter
        expression_df = pd.read_csv(expression_file, sep='\t', index_col=0)
        print(f"Loaded expression data with shape: {expression_df.shape}")

        # Get current column names
        original_columns = expression_df.columns.tolist()
        print(f"Original columns: {len(original_columns)} samples")

        # Create new column names using the mapping
        new_columns = []
        unmapped_columns = []

        for col in original_columns:
            if col in cell_mapping:
                new_columns.append(cell_mapping[col])
            else:
                new_columns.append(col)  # Keep original if not found in mapping
                unmapped_columns.append(col)

        if unmapped_columns:
            print(f"Warning: {len(unmapped_columns)} columns not found in metadata mapping")
            print(f"First few unmapped: {unmapped_columns[:5]}")

        # Update column names
        expression_df.columns = new_columns

        # Save the updated file
        expression_df.to_csv(output_file, sep='\t')
        print(f"Saved updated expression file to: {output_file}")
        print(f"Example new column names: {new_columns[:3]}")

    except Exception as e:
        print(f"Error processing expression file: {e}")
        sys.exit(1)

def main():
    """Main function to process both files."""

    # Define file paths
    data_dir = Path(__file__).resolve().parent
    metadata_file = data_dir / "cell_metadata.csv"
    reactions_file = data_dir / "reactions.tsv"
    expression_file = data_dir / "linear_gene_expression_matrix.tsv"

    # Define output file paths
    reactions_output = data_dir / "reactions_with_celltype.tsv"
    expression_output = data_dir / "linear_gene_expression_matrix_with_celltype.tsv"

    # Check if input files exist
    for file_path in [metadata_file, reactions_file, expression_file]:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)

    print("=== Adding Cell Type Information to Sample Names ===")
    print(f"Data directory: {data_dir}")

    # Step 1: Load cell metadata and create mapping
    cell_mapping = load_cell_metadata(metadata_file)

    # Step 2: Process reactions file
    process_reactions_file(reactions_file, cell_mapping, reactions_output)

    # Step 3: Process expression file
    process_expression_file(expression_file, cell_mapping, expression_output)

    print("\n=== Processing Complete ===")
    print("Created files:")
    print(f"  - {reactions_output}")
    print(f"  - {expression_output}")

    # Print summary statistics
    print("\nSummary:")
    print(f"  - Total cell mappings: {len(cell_mapping)}")
    print(f"  - Cell types found: {len(set(mapping.split('_')[0] for mapping in cell_mapping.values()))}")

    # Show unique cell types
    cell_types = set(mapping.split('_')[0] for mapping in cell_mapping.values())
    print(f"  - Cell types: {', '.join(sorted(cell_types))}")

if __name__ == "__main__":
    main()
