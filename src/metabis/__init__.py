"""Init file for metabolic_model module."""
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


from .analysis.experiment_analysis import ExperimentAnalysis
from .analysis.group_comparison import GroupComparison
from .domain.expression_data import ExpressionData
from .domain.format_detection import (
    detect_expression_csv_params,
    detect_reaction_csv_params,
)
from .domain.metabolic_model import MetabolicModel
from .domain.metabolic_model.loaders import check_subsystem_categories_file
from .domain.reaction_data import ReactionData
from .managers.data_manager import DataManager, DataManagerMode
from .viz.plot_gene_reaction_influence import (
    plot_gene_reaction_influence_heatmap,
    plot_subsystem_influence_summary,
    plot_subsystem_ranking_by_influence,
)
from .viz.plot_metabolic_graph import (
    plot_metabolic_graph,
)
from .viz.plot_reduced_dimension import (
    plot_pca_biplot,
    plot_pca_explained_variance,
    plot_pca_loadings_heatmap,
    plot_reduced_dimension,
)
from .viz.plot_statistical_distributions import (
    create_statistical_results_export_bundle,
    create_statistical_results_table,
    plot_statistical_histograms,
    plot_statistical_scatterplots,
)
from .viz.plot_subsystem_ranking import (
    plot_subsystem_comparison,
)
from .viz.plot_utils import apply_academic_style
from .viz.plot_violins import plot_reaction_violins
from .viz.table_formatting import build_external_link_formatter


