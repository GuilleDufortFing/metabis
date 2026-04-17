from .builder import HoverTextBuilder
from .sections.gene_basic_info import GeneBasicInfoSection
from .sections.reaction_basic_info import ReactionBasicInfoSection
from .sections.metabolite_basic_info import MetaboliteBasicInfoSection
from .sections.subsystem_info import SubsystemSection
from .sections.statistics import StatisticsSection
from .sections.influence import InfluenceSection
from .sections.distribution_plot import DistributionPlotSection
from .sections.experiment_analysis import ExperimentAnalysisSection
from .sections.color_explanation import ColorExplanationSection
from .sections.no_experiment import NoExperimentSection
from ..data_manager_mode import DataManagerMode

def create_gene_builder_for_mode(mode: DataManagerMode) -> HoverTextBuilder:
    """Create a gene hover builder optimized for the specified mode."""
    builder = HoverTextBuilder().add_section(GeneBasicInfoSection())

    if mode == DataManagerMode.GROUP_COMPARISON:
        builder.add_section(StatisticsSection("gene"))
        builder.add_section(DistributionPlotSection("gene"))
        builder.add_section(ColorExplanationSection())
    elif mode == DataManagerMode.EXPERIMENT_ANALYSIS:
        builder.add_section(ExperimentAnalysisSection())
        builder.add_section(DistributionPlotSection("gene"))
        builder.add_section(ColorExplanationSection())
    else:  # NO_EXPERIMENT
        builder.add_section(NoExperimentSection())

    return builder

def create_reaction_builder_for_mode(mode: DataManagerMode) -> HoverTextBuilder:
    """Create a reaction hover builder optimized for the specified mode."""
    builder = HoverTextBuilder().add_section(ReactionBasicInfoSection())

    if mode == DataManagerMode.GROUP_COMPARISON:
        builder.add_section(StatisticsSection("reaction"))
        builder.add_section(InfluenceSection())
        builder.add_section(DistributionPlotSection("reaction"))
        builder.add_section(ColorExplanationSection())
    elif mode == DataManagerMode.EXPERIMENT_ANALYSIS:
        builder.add_section(ExperimentAnalysisSection())
        # builder.add_section(DistributionPlotSection("reaction"))
        builder.add_section(ColorExplanationSection())
    else:  # NO_EXPERIMENT
        builder.add_section(NoExperimentSection())

    return builder

def create_metabolite_builder_for_mode(mode: DataManagerMode) -> HoverTextBuilder:
    """Create a metabolite hover builder optimized for the specified mode."""
    builder = HoverTextBuilder().add_section(MetaboliteBasicInfoSection())

    if mode == DataManagerMode.GROUP_COMPARISON:
        # Metabolites could have statistics in the future
        pass
    elif mode == DataManagerMode.EXPERIMENT_ANALYSIS:
        # Metabolites could have experiment analysis in the future
        pass
    else:  # NO_EXPERIMENT
        builder.add_section(NoExperimentSection())

    return builder

def create_subsystem_builder_for_mode(mode: DataManagerMode) -> HoverTextBuilder:
    """Create a subsystem hover builder optimized for the specified mode."""
    builder = HoverTextBuilder().add_section(SubsystemSection())

    # Subsystem builders are currently only used in GROUP_COMPARISON mode
    # But we prepare for future expansion
    if mode == DataManagerMode.NO_EXPERIMENT:
        builder.add_section(NoExperimentSection())
    elif mode == DataManagerMode.EXPERIMENT_ANALYSIS:
        # Add EA stats plus color explanation
        builder.add_section(ExperimentAnalysisSection())
        builder.add_section(ColorExplanationSection())
    else:
        # Add unified color explanation for subsystems in active modes
        builder.add_section(ColorExplanationSection())

    return builder
