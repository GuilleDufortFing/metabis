from ..workspace_mode import AnalysisWorkspaceMode
from .builder import HoverTextBuilder
from .sections.activity import ActivitySection
from .sections.activity_distribution import ActivityDistributionSection
from .sections.blocked_reaction import BlockedReactionSection
from .sections.color_explanation import ColorExplanationSection
from .sections.distribution_plot import DistributionPlotSection
from .sections.gene_basic_info import GeneBasicInfoSection
from .sections.influence import InfluenceSection
from .sections.metabolite_basic_info import MetaboliteBasicInfoSection
from .sections.no_experiment import NoExperimentSection
from .sections.reaction_basic_info import ReactionBasicInfoSection
from .sections.statistics import StatisticsSection
from .sections.subsystem_info import SubsystemSection


def create_gene_builder_for_mode(mode: AnalysisWorkspaceMode) -> HoverTextBuilder:
    """Create a gene hover builder optimized for the specified mode."""
    builder = HoverTextBuilder().add_section(GeneBasicInfoSection())

    if mode == AnalysisWorkspaceMode.GROUP_COMPARISON:
        builder.add_section(StatisticsSection("gene"))
        builder.add_section(DistributionPlotSection("gene"))
        builder.add_section(ColorExplanationSection())
    elif mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS:
        builder.add_section(ActivitySection())
        builder.add_section(ActivityDistributionSection("gene"))
        builder.add_section(ColorExplanationSection())
    else:  # NO_EXPERIMENT
        builder.add_section(NoExperimentSection())

    return builder

def create_reaction_builder_for_mode(mode: AnalysisWorkspaceMode) -> HoverTextBuilder:
    """Create a reaction hover builder optimized for the specified mode."""
    builder = HoverTextBuilder().add_section(ReactionBasicInfoSection())
    # Before the statistics, because it explains why some of them are absent.
    builder.add_section(BlockedReactionSection())

    if mode == AnalysisWorkspaceMode.GROUP_COMPARISON:
        builder.add_section(StatisticsSection("reaction"))
        builder.add_section(InfluenceSection())
        builder.add_section(DistributionPlotSection("reaction"))
        builder.add_section(ColorExplanationSection())
    elif mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS:
        builder.add_section(ActivitySection())
        builder.add_section(ActivityDistributionSection("reaction"))
        builder.add_section(ColorExplanationSection())
    else:  # NO_EXPERIMENT
        builder.add_section(NoExperimentSection())

    return builder

def create_metabolite_builder_for_mode(mode: AnalysisWorkspaceMode) -> HoverTextBuilder:
    """Create a metabolite hover builder optimized for the specified mode."""
    builder = HoverTextBuilder().add_section(MetaboliteBasicInfoSection())

    if mode == AnalysisWorkspaceMode.GROUP_COMPARISON:
        # Metabolites could have statistics in the future
        pass
    elif mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS:
        # Metabolites could have experiment analysis in the future
        pass
    else:  # NO_EXPERIMENT
        builder.add_section(NoExperimentSection())

    return builder

def create_subsystem_builder_for_mode(mode: AnalysisWorkspaceMode) -> HoverTextBuilder:
    """Create a subsystem hover builder optimized for the specified mode."""
    builder = HoverTextBuilder().add_section(SubsystemSection())

    # Subsystem builders are currently only used in GROUP_COMPARISON mode
    # But we prepare for future expansion
    if mode == AnalysisWorkspaceMode.NO_EXPERIMENT:
        builder.add_section(NoExperimentSection())
    elif mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS:
        # Add EA stats plus color explanation
        builder.add_section(ActivitySection())
        builder.add_section(ColorExplanationSection())
    else:
        # Add unified color explanation for subsystems in active modes
        builder.add_section(ColorExplanationSection())

    return builder
