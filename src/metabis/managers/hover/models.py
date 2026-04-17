from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from ..data_provider import DataProvider
    from ..color_manager import ColorManager
    from ..group_comparison import GroupComparison
    from ..data_manager_mode import DataManagerMode

@dataclass
class HoverContext:
    """
    Context object containing view configuration, environment settings, 
    and services required for generating hover text.
    """
    # View Configuration
    mode: "DataManagerMode"
    hover_type: str  # "gene", "reaction", "metabolite", "subsystem"
    element_type: str # "gene", "reaction", ...
    element_id: Optional[str] = None
    
    # Visualization Settings
    main_direction: str = "forward"
    subsystem_metric: str = "log2_fc"
    color_metric: str = "log2_fc"
    
    # Toggles
    show_plot: bool = False
    show_stats: bool = False
    show_influence: bool = False
    show_color_reference: bool = False
    show_comparison: bool = False
    
    # Context-specific State
    current_subsystem: Optional[str] = None
    current_category: Optional[str] = None
    for_genes: bool = False
    
    # Services & Dependencies
    data_provider: Optional["DataProvider"] = None
    color_manager: Optional["ColorManager"] = None
    group_comparison: Optional["GroupComparison"] = None
    
    # Extra
    analysis_type: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        """Backward compatibility method to allow dict-like access."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Backward compatibility method to allow dict-like access."""
        return getattr(self, key)

class ComparisonStatsData(TypedDict, total=False):
    has_comparison_data: bool
    group1: str
    group2: str
    forward: Dict[str, Any]
    backward: Dict[str, Any]
    is_significant: bool

class ExperimentStatsData(TypedDict, total=False):
    has_experiment_data: bool
    group_id: str
    z_score: float
    average_activity: float
    element_type: str

class DistributionData(TypedDict, total=False):
    has_distributions: bool
    group1_values: List[float]
    group2_values: List[float]
    group1_name: str
    group2_name: str
    backward: Optional['DistributionData']

class InfluenceGeneData(TypedDict, total=False):
    gene_symbol: str
    correlation: float
    pct_variance: float

class InfluenceData(TypedDict, total=False):
    has_influence: bool
    top_genes: List[InfluenceGeneData]
    max_influence: float

class HoverData(TypedDict, total=False):
    """
    Typed dictionary representing the data structure passed to hover builders.
    """
    # Common
    is_in_model: bool
    metadata: Dict[str, Any]
    
    # Identifiers
    gene_id: str
    reaction_id: str
    metabolite_id: str
    primary_id: str
    
    # Display Info
    symbol: str
    name: str
    label: str
    description: str
    equation: str
    
    # Categorization
    subsystem: str
    subsystem_category: str
    consumed_by_subsystems: List[str]
    produced_by_subsystems: List[str]
    
    # Analysis Data
    comparison_stats: ComparisonStatsData
    experiment_stats: ExperimentStatsData
    distributions: DistributionData
    influence: InfluenceData
    
    # Subsystem Specific
    subsystem_display_name: str
    n_significant: int
    n_total: int
    regulation_score: float
    total_metric_value: float
    delta_subsystem_score: Optional[float]
    delta_subsystem_meta: Dict[str, Any]

