"""Pydantic models for optimization API requests and responses."""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RankingMetric(str, Enum):
    """Metrics available for ranking clusters."""
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    PROFIT_FACTOR = "profit_factor"


class ShortlistCondition(BaseModel):
    """A single condition for shortlisting variants."""
    metric: str = Field(..., description="Metric column name to filter on")
    operator: Literal[">", ">=", "<", "<=", "=="] = Field(..., description="Comparison operator")
    value: float = Field(..., description="Threshold value for comparison")


class ShortlistConfig(BaseModel):
    """Configuration for optional shortlisting/filtering of variants."""
    enabled: bool = Field(default=False, description="Whether shortlisting is enabled")
    conditions: List[ShortlistCondition] = Field(
        default_factory=list,
        description="List of conditions to apply (combined with AND logic)"
    )


class HDBSCANConfig(BaseModel):
    """Configuration for HDBSCAN clustering."""
    min_cluster_size: int = Field(default=5, ge=2, description="Minimum cluster size")
    min_samples: int = Field(default=3, ge=1, description="Minimum samples for core points")


class HeatmapParams(BaseModel):
    """
    Parameter mapping for heatmap visualization.

    - 2 params: x_param, y_param only (const_param1 and const_param2 = None)
    - 3 params: x_param, y_param, const_param1 (const_param2 = None)
    - 4 params: x_param, y_param, const_param1, const_param2
    """
    x_param: str = Field(..., description="Parameter for heatmap X-axis")
    y_param: str = Field(..., description="Parameter for heatmap Y-axis")
    const_param1: Optional[str] = Field(
        default=None,
        description="First constant parameter (creates separate heatmaps per value)"
    )
    const_param2: Optional[str] = Field(
        default=None,
        description="Second constant parameter (creates grid of heatmaps)"
    )


class OptimizationRequest(BaseModel):
    """Request model for running the optimization pipeline."""
    csv_path: str = Field(..., description="Path to the backtest results CSV file")
    strategy_params: List[str] = Field(
        ...,
        min_length=2,
        max_length=4,
        description="List of strategy parameter column names (2-4 params)"
    )
    shortlist_config: Optional[ShortlistConfig] = Field(
        default=None,
        description="Optional configuration for shortlisting variants"
    )
    kmeans_k: Optional[int] = Field(
        default=None,
        ge=2,
        description="Number of K-Means clusters. If None, auto-calculated as 20"
    )
    hdbscan_config: HDBSCANConfig = Field(
        default_factory=HDBSCANConfig,
        description="HDBSCAN clustering configuration"
    )
    num_best_clusters: int = Field(
        default=2,
        ge=1,
        description="Number of top clusters to return"
    )
    ranking_metric: RankingMetric = Field(
        ...,
        description="Metric used to rank clusters (sharpe_ratio, sortino_ratio, profit_factor)"
    )
    heatmap_params: HeatmapParams = Field(
        ...,
        description="Parameter mapping for heatmap axes"
    )


class ClusterStats(BaseModel):
    """Statistics for a single cluster."""
    cluster_id: int
    count: int
    median: float
    mean: float
    std: float


class OptimizationResponse(BaseModel):
    """Response model containing all optimization results."""
    # Heatmaps
    initial_heatmaps: List[Dict[str, Any]] = Field(
        ..., description="Initial parameter landscape heatmaps (Plotly JSON)"
    )
    shortlisted_heatmaps: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Heatmaps with shortlisted variants highlighted (if enabled)"
    )

    # PCA visualizations
    pca_variance_chart: Dict[str, Any] = Field(
        ..., description="PCA explained variance bar chart (Plotly JSON)"
    )
    pca_scatter: Dict[str, Any] = Field(
        ..., description="PCA scatter plot PC1 vs PC2 (Plotly JSON)"
    )

    # K-Means results
    kmeans_scatter: Dict[str, Any] = Field(
        ..., description="K-Means cluster scatter plot (Plotly JSON)"
    )
    kmeans_cluster_stats: List[Dict[str, Any]] = Field(
        ..., description="K-Means cluster statistics"
    )

    # HDBSCAN results
    hdbscan_scatter: Dict[str, Any] = Field(
        ..., description="HDBSCAN cluster scatter plot (Plotly JSON)"
    )
    hdbscan_cluster_stats: List[Dict[str, Any]] = Field(
        ..., description="HDBSCAN cluster statistics"
    )

    # Final outputs
    final_heatmaps: List[List[Dict[str, Any]]] = Field(
        ..., description="Heatmaps highlighting each best cluster"
    )
    best_clusters_data: List[List[Dict[str, Any]]] = Field(
        ..., description="Variant data for each best cluster"
    )

    # Metadata
    num_variants_processed: int = Field(..., description="Total variants in input")
    num_variants_after_shortlist: int = Field(..., description="Variants after shortlisting")
    best_cluster_ids: List[int] = Field(..., description="IDs of the best clusters")


# ============== Step-by-Step Workflow Models ==============

# Step 1: Load Data - show head and info
class StepLoadDataRequest(BaseModel):
    """Request for Step 1: Load CSV and show data preview."""
    csv_paths: List[str] = Field(
        ...,
        min_length=1,
        description="List of paths to backtest results CSV files (will be concatenated)"
    )
    strategy_params: List[str] = Field(
        ...,
        min_length=2,
        max_length=4,
        description="List of strategy parameter column names (2-4 params)"
    )


class ColumnInfo(BaseModel):
    """Information about a DataFrame column."""
    name: str
    dtype: str
    non_null_count: int
    null_count: int


class StepLoadDataResponse(BaseModel):
    """Response for Step 1: Data preview and info."""
    session_id: str = Field(..., description="Session ID for subsequent steps")
    num_files: int = Field(..., description="Number of CSV files loaded")
    num_rows: int = Field(..., description="Total number of rows")
    num_columns: int = Field(..., description="Total number of columns")
    columns: List[str] = Field(..., description="All column names")
    column_info: List[ColumnInfo] = Field(..., description="Column dtype and null info")
    head_data: List[Dict[str, Any]] = Field(..., description="First 10 rows of data")
    strategy_params: List[str] = Field(..., description="Selected strategy parameters")
    available_metrics: List[str] = Field(..., description="Available metric columns")


# Step 2: Generate Heatmaps (configurable, can be called multiple times)
class StepHeatmapRequest(BaseModel):
    """Request for Step 2: Generate heatmaps with configurable options."""
    session_id: str = Field(..., description="Session ID from Step 1")
    x_param: str = Field(..., description="Parameter for X-axis")
    y_param: str = Field(..., description="Parameter for Y-axis")
    const_param: Optional[str] = Field(
        default=None,
        description="Constant parameter (creates separate heatmaps per value)"
    )
    const_values: Optional[List[Any]] = Field(
        default=None,
        description="Specific values of const_param to show (None = all)"
    )
    metrics: Optional[List[str]] = Field(
        default=None,
        description="Metrics to display (None = all available)"
    )
    show_shortlisted: bool = Field(
        default=False,
        description="Highlight shortlisted variants if shortlist is applied"
    )


class StepHeatmapResponse(BaseModel):
    """Response for Step 2: Generated heatmaps."""
    session_id: str
    heatmaps: List[Dict[str, Any]] = Field(
        ..., description="Heatmaps (Plotly JSON)"
    )
    num_heatmaps: int = Field(..., description="Number of heatmaps generated")
    x_param: str
    y_param: str
    const_param: Optional[str]
    metrics_shown: List[str]


# Step 2b: Apply Shortlist (optional, can be called before heatmaps)
class StepShortlistRequest(BaseModel):
    """Request for Step 2b: Apply shortlisting conditions."""
    session_id: str = Field(..., description="Session ID from Step 1")
    shortlist_config: Optional[ShortlistConfig] = Field(
        default=None,
        description="Optional configuration for shortlisting variants"
    )


class StepShortlistResponse(BaseModel):
    """Response for Step 2."""
    session_id: str
    shortlisted_heatmaps: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Heatmaps with shortlisted variants highlighted"
    )
    num_variants_after_shortlist: int
    shortlist_applied: bool


class StepPCARequest(BaseModel):
    """Request for Step 3: Perform PCA."""
    session_id: str = Field(..., description="Session ID from previous step")


class StepPCAResponse(BaseModel):
    """Response for Step 3."""
    session_id: str
    pca_variance_chart: Dict[str, Any] = Field(
        ..., description="PCA explained variance chart (Plotly JSON)"
    )
    pca_scatter: Dict[str, Any] = Field(
        ..., description="PCA scatter plot PC1 vs PC2 (Plotly JSON)"
    )
    explained_variance_ratio: List[float] = Field(
        ..., description="Explained variance ratio per component"
    )


class StepKMeansRequest(BaseModel):
    """Request for Step 4: K-Means clustering."""
    session_id: str = Field(..., description="Session ID from previous step")
    k: Optional[int] = Field(
        default=None,
        ge=2,
        description="Number of clusters. If None, auto-calculated"
    )


class StepKMeansResponse(BaseModel):
    """Response for Step 4."""
    session_id: str
    kmeans_scatter: Dict[str, Any] = Field(
        ..., description="K-Means cluster scatter plot (Plotly JSON)"
    )
    kmeans_cluster_stats: List[Dict[str, Any]] = Field(
        ..., description="K-Means cluster statistics for default metric (sharpe_ratio)"
    )
    k_used: int = Field(..., description="Number of clusters used")
    num_variants_in_best_kmeans: int = Field(
        ..., description="Variants in best K-Means cluster (filtered for HDBSCAN)"
    )
    all_cluster_stats: Optional[Dict[str, List[Dict[str, Any]]]] = Field(
        default=None,
        description="Cluster statistics for all metrics (sharpe_ratio, sortino_ratio, profit_factor)"
    )


class HDBSCANGridConfig(BaseModel):
    """Configuration for HDBSCAN grid search."""
    min_cluster_sizes: List[int] = Field(
        default=[3, 5, 10, 15, 20],
        description="List of min_cluster_size values to try"
    )
    min_sample_sizes: List[int] = Field(
        default=[3, 5, 10, 15, 20],
        description="List of min_samples values to try"
    )
    threshold_cluster_prob: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Probability threshold for core cluster members"
    )


class StepHDBSCANGridRequest(BaseModel):
    """Request for Step 5: HDBSCAN grid search."""
    session_id: str = Field(..., description="Session ID from previous step")
    grid_config: HDBSCANGridConfig = Field(
        default_factory=HDBSCANGridConfig,
        description="HDBSCAN grid search configuration"
    )
    ranking_metric: RankingMetric = Field(
        default=RankingMetric.SHARPE_RATIO,
        description="Metric for ranking clusters"
    )


class HDBSCANConfigResult(BaseModel):
    """Result for a single HDBSCAN configuration."""
    min_cluster_size: int
    min_samples: int
    num_clusters: int = Field(..., description="Number of clusters found (excluding noise)")
    cluster_stats: List[Dict[str, Any]]


class StepHDBSCANGridResponse(BaseModel):
    """Response for Step 5."""
    session_id: str
    hdbscan_grid_chart: Dict[str, Any] = Field(
        ..., description="Grid of HDBSCAN scatter plots (Plotly JSON)"
    )
    hdbscan_core_grid_chart: Dict[str, Any] = Field(
        ..., description="Grid of HDBSCAN core points scatter plots (Plotly JSON)"
    )
    config_results: List[HDBSCANConfigResult] = Field(
        ..., description="Results for each configuration"
    )
    available_configs: List[List[int]] = Field(
        ..., description="List of [min_cluster_size, min_samples] pairs"
    )


class StepHDBSCANFinalRequest(BaseModel):
    """Request for Step 6: Final HDBSCAN with selected config."""
    session_id: str = Field(..., description="Session ID from previous step")
    min_cluster_size: int = Field(..., ge=2, description="Selected min_cluster_size")
    min_samples: int = Field(..., ge=1, description="Selected min_samples")
    ranking_metric: RankingMetric = Field(
        default=RankingMetric.SHARPE_RATIO,
        description="Metric for ranking clusters"
    )


class StepHDBSCANFinalResponse(BaseModel):
    """Response for Step 6."""
    session_id: str
    hdbscan_scatter: Dict[str, Any] = Field(
        ..., description="Final HDBSCAN cluster scatter plot - all points (Plotly JSON)"
    )
    hdbscan_core_scatter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Final HDBSCAN cluster scatter plot - core points only (Plotly JSON)"
    )
    hdbscan_cluster_stats: List[Dict[str, Any]] = Field(
        ..., description="HDBSCAN cluster statistics"
    )
    num_clusters: int = Field(..., description="Number of clusters found")
    num_core_points: Optional[int] = Field(
        default=None, description="Number of core points (high probability)"
    )


class StepBestClustersRequest(BaseModel):
    """Request for Step 7: Get best clusters and final heatmaps."""
    session_id: str = Field(..., description="Session ID from previous step")
    num_best_clusters: int = Field(
        default=2,
        ge=1,
        description="Number of top clusters to return"
    )
    ranking_metric: RankingMetric = Field(
        default=RankingMetric.SHARPE_RATIO,
        description="Metric for ranking clusters"
    )


class StepBestClustersResponse(BaseModel):
    """Response for Step 7."""
    session_id: str
    final_heatmaps: List[List[Dict[str, Any]]] = Field(
        ..., description="Heatmaps highlighting each best cluster (all cluster variants)"
    )
    final_core_heatmaps: Optional[List[List[Dict[str, Any]]]] = Field(
        default=None,
        description="Heatmaps highlighting only core cluster variants (high probability)"
    )
    best_clusters_data: List[List[Dict[str, Any]]] = Field(
        ..., description="Variant data for each best cluster"
    )
    best_cluster_ids: List[int] = Field(
        ..., description="IDs of the best clusters"
    )
    cluster_const_values: Optional[List[List[Any]]] = Field(
        default=None,
        description="Constant parameter values present in each cluster (for dropdown filtering)"
    )
    cluster_core_counts: Optional[List[int]] = Field(
        default=None,
        description="Number of core points in each cluster"
    )


# ============== Step 8: Report Generation Models ==============

class ReportShortlistCondition(BaseModel):
    """Shortlist condition for report generation."""
    metric: str
    operator: Literal[">", ">=", "<", "<=", "=="]
    value: float


class StepGenerateReportRequest(BaseModel):
    """Request for Step 8: Generate PDF report with all configurations."""
    session_id: str = Field(..., description="Session ID from previous steps")

    # Report metadata
    report_title: str = Field(..., description="Custom title for the report (e.g., 'NIFTY 50 - Momentum Strategy Optimization')")

    # Step 1: Load Data config
    csv_path: str = Field(..., description="Path to CSV file")
    strategy_params: List[str] = Field(..., min_length=2, max_length=4, description="Strategy parameter columns")

    # Step 2: Heatmap config
    x_param: str = Field(..., description="X-axis parameter for heatmaps")
    y_param: str = Field(..., description="Y-axis parameter for heatmaps")
    const_param: Optional[str] = Field(default=None, description="Constant parameter for heatmaps")

    # Step 2b: Shortlist config
    shortlist_enabled: bool = Field(default=False, description="Whether shortlisting is enabled")
    shortlist_conditions: List[ReportShortlistCondition] = Field(
        default_factory=list,
        description="Shortlist conditions if enabled"
    )

    # Step 4: K-Means config
    kmeans_k: Optional[int] = Field(default=None, ge=2, description="K value (None for auto)")

    # Step 5-6: HDBSCAN config
    hdbscan_min_cluster_size: int = Field(..., ge=2, description="HDBSCAN min_cluster_size")
    hdbscan_min_samples: int = Field(..., ge=1, description="HDBSCAN min_samples")
    hdbscan_grid_min_sizes: List[int] = Field(
        default=[3, 5, 10, 15, 20],
        description="Min cluster sizes for grid search"
    )
    hdbscan_grid_min_samples: List[int] = Field(
        default=[3, 5, 10, 15, 20],
        description="Min samples for grid search"
    )
    core_probability_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Core point probability threshold")

    # Step 7: Best Clusters config
    num_best_clusters: int = Field(default=2, ge=1, description="Number of best clusters to show")

    # Output config
    report_filename: str = Field(
        default="optimization_report.pdf",
        description="Filename for the PDF report (without path)"
    )
    save_path: str = Field(
        ...,
        description="Directory path to save the report"
    )


class StepGenerateReportResponse(BaseModel):
    """Response for Step 8: Generated report."""
    session_id: str
    report_generated: bool = Field(..., description="Whether report was generated successfully")
    report_path: Optional[str] = Field(default=None, description="Path where report was saved (if save_path provided)")
    download_available: bool = Field(default=False, description="Whether report is available for download")
    report_filename: str = Field(..., description="Generated filename for the report")
    message: str = Field(..., description="Status message")
