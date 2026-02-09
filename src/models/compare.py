"""Pydantic models for the Compare Variants feature."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChartType(str, Enum):
    """Available chart types for comparison visualization."""

    TABLE = "table"
    GROUPED_BAR = "grouped_bar"
    RADAR = "radar"
    RANKING_TABLE = "ranking_table"
    LINE = "line"  # Cumulative returns line chart
    # New visualization types
    DRAWDOWN = "drawdown"
    ROLLING_HEATMAP = "rolling_heatmap"
    RISK_RETURN_SCATTER = "risk_return_scatter"
    MONTHLY_HEATMAP = "monthly_heatmap"
    UNDERWATER = "underwater"


class VariantInfo(BaseModel):
    """Information about a loaded variant."""

    name: str = Field(..., description="Variant name (derived from filename)")
    file_path: str = Field(..., description="Full path to the CSV file")
    num_trades: int = Field(..., description="Number of trade records in the file")


class VariantMetrics(BaseModel):
    """Calculated performance metrics for a variant."""

    name: str = Field(..., description="Variant name")
    # Return metrics
    total_pnl: float = Field(..., description="Total PnL% = sum(PnL%)")
    avg_pnl: float = Field(..., description="Avg PnL% = Total PnL% / Total weeks")
    avg_annual_roi: float = Field(..., description="Avg Annual ROI = Avg PnL% * 52")
    last_52w_pnl: float = Field(..., description="Last 52W PnL% = PnL[-52:].sum()")
    # Volatility metrics
    annualized_sd: float = Field(..., description="Annualized SD = std(PnL%) * sqrt(52)")
    negative_annualized_sd: float = Field(..., description="Negative Ann SD = std(PnL[PnL<0]) * sqrt(52)")
    # Drawdown metrics
    max_drawdown: float = Field(..., description="Max DD = max(CUMMAX - CUMSUM)")
    ulcer_index: float = Field(..., description="Ulcer Index = std(CUMMAX - CUMSUM)")
    var_5pct: float = Field(..., description="VaR 5% = percentile(PnL, 5)")
    expected_shortfall_95: float = Field(..., description="ES 95% = mean(PnL[PnL <= VaR5])")
    # Risk-adjusted metrics
    sharpe_ratio: float = Field(..., description="Sharpe = (Avg Annual ROI - 7) / Annualized SD")
    sortino_ratio: float = Field(..., description="Sortino = (Avg Annual ROI - 7) / Negative Ann SD")
    comfort_ratio: float = Field(..., description="Comfort Ratio = Avg Annual ROI / Ulcer Index")
    # Rolling stats
    rolling_roi_mean: float = Field(..., description="52W Rolling ROI Mean = PnL.rolling(52).sum().mean()")
    rolling_roi_std: float = Field(..., description="52W Rolling ROI Std = PnL.rolling(52).sum().std()")
    rolling_roi_mean_minus_std: float = Field(..., description="52W Rolling ROI Mean - Std")
    # Win/loss metrics
    max_loss: float = Field(..., description="Max Loss = min(PnL%)")
    max_profit: float = Field(..., description="Max Win = max(PnL%)")
    win_rate: float = Field(..., description="Win Rate = len(PnL>=0) / len(PnL) * 100")
    profit_factor: float = Field(..., description="Profit Factor = PnL[>=0].sum / abs(PnL[<0].sum)")
    # Trade stats
    avg_orders_per_cycle: float = Field(..., description="Avg Orders = sum(Trades) / len(PnL)")
    # Count metrics
    num_periods: int = Field(..., description="Number of data periods")
    num_positive: int = Field(..., description="Number of winning periods (PnL >= 0)")
    num_negative: int = Field(..., description="Number of losing periods (PnL < 0)")
    weeks_below_x_pct: int = Field(..., description="No. of Weeks < X%")
    weeks_min_notional_below_y_pct: int = Field(..., description="No. of Weeks with Notional < Y%")


class AggregateStats(BaseModel):
    """Aggregate statistics across all variants."""

    mean: Dict[str, float] = Field(..., description="Mean values for each metric")
    median: Dict[str, float] = Field(..., description="Median values for each metric")
    std: Dict[str, float] = Field(..., description="Standard deviation for each metric")
    min: Dict[str, float] = Field(..., description="Minimum values for each metric")
    max: Dict[str, float] = Field(..., description="Maximum values for each metric")


# Request Models


class CompareLoadRequest(BaseModel):
    """Request model for loading CSV files for comparison."""

    csv_paths: List[str] = Field(
        ...,
        min_length=1,
        description="List of CSV file paths to load (each file = one variant)"
    )


class CompareCalculateMetricsRequest(BaseModel):
    """Request model for calculating performance metrics."""

    session_id: str = Field(..., description="Session ID from load step")
    pnl_column: str = Field(
        default="Pnl%",
        description="Column name containing PnL percentage values"
    )
    year_column: str = Field(
        default="Year",
        description="Column name containing year values"
    )
    trades_column: str = Field(
        default="Trades",
        description="Column name containing number of trades"
    )
    expiry_column: str = Field(
        default="Expiry",
        description="Column name for Expiry date (used for sorting data chronologically)"
    )
    # New fields for threshold metrics
    min_notional_column: str = Field(
        default="",
        description="Column name for Min Notional PnL (optional)"
    )
    threshold_x_pct: float = Field(
        default=-5.0,
        description="Threshold x% for weeks below calculation"
    )
    threshold_y_pct: float = Field(
        default=-10.0,
        description="Threshold y% for min notional calculation"
    )


class CompareGenerateChartRequest(BaseModel):
    """Request model for generating comparison charts."""

    session_id: str = Field(..., description="Session ID from load step")
    chart_type: ChartType = Field(
        default=ChartType.TABLE,
        description="Type of chart to generate"
    )
    metrics: Optional[List[str]] = Field(
        default=None,
        description="Specific metrics to include in the chart (None = all metrics)"
    )
    normalize: bool = Field(
        default=False,
        description="Whether to normalize metrics for comparison (useful for radar chart)"
    )
    selected_variants: Optional[List[str]] = Field(
        default=None,
        description="Specific variants to include (None = all variants)"
    )
    date_from: Optional[str] = Field(
        default=None,
        description="Start date filter for time-series charts (YYYY-MM-DD)"
    )
    date_to: Optional[str] = Field(
        default=None,
        description="End date filter for time-series charts (YYYY-MM-DD)"
    )
    alias_map: Optional[Dict[str, str]] = Field(
        default=None,
        description="Map of original variant names to display aliases"
    )


class CompareExportRequest(BaseModel):
    """Request model for exporting comparison table."""

    session_id: str = Field(..., description="Session ID from load step")
    alias_map: Optional[Dict[str, str]] = Field(
        default=None,
        description="Map of original variant names to display aliases"
    )
    format: str = Field(
        default="csv",
        description="Export format: 'csv' or 'pdf'"
    )
    filename: Optional[str] = Field(
        default=None,
        description="Custom filename for the export (without extension)"
    )
    title: Optional[str] = Field(
        default=None,
        description="Custom title for the report"
    )
    selected_variants: Optional[List[str]] = Field(
        default=None,
        description="List of variant names to include in export (None = all)"
    )


# Response Models


class CompareLoadResponse(BaseModel):
    """Response model for loading CSV files."""

    session_id: str = Field(..., description="Session ID for subsequent requests")
    num_files: int = Field(..., description="Number of files loaded")
    variants: List[VariantInfo] = Field(..., description="Information about each variant")
    columns: List[str] = Field(..., description="Available column names in the CSV")
    preview_data: Dict[str, List[Dict[str, Any]]] = Field(
        ...,
        description="Preview data (first 5 rows) for each variant"
    )


class CommonDateRange(BaseModel):
    """Common date range used for comparison."""

    start: Optional[str] = Field(None, description="Common start date (YYYY-MM-DD)")
    end: Optional[str] = Field(None, description="Common end date (YYYY-MM-DD)")


class CompareCalculateMetricsResponse(BaseModel):
    """Response model for calculated metrics."""

    session_id: str = Field(..., description="Session ID")
    variant_metrics: List[VariantMetrics] = Field(
        ...,
        description="Calculated metrics for each variant"
    )
    aggregate_stats: AggregateStats = Field(
        ...,
        description="Aggregate statistics across all variants"
    )
    common_date_range: Optional[CommonDateRange] = Field(
        None,
        description="Common date range used for comparison (intersection of all variants)"
    )


class CompareGenerateChartResponse(BaseModel):
    """Response model for generated charts."""

    session_id: str = Field(..., description="Session ID")
    chart_type: ChartType = Field(..., description="Type of chart generated")
    chart_data: Dict[str, Any] = Field(
        ...,
        description="Plotly chart JSON data"
    )
    table_data: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Table data (if applicable)"
    )
