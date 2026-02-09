"""Comparison module for variant performance analysis."""

from .metrics import calculate_variant_metrics, calculate_aggregate_stats
from .visualization import (
    generate_comparison_table,
    generate_grouped_bar_chart,
    generate_radar_chart,
    generate_cumulative_line_chart,
    generate_box_plot,
    generate_ranking_table,
    generate_correlation_heatmap,
)

__all__ = [
    "calculate_variant_metrics",
    "calculate_aggregate_stats",
    "generate_comparison_table",
    "generate_grouped_bar_chart",
    "generate_radar_chart",
    "generate_cumulative_line_chart",
    "generate_box_plot",
    "generate_ranking_table",
    "generate_correlation_heatmap",
]
