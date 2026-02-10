"""Visualization functions for variant comparison charts."""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Chart dimensions
CHART_HEIGHT_STANDARD = 500
CHART_HEIGHT_LARGE = 700

# Color palette for variants (distinct colors)
VARIANT_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
]

# Metric display names and formatting (all numeric values use 2 decimal places)
METRIC_DISPLAY = {
    # Return metrics
    "total_pnl": ("Total PnL %", ".2f"),
    "avg_pnl": ("Avg PnL %", ".2f"),
    "avg_annual_roi": ("Avg Annual ROI %", ".2f"),
    "last_52w_pnl": ("Last 52W PnL %", ".2f"),
    # Volatility metrics
    "annualized_sd": ("Annualized SD", ".2f"),
    "negative_annualized_sd": ("Neg. Annualized SD", ".2f"),
    # Drawdown metrics
    "max_drawdown": ("Max Drawdown %", ".2f"),
    "ulcer_index": ("Ulcer Index", ".2f"),
    "var_5pct": ("VaR 5%", ".2f"),
    "expected_shortfall_95": ("ES 95%", ".2f"),
    # Risk-adjusted metrics
    "sharpe_ratio": ("Sharpe Ratio", ".2f"),
    "sortino_ratio": ("Sortino Ratio", ".2f"),
    "comfort_ratio": ("Comfort Ratio", ".2f"),
    # Rolling stats
    "rolling_roi_mean": ("52W Rolling ROI Mean", ".2f"),
    "rolling_roi_std": ("52W Rolling ROI Std", ".2f"),
    "rolling_roi_mean_minus_std": ("52W ROI Mean - Std", ".2f"),
    # Win/loss metrics
    "max_loss": ("Max Loss %", ".2f"),
    "max_profit": ("Max Profit %", ".2f"),
    "win_rate": ("Win Rate %", ".2f"),
    "profit_factor": ("Profit Factor", ".2f"),
    # Trade stats
    "avg_orders_per_cycle": ("Avg Orders/Cycle", ".2f"),
    # Count metrics
    "weeks_below_x_pct": ("Weeks < X%", "d"),
    "weeks_min_notional_below_y_pct": ("Weeks Min Notional < Y%", "d"),
}

# Default metrics to show in charts
DEFAULT_METRICS = [
    "sharpe_ratio", "sortino_ratio", "max_drawdown",
    "total_pnl", "avg_annual_roi", "win_rate"
]

# Metrics organized by section for table display
METRIC_SECTIONS = [
    {
        "name": "Return Metrics",
        "metrics": ["total_pnl", "avg_pnl", "avg_annual_roi", "last_52w_pnl"]
    },
    {
        "name": "Volatility Metrics",
        "metrics": ["annualized_sd", "negative_annualized_sd"]
    },
    {
        "name": "Drawdown & Risk Metrics",
        "metrics": ["max_drawdown", "ulcer_index", "var_5pct", "expected_shortfall_95"]
    },
    {
        "name": "Risk-Adjusted Metrics",
        "metrics": ["sharpe_ratio", "sortino_ratio", "comfort_ratio"]
    },
    {
        "name": "Rolling Statistics",
        "metrics": ["rolling_roi_mean", "rolling_roi_std", "rolling_roi_mean_minus_std"]
    },
    {
        "name": "Win/Loss Metrics",
        "metrics": ["max_loss", "max_profit", "win_rate", "profit_factor"]
    },
    {
        "name": "Trade Statistics",
        "metrics": ["avg_orders_per_cycle"]
    },
    {
        "name": "Threshold Metrics",
        "metrics": ["weeks_below_x_pct", "weeks_min_notional_below_y_pct"]
    }
]


def get_metric_display(
    metric_key: str,
    threshold_x: float = -5.0,
    threshold_y: float = -10.0
) -> Tuple[str, str]:
    """
    Get the display name and format for a metric.

    For threshold-based metrics, dynamically substitutes X and Y values.

    Args:
        metric_key: The metric key
        threshold_x: The X threshold value for "Weeks < X%"
        threshold_y: The Y threshold value for "Weeks Min Notional < Y%"

    Returns:
        Tuple of (display_name, format_string)
    """
    if metric_key == "weeks_below_x_pct":
        return (f"Weeks < {threshold_x}%", "d")
    elif metric_key == "weeks_min_notional_below_y_pct":
        return (f"Weeks Min Notional < {threshold_y}%", "d")
    elif metric_key in METRIC_DISPLAY:
        return METRIC_DISPLAY[metric_key]
    else:
        # Fallback for unknown metrics
        return (metric_key.replace("_", " ").title(), ".2f")


def _get_layout_template() -> Dict[str, Any]:
    """Get common layout template for charts."""
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, system-ui, sans-serif", "size": 12},
        "margin": {"l": 60, "r": 40, "t": 60, "b": 60},
        "hovermode": "closest",
    }


def generate_comparison_table(
    variant_metrics: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    alias_map: Optional[Dict[str, str]] = None,
    threshold_x: float = -5.0,
    threshold_y: float = -10.0,
    include_sections: bool = True
) -> Dict[str, Any]:
    """
    Generate a comparison table with all variants and metrics.

    Args:
        variant_metrics: List of metric dictionaries for each variant
        metrics: Specific metrics to include (None = all)
        alias_map: Map of original variant names to display aliases
        threshold_x: The X threshold value for "Weeks < X%" display
        threshold_y: The Y threshold value for "Weeks Min Notional < Y%" display
        include_sections: Whether to include section header rows

    Returns:
        Dictionary with table_data for rendering
    """
    if not variant_metrics:
        return {"table_data": []}

    alias_map = alias_map or {}

    # Build table rows organized by sections
    table_data = []

    if include_sections:
        # Use METRIC_SECTIONS to organize metrics with section headers
        for section in METRIC_SECTIONS:
            section_name = section["name"]
            section_metrics = section["metrics"]

            # Filter to only include metrics that exist and are requested
            valid_metrics = [m for m in section_metrics if m in METRIC_DISPLAY or m in ["weeks_below_x_pct", "weeks_min_notional_below_y_pct"]]
            if metrics is not None:
                valid_metrics = [m for m in valid_metrics if m in metrics]

            if not valid_metrics:
                continue

            # Add section header row
            section_row = {
                "metric": section_name,
                "metric_key": "_section_header",
                "is_section_header": True
            }
            # Add empty values for each variant
            for vm in variant_metrics:
                original_name = vm["name"]
                display_name_variant = alias_map.get(original_name, original_name)
                section_row[display_name_variant] = ""
                section_row[f"{display_name_variant}_raw"] = None
            table_data.append(section_row)

            # Add metrics in this section
            for metric in valid_metrics:
                display_name, fmt = get_metric_display(metric, threshold_x, threshold_y)
                row = {"metric": display_name, "metric_key": metric, "is_section_header": False}

                for vm in variant_metrics:
                    original_name = vm["name"]
                    display_name_variant = alias_map.get(original_name, original_name)
                    value = vm.get(metric, 0)
                    row[display_name_variant] = format(value, fmt)
                    row[f"{display_name_variant}_raw"] = value

                table_data.append(row)
    else:
        # Original behavior without sections
        if metrics is None:
            metrics = list(METRIC_DISPLAY.keys())

        for metric in metrics:
            if metric not in METRIC_DISPLAY and metric not in ["weeks_below_x_pct", "weeks_min_notional_below_y_pct"]:
                continue

            display_name, fmt = get_metric_display(metric, threshold_x, threshold_y)
            row = {"metric": display_name, "metric_key": metric, "is_section_header": False}

            for vm in variant_metrics:
                original_name = vm["name"]
                display_name_variant = alias_map.get(original_name, original_name)
                value = vm.get(metric, 0)
                row[display_name_variant] = format(value, fmt)
                row[f"{display_name_variant}_raw"] = value

            table_data.append(row)

    return {"table_data": table_data}


def generate_grouped_bar_chart(
    variant_metrics: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    normalize: bool = False,
    alias_map: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Generate a grouped bar chart comparing metrics across variants.

    Args:
        variant_metrics: List of metric dictionaries for each variant
        metrics: Specific metrics to include (default: key metrics)
        normalize: Whether to normalize values (0-1 scale)
        alias_map: Map of original variant names to display aliases

    Returns:
        Plotly chart JSON
    """
    if not variant_metrics:
        return {}

    alias_map = alias_map or {}

    if metrics is None:
        metrics = DEFAULT_METRICS

    # Filter to valid metrics
    metrics = [m for m in metrics if m in METRIC_DISPLAY]

    # Build data matrix
    n_variants = len(variant_metrics)
    n_metrics = len(metrics)

    fig = go.Figure()

    for i, vm in enumerate(variant_metrics):
        original_name = vm["name"]
        display_name = alias_map.get(original_name, original_name)
        values = []
        texts = []

        for metric in metrics:
            val = vm.get(metric, 0)

            if normalize:
                # Normalize across variants for this metric
                all_vals = [v.get(metric, 0) for v in variant_metrics]
                min_val, max_val = min(all_vals), max(all_vals)
                if max_val != min_val:
                    val = (val - min_val) / (max_val - min_val)
                else:
                    val = 0.5

            values.append(val)
            _, fmt = METRIC_DISPLAY[metric]
            texts.append(format(vm.get(metric, 0), fmt))

        fig.add_trace(go.Bar(
            name=display_name,
            x=[METRIC_DISPLAY[m][0] for m in metrics],
            y=values,
            text=texts,
            textposition='auto',
            marker_color=VARIANT_COLORS[i % len(VARIANT_COLORS)],
            hovertemplate="<b>%{x}</b><br>" +
                          f"{display_name}: " + "%{text}<extra></extra>"
        ))

    layout = _get_layout_template()
    layout.update({
        "title": {"text": "Metric Comparison", "x": 0.5},
        "barmode": "group",
        "height": CHART_HEIGHT_STANDARD,
        "xaxis": {"title": "Metric", "tickangle": -45},
        "yaxis": {"title": "Normalized Value" if normalize else "Value"},
        "legend": {"orientation": "h", "y": -0.2},
    })

    fig.update_layout(**layout)
    return json.loads(fig.to_json())


def generate_radar_chart(
    variant_metrics: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    alias_map: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Generate a radar/spider chart for multi-dimensional comparison.

    All values are normalized to 0-1 scale for fair comparison.

    Args:
        variant_metrics: List of metric dictionaries for each variant
        metrics: Specific metrics to include
        alias_map: Map of original variant names to display aliases

    Returns:
        Plotly chart JSON
    """
    if not variant_metrics:
        return {}

    alias_map = alias_map or {}

    if metrics is None:
        metrics = DEFAULT_METRICS

    # Filter to valid metrics
    metrics = [m for m in metrics if m in METRIC_DISPLAY]

    # Metrics where lower is better (need to invert for radar)
    invert_metrics = {"max_drawdown", "negative_annualized_sd", "max_loss"}

    # Normalize all metrics to 0-1 scale
    normalized_data = {}
    for metric in metrics:
        values = [vm.get(metric, 0) for vm in variant_metrics]
        min_val, max_val = min(values), max(values)

        norm_values = []
        for v in values:
            if max_val != min_val:
                norm = (v - min_val) / (max_val - min_val)
            else:
                norm = 0.5

            # Invert if lower is better
            if metric in invert_metrics:
                norm = 1 - norm

            norm_values.append(norm)

        normalized_data[metric] = norm_values

    fig = go.Figure()

    for i, vm in enumerate(variant_metrics):
        original_name = vm["name"]
        display_name = alias_map.get(original_name, original_name)
        r_values = [normalized_data[m][i] for m in metrics]
        # Close the polygon
        r_values.append(r_values[0])

        theta = [METRIC_DISPLAY[m][0] for m in metrics]
        theta.append(theta[0])

        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=theta,
            fill='toself',
            name=display_name,
            line_color=VARIANT_COLORS[i % len(VARIANT_COLORS)],
            opacity=0.7,
        ))

    layout = _get_layout_template()
    layout.update({
        "title": {"text": "Performance Profile Comparison", "x": 0.5},
        "height": CHART_HEIGHT_LARGE,
        "polar": {
            "radialaxis": {
                "visible": True,
                "range": [0, 1],
                "tickvals": [0.2, 0.4, 0.6, 0.8, 1.0],
            }
        },
        "legend": {"orientation": "h", "y": -0.1},
    })

    fig.update_layout(**layout)
    return json.loads(fig.to_json())


def _parse_expiry_date(expiry_val) -> Optional[str]:
    """Parse expiry date from various formats."""
    if pd.isna(expiry_val):
        return None

    expiry_str = str(expiry_val)
    # Handle format like """2025-05-15T00:00:00.000Z"""
    expiry_str = expiry_str.strip('"').strip()

    try:
        # Try parsing ISO format
        if 'T' in expiry_str:
            return expiry_str.split('T')[0]
        return expiry_str
    except Exception:
        return None


def _get_sorted_data_with_expiry(
    df: pd.DataFrame,
    pnl_column: str,
    expiry_column: str = "Expiry"
) -> tuple:
    """
    Sort dataframe by expiry date and return returns with dates.

    Returns:
        Tuple of (dates list, returns Series)
    """
    df_copy = df.copy()

    # Try to find and parse expiry column
    if expiry_column in df_copy.columns:
        df_copy['_parsed_expiry'] = df_copy[expiry_column].apply(_parse_expiry_date)
        df_copy['_expiry_date'] = pd.to_datetime(df_copy['_parsed_expiry'], errors='coerce')
        df_copy = df_copy.dropna(subset=['_expiry_date', pnl_column])
        df_copy = df_copy.sort_values('_expiry_date')
        dates = df_copy['_expiry_date'].dt.strftime('%Y-%m-%d').tolist()
        returns = df_copy[pnl_column]
    elif 'Week start date' in df_copy.columns:
        df_copy['_parsed_date'] = df_copy['Week start date'].apply(_parse_expiry_date)
        df_copy['_date'] = pd.to_datetime(df_copy['_parsed_date'], errors='coerce')
        df_copy = df_copy.dropna(subset=['_date', pnl_column])
        df_copy = df_copy.sort_values('_date')
        dates = df_copy['_date'].dt.strftime('%Y-%m-%d').tolist()
        returns = df_copy[pnl_column]
    else:
        # Fallback to index-based
        df_copy = df_copy.dropna(subset=[pnl_column])
        dates = list(range(len(df_copy)))
        returns = df_copy[pnl_column]

    return dates, returns


def _apply_date_filter(
    dates: List[str],
    returns: pd.Series,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
) -> tuple:
    """
    Filter dates and returns by date range.

    Args:
        dates: List of date strings (YYYY-MM-DD)
        returns: Pandas Series of returns
        date_from: Optional start date (YYYY-MM-DD)
        date_to: Optional end date (YYYY-MM-DD)

    Returns:
        Tuple of (filtered dates list, filtered returns Series)
    """
    if not date_from and not date_to:
        return dates, returns

    # Convert to DataFrame for easier filtering
    filter_df = pd.DataFrame({
        'date': dates,
        'returns': returns.values
    })
    filter_df['date_parsed'] = pd.to_datetime(filter_df['date'], errors='coerce')

    if date_from:
        from_date = pd.to_datetime(date_from)
        filter_df = filter_df[filter_df['date_parsed'] >= from_date]

    if date_to:
        to_date = pd.to_datetime(date_to)
        filter_df = filter_df[filter_df['date_parsed'] <= to_date]

    filtered_dates = filter_df['date'].tolist()
    filtered_returns = pd.Series(filter_df['returns'].values)

    return filtered_dates, filtered_returns


def generate_cumulative_line_chart(
    variant_data: Dict[str, pd.DataFrame],
    pnl_column: str = "Pnl%",
    expiry_column: str = "Expiry",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    alias_map: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Generate a combined chart showing cumulative PnL with drawdown overlay.

    The chart displays:
    - Cumulative PnL lines on the primary y-axis (left)
    - Drawdown areas on the secondary y-axis (right)

    Args:
        variant_data: Dictionary mapping variant names to DataFrames
        pnl_column: Column name for PnL percentage
        expiry_column: Column name for expiry date
        date_from: Optional start date filter (YYYY-MM-DD)
        date_to: Optional end date filter (YYYY-MM-DD)
        alias_map: Map of original variant names to display aliases

    Returns:
        Plotly chart JSON
    """
    if not variant_data:
        return {}

    alias_map = alias_map or {}
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for i, (name, df) in enumerate(variant_data.items()):
        if pnl_column not in df.columns:
            continue

        display_name = alias_map.get(name, name)
        dates, returns = _get_sorted_data_with_expiry(df, pnl_column, expiry_column)

        if len(returns) == 0:
            continue

        # Apply date filtering
        if date_from or date_to:
            dates, returns = _apply_date_filter(dates, returns, date_from, date_to)

        if len(returns) == 0:
            continue

        # Calculate cumulative PnL as simple running sum
        cumulative = returns.cumsum()

        # Calculate drawdown
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max

        color = VARIANT_COLORS[i % len(VARIANT_COLORS)]

        # Add cumulative PnL line (primary y-axis)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cumulative.tolist(),
                mode='lines',
                name=f"{display_name}",
                line=dict(color=color, width=2),
                hovertemplate="<b>%{fullData.name}</b><br>" +
                              "Expiry: %{x}<br>" +
                              "Cumulative PnL: %{y:.2f}%<extra></extra>"
            ),
            secondary_y=False
        )

        # Add drawdown area (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown.tolist(),
                mode='lines',
                name=f"{display_name} (Drawdown)",
                fill='tozeroy',
                line=dict(color=color, width=1, dash='dot'),
                fillcolor=color.replace(')', ', 0.2)').replace('rgb', 'rgba') if color.startswith('rgb') else f"rgba(128, 128, 128, 0.2)",
                opacity=0.7,
                hovertemplate="<b>%{fullData.name}</b><br>" +
                              "Expiry: %{x}<br>" +
                              "Drawdown: %{y:.2f}%<extra></extra>"
            ),
            secondary_y=True
        )

    layout = _get_layout_template()
    layout.update({
        "title": {"text": "Cumulative PnL with Drawdown", "x": 0.5, "y": 0.98},
        "height": CHART_HEIGHT_LARGE,
        "xaxis": {"title": "Expiry Date", "tickangle": -45},
        "legend": {"orientation": "h", "y": -0.08, "x": 0.5, "xanchor": "center"},
        "hovermode": "x unified",
        "margin": {"l": 60, "r": 60, "t": 40, "b": 70},
        "autosize": True,
    })

    fig.update_layout(**layout)

    # Update y-axes titles
    fig.update_yaxes(title_text="Cumulative PnL %", secondary_y=False)
    fig.update_yaxes(title_text="Drawdown %", secondary_y=True, range=[None, 0])

    return json.loads(fig.to_json())


def generate_drawdown_chart(
    variant_data: Dict[str, pd.DataFrame],
    pnl_column: str = "Pnl%",
    expiry_column: str = "Expiry",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    alias_map: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Generate a standalone drawdown chart (kept for backwards compatibility).

    Args:
        variant_data: Dictionary mapping variant names to DataFrames
        pnl_column: Column name for PnL percentage
        expiry_column: Column name for expiry date
        date_from: Optional start date filter (YYYY-MM-DD)
        date_to: Optional end date filter (YYYY-MM-DD)
        alias_map: Map of original variant names to display aliases

    Returns:
        Plotly chart JSON
    """
    if not variant_data:
        return {}

    alias_map = alias_map or {}
    fig = go.Figure()

    for i, (name, df) in enumerate(variant_data.items()):
        if pnl_column not in df.columns:
            continue

        display_name = alias_map.get(name, name)
        dates, returns = _get_sorted_data_with_expiry(df, pnl_column, expiry_column)

        if len(returns) == 0:
            continue

        # Apply date filtering
        if date_from or date_to:
            dates, returns = _apply_date_filter(dates, returns, date_from, date_to)

        if len(returns) == 0:
            continue

        # Calculate cumulative PnL as simple running sum
        cumulative = returns.cumsum()

        # Calculate running maximum of cumulative PnL
        running_max = cumulative.cummax()

        # Calculate drawdown as difference from peak
        drawdown = cumulative - running_max

        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdown.tolist(),
            mode='lines',
            name=display_name,
            fill='tozeroy',
            line=dict(color=VARIANT_COLORS[i % len(VARIANT_COLORS)], width=2),
            hovertemplate="<b>%{fullData.name}</b><br>" +
                          "Expiry: %{x}<br>" +
                          "Drawdown: %{y:.2f}%<extra></extra>"
        ))

    layout = _get_layout_template()
    layout.update({
        "title": {"text": "Drawdown from Peak", "x": 0.5},
        "height": CHART_HEIGHT_STANDARD,
        "xaxis": {"title": "Expiry Date", "tickangle": -45},
        "yaxis": {"title": "Drawdown %", "range": [None, 0]},
        "legend": {"orientation": "h", "y": -0.2},
        "hovermode": "x unified",
    })

    fig.update_layout(**layout)
    return json.loads(fig.to_json())


def generate_box_plot(
    variant_data: Dict[str, pd.DataFrame],
    pnl_column: str = "Pnl%"
) -> Dict[str, Any]:
    """
    Generate box plots showing return distribution per variant.

    Args:
        variant_data: Dictionary mapping variant names to DataFrames
        pnl_column: Column name for PnL percentage

    Returns:
        Plotly chart JSON
    """
    if not variant_data:
        return {}

    fig = go.Figure()

    for i, (name, df) in enumerate(variant_data.items()):
        if pnl_column not in df.columns:
            continue

        returns = df[pnl_column].dropna()

        fig.add_trace(go.Box(
            y=returns.values,
            name=name,
            marker_color=VARIANT_COLORS[i % len(VARIANT_COLORS)],
            boxmean='sd',  # Show mean and standard deviation
            hovertemplate="<b>%{fullData.name}</b><br>" +
                          "Value: %{y:.3f}%<extra></extra>"
        ))

    layout = _get_layout_template()
    layout.update({
        "title": {"text": "Return Distribution", "x": 0.5},
        "height": CHART_HEIGHT_STANDARD,
        "yaxis": {"title": "Weekly Return %"},
        "showlegend": False,
    })

    fig.update_layout(**layout)
    return json.loads(fig.to_json())


def generate_ranking_table(
    variant_metrics: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    alias_map: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Generate a ranking table showing variant rankings for each metric.

    Args:
        variant_metrics: List of metric dictionaries for each variant
        metrics: Specific metrics to rank
        alias_map: Map of original variant names to display aliases

    Returns:
        Dictionary with ranking_data
    """
    if not variant_metrics:
        return {"ranking_data": []}

    alias_map = alias_map or {}

    if metrics is None:
        metrics = DEFAULT_METRICS

    # Metrics where lower is better
    lower_better = {"max_drawdown", "negative_annualized_sd", "max_loss", "annualized_sd"}

    ranking_data = []

    for metric in metrics:
        if metric not in METRIC_DISPLAY:
            continue

        display_name, fmt = METRIC_DISPLAY[metric]

        # Get values and sort
        values = [(vm["name"], vm.get(metric, 0)) for vm in variant_metrics]

        # Sort: ascending for "lower is better", descending otherwise
        ascending = metric in lower_better
        values.sort(key=lambda x: x[1], reverse=not ascending)

        # Build ranking row
        row = {"metric": display_name, "metric_key": metric}
        for rank, (original_name, val) in enumerate(values, 1):
            display_name_variant = alias_map.get(original_name, original_name)
            row[f"rank_{rank}"] = f"{display_name_variant}: {format(val, fmt)}"

        ranking_data.append(row)

    return {"ranking_data": ranking_data}


def generate_correlation_heatmap(
    variant_metrics: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate a correlation heatmap showing metric correlations.

    Args:
        variant_metrics: List of metric dictionaries for each variant
        metrics: Specific metrics to include

    Returns:
        Plotly chart JSON
    """
    if not variant_metrics or len(variant_metrics) < 2:
        return {}

    if metrics is None:
        metrics = list(METRIC_DISPLAY.keys())

    # Filter to valid metrics with data
    valid_metrics = []
    for m in metrics:
        if m in METRIC_DISPLAY:
            values = [vm.get(m, 0) for vm in variant_metrics]
            if any(v != 0 for v in values):
                valid_metrics.append(m)

    if len(valid_metrics) < 2:
        return {}

    # Build correlation matrix
    df = pd.DataFrame(variant_metrics)
    df = df[valid_metrics]
    corr_matrix = df.corr()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[METRIC_DISPLAY[m][0] for m in valid_metrics],
        y=[METRIC_DISPLAY[m][0] for m in valid_metrics],
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>" +
                      "Correlation: %{z:.3f}<extra></extra>"
    ))

    layout = _get_layout_template()
    layout.update({
        "title": {"text": "Metric Correlations", "x": 0.5},
        "height": CHART_HEIGHT_LARGE,
        "xaxis": {"tickangle": -45},
    })

    fig.update_layout(**layout)
    return json.loads(fig.to_json())
