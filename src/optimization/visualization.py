"""Visualization functions for optimization pipeline - returns Plotly JSON."""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import scipy for convex hulls (optional - will gracefully degrade if not available)
try:
    from scipy.spatial import ConvexHull
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ==================== Chart Constants ====================

# Standard chart dimensions
CHART_HEIGHT_STANDARD = 550
CHART_HEIGHT_LARGE = 800
CHART_HEIGHT_GRID = 450

# Marker defaults for consistent styling
MARKER_DEFAULTS = {
    'size': 10,
    'opacity': 0.85,
    'line_width': 1.5,
    'line_color': 'white'
}

# Bright color palette for clusters
CLUSTER_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5'
]

# Enhanced 9-stop RdYlGn colorscale with better contrast
COLORSCALE_RDYLGN = [
    [0.0, '#a50026'],    # Dark red (very low/bad)
    [0.125, '#d73027'],  # Red
    [0.25, '#f46d43'],   # Orange-red
    [0.375, '#fdae61'],  # Orange
    [0.5, '#fee08b'],    # Yellow (middle)
    [0.625, '#d9ef8b'],  # Light green
    [0.75, '#a6d96a'],   # Green
    [0.875, '#66bd63'],  # Medium green
    [1.0, '#1a9850']     # Dark green (high/good)
]

COLORSCALE_RDYLGN_R = [
    [0.0, '#1a9850'],    # Dark green (low/good for drawdown)
    [0.125, '#66bd63'],
    [0.25, '#a6d96a'],
    [0.375, '#d9ef8b'],
    [0.5, '#fee08b'],    # Yellow (middle)
    [0.625, '#fdae61'],
    [0.75, '#f46d43'],
    [0.875, '#d73027'],
    [1.0, '#a50026']     # Dark red (high/bad for drawdown)
]


# Standard metrics for heatmaps (matching notebook order)
HEATMAP_METRICS = [
    ('total_pnl', 'Total PnL', False),
    ('profit_factor', 'Profit Factor', False),
    ('avg_annual_roi', 'ROI Mean', False),
    ('win_ratio', 'Win Ratio', False),
    ('sortino_ratio', 'Sortino Ratio', False),
    ('sharpe_ratio', 'Sharpe Ratio', False),
    ('max_draw_down', 'Max Draw Down', True),  # True = reverse colorscale
]


def _fig_to_json(fig: go.Figure) -> Dict[str, Any]:
    """Convert Plotly figure to JSON dict."""
    return json.loads(fig.to_json())


def _calculate_percentiles_and_ranks(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate percentiles and ranks for an array of values.

    Args:
        values: Array of numeric values

    Returns:
        Tuple of (percentiles, ranks) arrays
    """
    # Handle NaN values
    valid_mask = np.isfinite(values)
    n_valid = np.sum(valid_mask)

    percentiles = np.full_like(values, np.nan, dtype=float)
    ranks = np.full_like(values, np.nan, dtype=float)

    if n_valid > 0:
        valid_values = values[valid_mask]
        # Calculate percentiles (0-100)
        sorted_indices = np.argsort(valid_values)
        percentile_values = (np.arange(n_valid) / (n_valid - 1) * 100) if n_valid > 1 else np.array([50.0])

        # Map back to original positions
        temp_percentiles = np.zeros(n_valid)
        temp_percentiles[sorted_indices] = percentile_values
        percentiles[valid_mask] = temp_percentiles

        # Calculate ranks (1 = best, n = worst, higher values = better rank)
        temp_ranks = np.zeros(n_valid)
        temp_ranks[sorted_indices] = np.arange(n_valid, 0, -1)
        ranks[valid_mask] = temp_ranks

    return percentiles, ranks


def _calculate_cluster_centroids(
    df: pd.DataFrame,
    cluster_col: str = 'cluster'
) -> Dict[int, Tuple[float, float]]:
    """
    Calculate the centroid (mean PC1, PC2) for each cluster.

    Args:
        df: DataFrame with PC1, PC2 and cluster column
        cluster_col: Name of cluster column

    Returns:
        Dict mapping cluster_id to (centroid_x, centroid_y)
    """
    centroids = {}
    for cluster_id in df[cluster_col].unique():
        if cluster_id == -1:  # Skip noise
            continue
        cluster_data = df[df[cluster_col] == cluster_id]
        centroid_x = cluster_data['PC1'].mean()
        centroid_y = cluster_data['PC2'].mean()
        centroids[cluster_id] = (centroid_x, centroid_y)
    return centroids


def _get_convex_hull_boundary(
    points: np.ndarray
) -> Optional[np.ndarray]:
    """
    Get the convex hull boundary points for a set of 2D points.

    Args:
        points: Nx2 array of (x, y) points

    Returns:
        Array of boundary points (closed polygon) or None if hull cannot be computed
    """
    if not SCIPY_AVAILABLE:
        return None

    if len(points) < 3:
        return None

    try:
        hull = ConvexHull(points)
        # Get hull vertices and close the polygon
        hull_points = points[hull.vertices]
        # Close the polygon by appending the first point
        hull_points = np.vstack([hull_points, hull_points[0]])
        return hull_points
    except Exception:
        return None


def _get_cluster_stats_annotation(
    df: pd.DataFrame,
    cluster_col: str,
    metric_col: str = 'sharpe_ratio'
) -> Dict[int, Dict[str, Any]]:
    """
    Get statistics for each cluster for annotation purposes.

    Args:
        df: DataFrame with cluster and metric columns
        cluster_col: Name of cluster column
        metric_col: Name of metric column for stats

    Returns:
        Dict mapping cluster_id to stats dict with n, mean, etc.
    """
    stats = {}
    for cluster_id in df[cluster_col].unique():
        if cluster_id == -1:
            continue
        cluster_data = df[df[cluster_col] == cluster_id]
        if metric_col in cluster_data.columns:
            stats[cluster_id] = {
                'n': len(cluster_data),
                'mean': cluster_data[metric_col].mean(),
                'median': cluster_data[metric_col].median(),
                'std': cluster_data[metric_col].std()
            }
        else:
            stats[cluster_id] = {'n': len(cluster_data)}
    return stats


def generate_pca_variance_chart(
    explained_variance: np.ndarray,
    cumulative_variance: np.ndarray
) -> Dict[str, Any]:
    """
    Generate PCA explained variance bar chart (matching Jupyter notebook style).

    Args:
        explained_variance: Variance ratio per component
        cumulative_variance: Cumulative variance

    Returns:
        Plotly JSON dict
    """
    # Convert numpy arrays to Python lists
    explained_variance = list(explained_variance)
    cumulative_variance = list(cumulative_variance)

    n_components = len(explained_variance)
    components = list(range(1, n_components + 1))

    fig = go.Figure()

    # Bar chart for individual variance (matching notebook: alpha=0.5, align='center')
    fig.add_trace(go.Bar(
        x=components,
        y=explained_variance,
        name='Individual Explained Variance',
        marker_color='steelblue',
        opacity=0.5,
        hovertemplate='PC%{x}<br>Variance: %{y:.2%}<extra></extra>'
    ))

    # Step line for cumulative variance (matching notebook: where='mid')
    # For 'mid' step, we need to use 'vh' shape in Plotly
    fig.add_trace(go.Scatter(
        x=components,
        y=cumulative_variance,
        name='Cumulative Explained Variance',
        mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(size=8, color='red'),
        hovertemplate='PC%{x}<br>Cumulative: %{y:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text='Explained Variance Ratio by Principal Component',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Principal Component',
            tickmode='linear',
            tick0=1,
            dtick=1,
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            title='Explained Variance Ratio',
            tickformat='.0%',
            gridcolor='lightgray',
            showgrid=True,
            range=[0, 1.05]
        ),
        legend=dict(
            x=0.6,
            y=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        hovermode='x unified',
        height=450,
        bargap=0.3
    )

    return _fig_to_json(fig)


def generate_pca_scatter(
    pca_df: pd.DataFrame,
    hover_data: Optional[Dict[str, pd.Series]] = None,
    metrics_data: Optional[Dict[str, pd.Series]] = None,
    color_by_metric: Optional[str] = None,
    show_density_contours: bool = False
) -> Dict[str, Any]:
    """
    Generate interactive PCA scatter plot (PC1 vs PC2) with visible points.

    Args:
        pca_df: DataFrame with PC columns
        hover_data: Optional dict of strategy params for hover
        metrics_data: Optional dict of metrics for hover (sharpe, sortino, profit_factor)
        color_by_metric: Optional metric column name to color points by
        show_density_contours: Whether to show density contours

    Returns:
        Plotly JSON dict
    """
    df = pca_df.copy()
    df['variant_id'] = df.index

    # Add strategy params
    if hover_data:
        for key, values in hover_data.items():
            df[key] = values.values if hasattr(values, 'values') else values

    # Add key metrics
    if metrics_data:
        for key, values in metrics_data.items():
            df[key] = values.values if hasattr(values, 'values') else values

    # Build custom hover text
    hover_texts = []
    for idx, row in df.iterrows():
        parts = [f"<b>Variant: {row['variant_id']}</b>"]

        # Add strategy params
        if hover_data:
            parts.append("<br><b>Strategy Params:</b>")
            for key in hover_data.keys():
                if key in row:
                    val = row[key]
                    parts.append(f"  {key}: {val}")

        # Add key metrics
        if metrics_data:
            parts.append("<br><b>Metrics:</b>")
            for key in metrics_data.keys():
                if key in row:
                    val = row[key]
                    if isinstance(val, (float, np.floating)):
                        parts.append(f"  {key}: {val:.4f}")
                    else:
                        parts.append(f"  {key}: {val}")

        hover_texts.append('<br>'.join(parts))

    # Convert to Python lists
    x_data = df['PC1'].tolist()
    y_data = df['PC2'].tolist()

    fig = go.Figure()

    # Add density contours if requested
    if show_density_contours:
        fig.add_trace(go.Histogram2dContour(
            x=x_data,
            y=y_data,
            colorscale='Blues',
            reversescale=True,
            showscale=False,
            contours=dict(
                showlabels=False,
                coloring='heatmap'
            ),
            opacity=0.3,
            hoverinfo='skip',
            name='Density'
        ))

    # Determine marker color
    if color_by_metric and color_by_metric in df.columns:
        color_values = df[color_by_metric].tolist()
        marker_config = dict(
            size=MARKER_DEFAULTS['size'],
            color=color_values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=color_by_metric),
            line=dict(width=MARKER_DEFAULTS['line_width'], color=MARKER_DEFAULTS['line_color'])
        )
    else:
        marker_config = dict(
            size=MARKER_DEFAULTS['size'],
            color='rgba(59, 130, 246, 0.8)',  # Blue with alpha
            line=dict(width=MARKER_DEFAULTS['line_width'], color='rgba(255, 255, 255, 0.8)')
        )

    fig.add_trace(go.Scattergl(
        x=x_data,
        y=y_data,
        mode='markers',
        marker=marker_config,
        text=hover_texts,
        hoverinfo='text',
        name='Variants'
    ))

    fig.update_layout(
        title=dict(
            text='PCA: PC1 vs PC2',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Principal Component 1 (PC1)',
            gridcolor='lightgray',
            showgrid=True,
            zeroline=True,
            zerolinecolor='gray'
        ),
        yaxis=dict(
            title='Principal Component 2 (PC2)',
            gridcolor='lightgray',
            showgrid=True,
            zeroline=True,
            zerolinecolor='gray'
        ),
        hovermode='closest',
        height=CHART_HEIGHT_STANDARD,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5
        ),
        margin=dict(b=80)  # More bottom margin for legend
    )

    return _fig_to_json(fig)


def generate_cluster_scatter(
    df: pd.DataFrame,
    cluster_col: str,
    hover_cols: List[str],
    title: str = "Cluster Scatter Plot",
    metrics_cols: Optional[List[str]] = None,
    show_centroids: bool = False,
    show_hulls: bool = True,
    show_stats_annotations: bool = False,
    ranking_metric: str = 'sharpe_ratio'
) -> Dict[str, Any]:
    """
    Generate cluster scatter plot colored by cluster with visible points.

    Args:
        df: DataFrame with PC1, PC2, and cluster column
        cluster_col: Name of cluster column
        hover_cols: Strategy param columns to show on hover
        title: Plot title
        metrics_cols: Optional list of metric columns for hover
        show_centroids: Whether to show cluster centroids
        show_hulls: Whether to show convex hull boundaries
        show_stats_annotations: Whether to show n= and mean metric per cluster
        ranking_metric: Metric to use for stats annotations

    Returns:
        Plotly JSON dict
    """
    plot_df = df.copy()
    plot_df['variant_id'] = plot_df.index

    # Get unique clusters and sort them
    clusters = sorted(plot_df[cluster_col].unique())

    fig = go.Figure()

    # Calculate centroids and stats for annotations
    centroids = _calculate_cluster_centroids(plot_df, cluster_col)
    cluster_stats = _get_cluster_stats_annotation(plot_df, cluster_col, ranking_metric)

    for i, cluster in enumerate(clusters):
        cluster_data = plot_df[plot_df[cluster_col] == cluster]

        # Color: gray for noise (-1), bright colors for others
        if cluster == -1:
            color = 'rgba(180, 180, 180, 0.5)'
            cluster_name = 'Noise'
        else:
            color = CLUSTER_COLORS[int(cluster) % len(CLUSTER_COLORS)]
            cluster_name = f'Cluster {cluster}'

        # Add convex hull boundary if requested and available
        if show_hulls and cluster != -1 and SCIPY_AVAILABLE:
            points = cluster_data[['PC1', 'PC2']].values
            hull_boundary = _get_convex_hull_boundary(points)
            if hull_boundary is not None:
                fig.add_trace(go.Scatter(
                    x=hull_boundary[:, 0].tolist(),
                    y=hull_boundary[:, 1].tolist(),
                    mode='lines',
                    line=dict(color=color, width=2, dash='dot'),
                    fill='toself',
                    fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}' if color.startswith('#') else 'rgba(200,200,200,0.1)',
                    showlegend=False,
                    hoverinfo='skip',
                    name=f'Hull {cluster}'
                ))

        # Build custom hover text
        hover_texts = []
        for idx, row in cluster_data.iterrows():
            parts = [f"<b>Variant: {row['variant_id']}</b>"]
            parts.append(f"<b>Cluster: {cluster}</b>")

            # Add strategy params
            if hover_cols:
                parts.append("<br><b>Strategy Params:</b>")
                for col in hover_cols:
                    if col in row and col != cluster_col:
                        val = row[col]
                        if isinstance(val, (float, np.floating)):
                            parts.append(f"  {col}: {val:.4f}")
                        else:
                            parts.append(f"  {col}: {val}")

            # Add metrics
            if metrics_cols:
                parts.append("<br><b>Metrics:</b>")
                for col in metrics_cols:
                    if col in row:
                        val = row[col]
                        if isinstance(val, (float, np.floating)):
                            parts.append(f"  {col}: {val:.4f}")
                        else:
                            parts.append(f"  {col}: {val}")

            hover_texts.append('<br>'.join(parts))

        # Convert to lists for JSON serialization
        x_data = cluster_data['PC1'].tolist()
        y_data = cluster_data['PC2'].tolist()

        fig.add_trace(go.Scattergl(
            x=x_data,
            y=y_data,
            mode='markers',
            marker=dict(
                size=MARKER_DEFAULTS['size'],
                color=color,
                opacity=MARKER_DEFAULTS['opacity'],
                line=dict(width=MARKER_DEFAULTS['line_width'], color=MARKER_DEFAULTS['line_color'])
            ),
            text=hover_texts,
            hoverinfo='text',
            name=cluster_name
        ))

        # Add centroid marker if requested
        if show_centroids and cluster != -1 and cluster in centroids:
            cx, cy = centroids[cluster]
            fig.add_trace(go.Scatter(
                x=[cx],
                y=[cy],
                mode='markers',
                marker=dict(
                    size=15,
                    color=color,
                    symbol='x',
                    line=dict(width=3, color='black')
                ),
                showlegend=False,
                hoverinfo='text',
                text=f'Centroid Cluster {cluster}',
                name=f'Centroid {cluster}'
            ))

    # Add stats annotations
    annotations = []
    if show_stats_annotations:
        for cluster_id, stats in cluster_stats.items():
            if cluster_id in centroids:
                cx, cy = centroids[cluster_id]
                n = stats.get('n', 0)
                mean_val = stats.get('mean', None)
                annotation_text = f"n={n}"
                if mean_val is not None:
                    annotation_text += f"<br>μ={mean_val:.3f}"

                annotations.append(dict(
                    x=cx,
                    y=cy,
                    text=annotation_text,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    ax=30,
                    ay=-30,
                    font=dict(size=10, color='black'),
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='gray',
                    borderwidth=1
                ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Principal Component 1 (PC1)',
            gridcolor='lightgray',
            showgrid=True,
            zeroline=True,
            zerolinecolor='gray'
        ),
        yaxis=dict(
            title='Principal Component 2 (PC2)',
            gridcolor='lightgray',
            showgrid=True,
            zeroline=True,
            zerolinecolor='gray'
        ),
        hovermode='closest',
        legend=dict(
            title='Cluster',
            orientation='h',
            yanchor='top',
            y=1.15,
            xanchor='center',
            x=0.5
        ),
        height=CHART_HEIGHT_STANDARD,
        margin=dict(t=120),  # More top margin for legend above title
        annotations=annotations
    )

    return _fig_to_json(fig)


def _generate_single_heatmap(
    pivot_data: pd.DataFrame,
    metric_name: str,
    metric_title: str,
    reverse_colorscale: bool,
    x_param: str,
    y_param: str,
    vmin: float,
    vmax: float,
    highlight_cells: Optional[List[tuple]] = None,
    subtitle: str = "",
    show_percentiles: bool = True
) -> go.Figure:
    """Generate a single heatmap figure (static, non-interactive style)."""
    # Use enhanced 9-stop colorscale for better contrast
    colorscale = COLORSCALE_RDYLGN_R if reverse_colorscale else COLORSCALE_RDYLGN

    # Convert pivot data to lists for Plotly, replacing NaN with None
    z_array = pivot_data.values
    z_values = []
    text_values = []

    # Flatten values for percentile/rank calculation
    flat_values = z_array.flatten()
    percentiles, ranks = _calculate_percentiles_and_ranks(flat_values)
    percentiles_2d = percentiles.reshape(z_array.shape)
    ranks_2d = ranks.reshape(z_array.shape)
    total_valid = np.sum(np.isfinite(flat_values))

    # Build custom hover text with percentile and rank info
    hover_texts = []
    for i, row in enumerate(z_array):
        z_row = []
        text_row = []
        hover_row = []
        for j, val in enumerate(row):
            if pd.isna(val) or not np.isfinite(val):
                z_row.append(None)
                text_row.append('')
                hover_row.append('')
            else:
                z_row.append(float(val))
                text_row.append(f'{float(val):.2f}')

                # Build enhanced hover text
                x_val = pivot_data.columns[j]
                y_val = pivot_data.index[i]
                pct = percentiles_2d[i, j]
                rank = int(ranks_2d[i, j]) if np.isfinite(ranks_2d[i, j]) else '-'

                hover_parts = [
                    f"<b>{x_param}</b>: {x_val}",
                    f"<b>{y_param}</b>: {y_val}",
                    f"<b>{metric_name}</b>: {val:.4f}",
                ]
                if show_percentiles:
                    hover_parts.extend([
                        f"<b>Percentile</b>: {pct:.1f}%",
                        f"<b>Rank</b>: {rank}/{total_valid}"
                    ])
                hover_row.append('<br>'.join(hover_parts))

        z_values.append(z_row)
        text_values.append(text_row)
        hover_texts.append(hover_row)

    x_labels = [str(x) for x in pivot_data.columns]
    y_labels = [str(y) for y in pivot_data.index]

    fig = go.Figure()

    # Main heatmap with text on cells
    fig.add_trace(go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        text=text_values,
        texttemplate='%{text}',
        textfont=dict(size=10, color='black'),
        colorscale=colorscale,
        zmin=vmin,
        zmax=vmax,
        showscale=True,
        colorbar=dict(
            title=dict(text=metric_name, side='right'),
            thickness=20,
            len=0.9
        ),
        hoverongaps=False,
        customdata=hover_texts,
        hovertemplate='%{customdata}<extra></extra>',
        xgap=1,  # Small gap between cells for better visibility
        ygap=1
    ))

    # Add highlight boxes if specified
    shapes = []
    if highlight_cells:
        x_vals = list(pivot_data.columns)
        y_vals = list(pivot_data.index)

        for (x_val, y_val) in highlight_cells:
            if x_val in x_vals and y_val in y_vals:
                x_idx = x_vals.index(x_val)
                y_idx = y_vals.index(y_val)

                shapes.append(dict(
                    type='rect',
                    x0=x_idx - 0.5, x1=x_idx + 0.5,
                    y0=y_idx - 0.5, y1=y_idx + 0.5,
                    line=dict(color='black', width=3),
                    fillcolor='rgba(0,0,0,0)'
                ))

    title_text = f'{metric_title}'
    if subtitle:
        title_text = f'{subtitle}<br>{metric_title}'

    # Calculate dynamic height based on number of y values
    num_y = len(y_labels)
    num_x = len(x_labels)
    cell_size = 50  # pixels per cell
    min_height = 400
    min_width = 600
    dynamic_height = max(min_height, num_y * cell_size + 150)
    dynamic_width = max(min_width, num_x * cell_size + 200)

    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor='center', font=dict(size=16)),
        xaxis=dict(
            title=dict(text=x_param, font=dict(size=14)),
            type='category',
            side='bottom',
            tickangle=0,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title=dict(text=y_param, font=dict(size=14)),
            type='category',
            autorange='reversed',  # Match seaborn default
            tickfont=dict(size=11)
        ),
        shapes=shapes,
        height=dynamic_height,
        autosize=True,  # Allow responsive width
        margin=dict(l=100, r=120, t=100, b=80)
    )

    return fig


def generate_heatmaps(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    const_param1: Optional[str] = None,
    const_param2: Optional[str] = None,
    metrics: Optional[List[tuple]] = None,
    highlight_col: Optional[str] = None,
    highlight_val: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Generate Plotly heatmaps based on number of parameters.

    - 2 params (x, y): Single heatmap per metric
    - 3 params (x, y, const1): One heatmap per const1 value per metric
    - 4 params (x, y, const1, const2): Grid of heatmaps per (const1, const2) combo

    Args:
        df: DataFrame with parameter and metric columns
        x_param: X-axis parameter
        y_param: Y-axis parameter
        const_param1: First constant parameter (optional)
        const_param2: Second constant parameter (optional)
        metrics: List of (col_name, display_name, reverse_colorscale) tuples
        highlight_col: Column to use for highlighting
        highlight_val: Value to match for highlighting

    Returns:
        List of Plotly JSON dicts
    """
    if metrics is None:
        metrics = HEATMAP_METRICS

    # Filter to available metrics
    available_metrics = [(col, title, rev) for col, title, rev in metrics if col in df.columns]

    if not available_metrics:
        return []

    # Calculate global vmin/vmax for each metric
    vmin_vmax = {}
    for col, _, _ in available_metrics:
        v = df[col].values
        v = v[np.isfinite(v)]
        vmin_vmax[col] = (np.min(v) if len(v) > 0 else 0, np.max(v) if len(v) > 0 else 1)

    # Determine highlight cells
    highlight_cells_map = {}
    if highlight_col is not None and highlight_val is not None and highlight_col in df.columns:
        highlight_df = df[df[highlight_col] == highlight_val]
        for _, row in highlight_df.iterrows():
            key = (
                row.get(const_param1) if const_param1 else None,
                row.get(const_param2) if const_param2 else None
            )
            if key not in highlight_cells_map:
                highlight_cells_map[key] = []
            highlight_cells_map[key].append((row[x_param], row[y_param]))

    results = []

    # 2-param mode: single heatmap per metric
    if const_param1 is None and const_param2 is None:
        for col, title, reverse in available_metrics:
            pivot = df.pivot_table(index=y_param, columns=x_param, values=col, aggfunc='mean')
            vmin, vmax = vmin_vmax[col]
            highlight_cells = highlight_cells_map.get((None, None), None)

            fig = _generate_single_heatmap(
                pivot, col, title, reverse, x_param, y_param,
                vmin, vmax, highlight_cells
            )
            fig_json = _fig_to_json(fig)
            # Add metadata for navigation filtering
            fig_json['_metadata'] = {
                'metric': title,
                'metric_col': col,
                'const_value': None,
                'const_param': None
            }
            results.append(fig_json)

    # 3-param mode: one heatmap per const1 value per metric
    elif const_param2 is None:
        const1_values = sorted(df[const_param1].unique())

        for const1_val in const1_values:
            subset = df[df[const_param1] == const1_val]

            for col, title, reverse in available_metrics:
                pivot = subset.pivot_table(index=y_param, columns=x_param, values=col, aggfunc='mean')
                vmin, vmax = vmin_vmax[col]
                highlight_cells = highlight_cells_map.get((const1_val, None), None)

                fig = _generate_single_heatmap(
                    pivot, col, title, reverse, x_param, y_param,
                    vmin, vmax, highlight_cells,
                    subtitle=f'{const_param1} = {const1_val}'
                )
                fig_json = _fig_to_json(fig)
                # Add metadata for navigation filtering
                fig_json['_metadata'] = {
                    'metric': title,
                    'metric_col': col,
                    'const_value': str(const1_val),
                    'const_param': const_param1
                }
                results.append(fig_json)

    # 4-param mode: individual heatmaps per (const1, const2) combination
    # This enables filtering in the UI by both const values
    else:
        const1_values = sorted(df[const_param1].unique())
        const2_values = sorted(df[const_param2].unique())

        for const1_val in const1_values:
            for const2_val in const2_values:
                subset = df[(df[const_param1] == const1_val) & (df[const_param2] == const2_val)]

                if subset.empty:
                    continue

                for col, title, reverse in available_metrics:
                    vmin, vmax = vmin_vmax[col]
                    pivot = subset.pivot_table(index=y_param, columns=x_param, values=col, aggfunc='mean')
                    highlight_cells = highlight_cells_map.get((const1_val, const2_val), None)

                    # Generate individual heatmap with subtitle showing both const values
                    fig = _generate_single_heatmap(
                        pivot, col, title, reverse, x_param, y_param,
                        vmin, vmax, highlight_cells,
                        subtitle=f'{const_param1}={const1_val}, {const_param2}={const2_val}'
                    )
                    fig_json = _fig_to_json(fig)
                    # Add metadata for navigation filtering (supports filtering by both const values)
                    fig_json['_metadata'] = {
                        'metric': title,
                        'metric_col': col,
                        'const_value': str(const1_val),
                        'const_value2': str(const2_val),
                        'const_param': const_param1,
                        'const_param2': const_param2
                    }
                    results.append(fig_json)

    return results


def generate_cluster_stats_table(
    cluster_stats: pd.DataFrame,
    metric_name: str
) -> Dict[str, Any]:
    """
    Generate a table visualization for cluster statistics.

    Args:
        cluster_stats: DataFrame with cluster stats
        metric_name: Name of the metric

    Returns:
        Plotly JSON dict
    """
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Cluster', 'Count', 'Median', 'Mean', 'Std'],
            fill_color='steelblue',
            font=dict(color='white', size=12),
            align='center'
        ),
        cells=dict(
            values=[
                cluster_stats['cluster_id'],
                cluster_stats['count'],
                cluster_stats['median'].round(4),
                cluster_stats['mean'].round(4),
                cluster_stats['std'].round(4)
            ],
            fill_color='lavender',
            align='center'
        )
    )])

    fig.update_layout(
        title=f'Cluster Statistics by {metric_name}',
        height=400
    )

    return _fig_to_json(fig)


def generate_hdbscan_grid(
    cluster_dfs: List[pd.DataFrame],
    configs: List[tuple],
    strategy_params: List[str],
    ranking_metric: str,
    title: str = "HDBSCAN Clustering Grid"
) -> Dict[str, Any]:
    """
    Generate a grid of HDBSCAN scatter plots for different config combinations.

    This matches the notebook's visualization of running HDBSCAN with multiple
    min_cluster_size and min_samples combinations.

    Args:
        cluster_dfs: List of DataFrames, each with PC1, PC2, cluster columns
        configs: List of (min_cluster_size, min_samples) tuples
        strategy_params: List of strategy parameter column names for hover
        ranking_metric: Metric column name for hover display
        title: Overall plot title

    Returns:
        Plotly JSON dict with subplot grid
    """
    n = len(cluster_dfs)
    if n == 0:
        return {}

    n_cols = min(n, 2)
    n_rows = (n + n_cols - 1) // n_cols

    # Build compact subplot titles with cluster count and noise ratio
    subplot_titles = []
    for i, cfg in enumerate(configs):
        if i < len(cluster_dfs):
            cdf = cluster_dfs[i]
            num_clusters = len([c for c in cdf['cluster'].unique() if c != -1])
            noise_count = len(cdf[cdf['cluster'] == -1])
            total_count = len(cdf)
            noise_ratio = noise_count / total_count * 100 if total_count > 0 else 0
            # Single line compact title
            subplot_titles.append(
                f'size={cfg[0]}, samples={cfg[1]} | C:{num_clusters} N:{noise_ratio:.0f}%'
            )
        else:
            subplot_titles.append(f'size={cfg[0]}, samples={cfg[1]}')

    # Calculate safe vertical spacing - very minimal gap between rows
    if n_rows > 1:
        max_v_spacing = 1.0 / (n_rows - 1) - 0.01
        # Minimal spacing for subplot titles only
        v_spacing = min(0.05 if n_rows <= 2 else 0.04, max_v_spacing)
    else:
        v_spacing = 0.02

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=v_spacing
    )

    for i, cdf in enumerate(cluster_dfs):
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Get unique clusters and sort them
        clusters = sorted(cdf['cluster'].unique())

        for cluster_label in clusters:
            cluster_mask = cdf['cluster'] == cluster_label
            cluster_data = cdf[cluster_mask]

            # Color: gray for noise (-1), bright colors for others
            if cluster_label == -1:
                cluster_color = 'rgba(180, 180, 180, 0.5)'
                name = 'Noise'
            else:
                color_idx = int(cluster_label) % len(CLUSTER_COLORS)
                cluster_color = CLUSTER_COLORS[color_idx]
                name = f'Cluster {cluster_label}'

            # Build hover text with strategy params and ranking metric
            hover_texts = []
            for _, row_data in cluster_data.iterrows():
                parts = [f"<b>Cluster: {cluster_label}</b>"]
                for param in strategy_params:
                    if param in row_data:
                        val = row_data[param]
                        if isinstance(val, (float, np.floating)):
                            parts.append(f"{param}: {val:.4f}")
                        else:
                            parts.append(f"{param}: {val}")
                if ranking_metric in row_data:
                    parts.append(f"{ranking_metric}: {row_data[ranking_metric]:.4f}")
                hover_texts.append('<br>'.join(parts))

            # Convert to lists
            x_data = cluster_data['PC1'].tolist()
            y_data = cluster_data['PC2'].tolist()

            fig.add_trace(
                go.Scattergl(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=cluster_color,
                        line=dict(width=0.5, color='white'),
                        opacity=0.85
                    ),
                    name=name if i == 0 else None,
                    legendgroup=f'cluster_{cluster_label}',
                    showlegend=(i == 0),
                    text=hover_texts,
                    hoverinfo='text'
                ),
                row=row,
                col=col
            )

    # Calculate height based on rows - maximize plot area
    per_row_height = 280 if n_rows > 4 else 320
    total_height = per_row_height * n_rows + 70  # Extra for title with spacing

    fig.update_layout(
        height=total_height,
        width=520 * n_cols + 100,  # Extra width for legend on right
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            y=0.995,
            yanchor='top',
            font=dict(size=14)
        ),
        legend=dict(
            title='Cluster',
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=1.02
        ),
        showlegend=True,
        margin=dict(l=45, r=100, t=60, b=25)
    )

    # Update axis labels for all subplots with tighter standoff
    for i in range(n):
        row = i // n_cols + 1
        col = i % n_cols + 1
        fig.update_xaxes(
            title_text='PC1',
            title_standoff=2,
            row=row,
            col=col,
            gridcolor='lightgray'
        )
        fig.update_yaxes(
            title_text='PC2',
            title_standoff=2,
            row=row,
            col=col,
            gridcolor='lightgray'
        )

    return _fig_to_json(fig)


def generate_hdbscan_core_grid(
    cluster_core_dfs: List[pd.DataFrame],
    configs: List[tuple],
    strategy_params: List[str],
    ranking_metric: str,
    title: str = "HDBSCAN Clustering Grid (Core Points Only)"
) -> Dict[str, Any]:
    """
    Generate a grid of HDBSCAN scatter plots showing only core cluster points.

    Same as generate_hdbscan_grid but for filtered cluster cores (probability >= threshold).

    Args:
        cluster_core_dfs: List of DataFrames with high-probability cluster members
        configs: List of (min_cluster_size, min_samples) tuples
        strategy_params: List of strategy parameter column names for hover
        ranking_metric: Metric column name for hover display
        title: Overall plot title

    Returns:
        Plotly JSON dict with subplot grid
    """
    return generate_hdbscan_grid(
        cluster_dfs=cluster_core_dfs,
        configs=configs,
        strategy_params=strategy_params,
        ranking_metric=ranking_metric,
        title=title
    )
