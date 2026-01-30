"""Visualization functions for optimization pipeline - returns Plotly JSON."""

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    metrics_data: Optional[Dict[str, pd.Series]] = None
) -> Dict[str, Any]:
    """
    Generate interactive PCA scatter plot (PC1 vs PC2) with visible points.

    Args:
        pca_df: DataFrame with PC columns
        hover_data: Optional dict of strategy params for hover
        metrics_data: Optional dict of metrics for hover (sharpe, sortino, profit_factor)

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

    fig.add_trace(go.Scattergl(
        x=x_data,
        y=y_data,
        mode='markers',
        marker=dict(
            size=8,
            color='rgba(59, 130, 246, 0.8)',  # Blue with alpha
            line=dict(width=1, color='rgba(255, 255, 255, 0.8)')
        ),
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
        height=550
    )

    return _fig_to_json(fig)


def generate_cluster_scatter(
    df: pd.DataFrame,
    cluster_col: str,
    hover_cols: List[str],
    title: str = "Cluster Scatter Plot",
    metrics_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate cluster scatter plot colored by cluster with visible points.

    Args:
        df: DataFrame with PC1, PC2, and cluster column
        cluster_col: Name of cluster column
        hover_cols: Strategy param columns to show on hover
        title: Plot title
        metrics_cols: Optional list of metric columns for hover

    Returns:
        Plotly JSON dict
    """
    plot_df = df.copy()
    plot_df['variant_id'] = plot_df.index

    # Get unique clusters and sort them
    clusters = sorted(plot_df[cluster_col].unique())

    fig = go.Figure()

    # Color palette - bright colors for visibility
    colors = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
        '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5'
    ]

    for i, cluster in enumerate(clusters):
        cluster_data = plot_df[plot_df[cluster_col] == cluster]
        color = colors[i % len(colors)]

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
                size=10,
                color=color,
                opacity=0.85,
                line=dict(width=1, color='white')
            ),
            text=hover_texts,
            hoverinfo='text',
            name=f'Cluster {cluster}'
        ))

    fig.update_layout(
        title=dict(
            text=title,
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
        legend_title='Cluster',
        height=550
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
    subtitle: str = ""
) -> go.Figure:
    """Generate a single heatmap figure (static, non-interactive style)."""
    # Create heatmap - using a custom colorscale for better visibility
    # RdYlGn with explicit color stops for better contrast
    if not reverse_colorscale:
        colorscale = [
            [0.0, '#d73027'],    # Red (low/bad)
            [0.25, '#fc8d59'],   # Orange
            [0.5, '#fee08b'],    # Yellow (middle)
            [0.75, '#d9ef8b'],   # Light green
            [1.0, '#1a9850']     # Green (high/good)
        ]
    else:
        colorscale = [
            [0.0, '#1a9850'],    # Green (low/good for drawdown)
            [0.25, '#d9ef8b'],   # Light green
            [0.5, '#fee08b'],    # Yellow (middle)
            [0.75, '#fc8d59'],   # Orange
            [1.0, '#d73027']     # Red (high/bad for drawdown)
        ]

    # Convert pivot data to lists for Plotly, replacing NaN with None
    z_array = pivot_data.values
    z_values = []
    text_values = []
    for row in z_array:
        z_row = []
        text_row = []
        for val in row:
            if pd.isna(val) or not np.isfinite(val):
                z_row.append(None)
                text_row.append('')
            else:
                z_row.append(float(val))
                text_row.append(f'{float(val):.2f}')
        z_values.append(z_row)
        text_values.append(text_row)

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
        hovertemplate=f'{x_param}: %{{x}}<br>{y_param}: %{{y}}<br>{metric_name}: %{{z:.4f}}<extra></extra>',
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

    # 4-param mode: grid of heatmaps
    else:
        const1_values = sorted(df[const_param1].unique())
        const2_values = sorted(df[const_param2].unique())

        for col, title, reverse in available_metrics:
            vmin, vmax = vmin_vmax[col]

            # Create subplot grid
            n_cols = len(const2_values)
            n_rows = len(const1_values)

            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[
                    f'{const_param1}={c1}, {const_param2}={c2}'
                    for c1 in const1_values for c2 in const2_values
                ],
                horizontal_spacing=0.05,
                vertical_spacing=0.08
            )

            for i, const1_val in enumerate(const1_values):
                for j, const2_val in enumerate(const2_values):
                    subset = df[(df[const_param1] == const1_val) & (df[const_param2] == const2_val)]

                    if subset.empty:
                        continue

                    pivot = subset.pivot_table(index=y_param, columns=x_param, values=col, aggfunc='mean')

                    # Ensure consistent axis ordering
                    all_x = sorted(df[x_param].unique())
                    all_y = sorted(df[y_param].unique())
                    pivot = pivot.reindex(index=all_y, columns=all_x)

                    colorscale = 'RdYlGn' if not reverse else 'RdYlGn_r'

                    fig.add_trace(
                        go.Heatmap(
                            z=pivot.values,
                            x=[str(x) for x in pivot.columns],
                            y=[str(y) for y in pivot.index],
                            colorscale=colorscale,
                            zmin=vmin,
                            zmax=vmax,
                            text=np.round(pivot.values, 2),
                            texttemplate='%{text:.2f}',
                            textfont=dict(size=8),
                            showscale=(i == 0 and j == n_cols - 1),
                            colorbar=dict(title=col) if (i == 0 and j == n_cols - 1) else None
                        ),
                        row=i + 1, col=j + 1
                    )

                    # Add highlight rectangles
                    highlight_cells = highlight_cells_map.get((const1_val, const2_val), [])
                    x_vals = list(pivot.columns)
                    y_vals = list(pivot.index)

                    for (x_val, y_val) in highlight_cells:
                        if x_val in x_vals and y_val in y_vals:
                            x_idx = x_vals.index(x_val)
                            y_idx = y_vals.index(y_val)

                            fig.add_shape(
                                type='rect',
                                x0=x_idx - 0.5, x1=x_idx + 0.5,
                                y0=y_idx - 0.5, y1=y_idx + 0.5,
                                line=dict(color='black', width=2),
                                fillcolor='rgba(0,0,0,0)',
                                row=i + 1, col=j + 1
                            )

            fig.update_layout(
                title=f'{title}: {x_param} x {y_param} grid by {const_param1} and {const_param2}',
                height=300 * n_rows,
                width=400 * n_cols
            )

            results.append(_fig_to_json(fig))

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

    subplot_titles = [
        f'min_cluster_size={cfg[0]}, min_samples={cfg[1]}'
        for cfg in configs
    ]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.03,
        vertical_spacing=0.06
    )

    # Bright color palette for visibility
    colors = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
        '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5'
    ]

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
                color_idx = int(cluster_label) % len(colors)
                cluster_color = colors[color_idx]
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

    fig.update_layout(
        height=350 * n_rows,
        width=550 * n_cols,
        title_text=title,
        legend_title='Cluster',
        showlegend=True,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    # Update axis labels for all subplots
    for i in range(n):
        row = i // n_cols + 1
        col = i % n_cols + 1
        fig.update_xaxes(title_text='PC1', row=row, col=col, gridcolor='lightgray')
        fig.update_yaxes(title_text='PC2', row=row, col=col, gridcolor='lightgray')

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
