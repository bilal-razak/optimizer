"""Report generator for optimization results - creates PDF reports using ReportLab."""

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    ListFlowable,
    ListItem,
)

# Figure numbering counter (reset per report)
_figure_counter = 0


def _plotly_to_image_bytes(
    fig_json: Dict[str, Any],
    width: int = 700,
    height: int = 450,
    scale: float = 2.0
) -> bytes:
    """Convert Plotly JSON figure to PNG image bytes."""
    fig = go.Figure(fig_json)
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
        margin=dict(l=50, r=30, t=50, b=50)
    )
    img_bytes = pio.to_image(fig, format='png', width=width, height=height, scale=scale)
    return img_bytes


def _plotly_heatmap_to_rotated_image_bytes(
    fig_json: Dict[str, Any],
    width: int = 700,
    height: int = 500,
    scale: float = 2.0
) -> bytes:
    """Convert Plotly heatmap to image and rotate 90 degrees clockwise."""
    fig = go.Figure(fig_json)
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
        margin=dict(l=60, r=80, t=40, b=50)
    )
    img_bytes = pio.to_image(fig, format='png', width=width, height=height, scale=scale)

    # Rotate 90 degrees clockwise using PIL
    img = PILImage.open(io.BytesIO(img_bytes))
    rotated = img.rotate(-90, expand=True)

    # Save to bytes
    output = io.BytesIO()
    rotated.save(output, format='PNG')
    return output.getvalue()


def _create_image_from_bytes(img_bytes: bytes, width: float, height: float) -> Image:
    """Create a ReportLab Image from bytes."""
    img_buffer = io.BytesIO(img_bytes)
    return Image(img_buffer, width=width, height=height)


def _dataframe_to_table(
    df: pd.DataFrame,
    max_rows: int = 50,
    col_widths: Optional[List[float]] = None,
    int_cols: Optional[List[str]] = None
) -> Table:
    """Convert DataFrame to ReportLab Table."""
    if len(df) > max_rows:
        df = df.head(max_rows)

    int_cols = int_cols or []

    data = [df.columns.tolist()]
    for _, row in df.iterrows():
        row_data = []
        for col, val in zip(df.columns, row):
            if col in int_cols and not pd.isna(val):
                row_data.append(str(int(val)))
            elif isinstance(val, float):
                row_data.append(f'{val:.4f}')
            elif pd.isna(val):
                row_data.append('-')
            else:
                row_data.append(str(val))
        data.append(row_data)

    table = Table(data, colWidths=col_widths)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ])
    table.setStyle(style)
    return table


def _create_combined_cluster_stats_table(
    stats_sharpe: List[Dict[str, Any]],
    stats_sortino: List[Dict[str, Any]],
    stats_profit: List[Dict[str, Any]],
    top_n: int = 5
) -> Table:
    """Create a combined cluster stats table with all three metrics."""
    if not stats_sharpe:
        return Paragraph("No cluster statistics available.", getSampleStyleSheet()['Normal'])

    rows = []
    for i, s in enumerate(stats_sharpe[:top_n]):
        row = {
            'Cluster': int(s.get('cluster_id', i)),
            'Count': int(s.get('count', 0)),
            'Sharpe Mean': s.get('mean', 0),
            'Sharpe Median': s.get('median', 0),
        }
        if i < len(stats_sortino):
            row['Sortino Mean'] = stats_sortino[i].get('mean', 0)
            row['Sortino Median'] = stats_sortino[i].get('median', 0)
        else:
            row['Sortino Mean'] = '-'
            row['Sortino Median'] = '-'
        if i < len(stats_profit):
            row['PF Mean'] = stats_profit[i].get('mean', 0)
            row['PF Median'] = stats_profit[i].get('median', 0)
        else:
            row['PF Mean'] = '-'
            row['PF Median'] = '-'
        rows.append(row)

    df = pd.DataFrame(rows)
    return _dataframe_to_table(df, int_cols=['Cluster', 'Count'])


def _hdbscan_config_to_table(config_results: List[Dict[str, Any]]) -> Table:
    """Convert HDBSCAN config results to ReportLab Table."""
    if not config_results:
        return Paragraph("No configuration results available.", getSampleStyleSheet()['Normal'])

    rows = []
    for cfg in config_results:
        rows.append({
            'Min Cluster Size': int(cfg.get('min_cluster_size', 0)),
            'Min Samples': int(cfg.get('min_samples', 0)),
            'Clusters Found': int(cfg.get('num_clusters', 0))
        })

    df = pd.DataFrame(rows)
    return _dataframe_to_table(df, int_cols=['Min Cluster Size', 'Min Samples', 'Clusters Found'])


def generate_cluster_isolated_scatter(
    hdbscan_df: pd.DataFrame,
    target_cluster_id: int,
    strategy_params: List[str]
) -> Dict[str, Any]:
    """Generate PC1 vs PC2 scatter with target cluster colored and others in light grey."""
    fig = go.Figure()

    other_data = hdbscan_df[hdbscan_df['cluster'] != target_cluster_id]
    target_data = hdbscan_df[hdbscan_df['cluster'] == target_cluster_id]

    if len(other_data) > 0:
        fig.add_trace(go.Scattergl(
            x=other_data['PC1'].tolist(),
            y=other_data['PC2'].tolist(),
            mode='markers',
            marker=dict(size=6, color='rgba(200, 200, 200, 0.4)', line=dict(width=0)),
            name='Other Variants',
            hoverinfo='skip'
        ))

    if len(target_data) > 0:
        hover_texts = []
        for idx, row in target_data.iterrows():
            parts = [f"<b>Cluster {target_cluster_id}</b>", f"Variant: {idx}"]
            for param in strategy_params:
                if param in row:
                    val = row[param]
                    parts.append(f"{param}: {val:.4f}" if isinstance(val, float) else f"{param}: {val}")
            if 'sharpe_ratio' in row:
                parts.append(f"Sharpe: {row['sharpe_ratio']:.4f}")
            hover_texts.append('<br>'.join(parts))

        fig.add_trace(go.Scattergl(
            x=target_data['PC1'].tolist(),
            y=target_data['PC2'].tolist(),
            mode='markers',
            marker=dict(size=10, color='#e41a1c', opacity=0.9, line=dict(width=1, color='white')),
            text=hover_texts,
            hoverinfo='text',
            name=f'Cluster {target_cluster_id}'
        ))

    fig.update_layout(
        title=dict(text=f'Cluster {target_cluster_id} - PC1 vs PC2', x=0.5, xanchor='center'),
        xaxis=dict(title='PC1', gridcolor='lightgray'),
        yaxis=dict(title='PC2', gridcolor='lightgray'),
        height=400,
        margin=dict(l=50, r=30, t=50, b=50),
        showlegend=True
    )

    return json.loads(fig.to_json())


def _update_scatter_with_grey_noise(fig_json: Dict[str, Any]) -> Dict[str, Any]:
    """Update scatter plot to show cluster -1 in light grey."""
    fig = go.Figure(fig_json)
    for trace in fig.data:
        if hasattr(trace, 'name') and trace.name and '-1' in str(trace.name):
            trace.marker.color = 'rgba(200, 200, 200, 0.5)'
            trace.marker.size = 5
    return json.loads(fig.to_json())


def _get_styles():
    """Get custom paragraph styles for the report."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='ReportTitle',
        parent=styles['Title'],
        fontSize=28,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=10,
        spaceBefore=0,
        alignment=1,
    ))

    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=5,
        alignment=1,
    ))

    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1e40af'),
        spaceBefore=12,
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        name='SubsectionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#1e3a8a'),
        spaceBefore=8,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        name='InfoText',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#475569'),
        spaceAfter=3,
        spaceBefore=0,
    ))

    styles.add(ParagraphStyle(
        name='Caption',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#64748b'),
        alignment=1,
        spaceBefore=3,
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        name='FigureCaption',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#475569'),
        alignment=1,
        spaceBefore=5,
        spaceAfter=12,
        fontName='Helvetica-Oblique',
    ))

    styles.add(ParagraphStyle(
        name='ExecutiveSummaryText',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#374151'),
        spaceAfter=8,
        spaceBefore=4,
        leading=16,
    ))

    styles.add(ParagraphStyle(
        name='RecommendationText',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=6,
        spaceBefore=2,
        leftIndent=20,
    ))

    styles.add(ParagraphStyle(
        name='HeaderStyle',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#94a3b8'),
        alignment=1,
    ))

    styles.add(ParagraphStyle(
        name='FooterStyle',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#94a3b8'),
        alignment=2,  # Right aligned
    ))

    return styles


def _reset_figure_counter():
    """Reset the figure counter for a new report."""
    global _figure_counter
    _figure_counter = 0


def _get_next_figure_number() -> int:
    """Get the next figure number and increment counter."""
    global _figure_counter
    _figure_counter += 1
    return _figure_counter


def _create_figure_caption(caption_text: str, styles) -> Paragraph:
    """Create a figure caption with auto-numbering."""
    fig_num = _get_next_figure_number()
    return Paragraph(f"<b>Figure {fig_num}:</b> {caption_text}", styles['FigureCaption'])


def _header_footer(canvas, doc, report_title: str):
    """Draw header and footer on each page."""
    canvas.saveState()

    # Header
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#94a3b8'))
    canvas.drawCentredString(A4[0] / 2, A4[1] - 0.5 * cm, report_title)

    # Footer with page number
    page_num = canvas.getPageNumber()
    canvas.drawRightString(A4[0] - 1 * cm, 0.75 * cm, f"Page {page_num}")
    canvas.drawString(1 * cm, 0.75 * cm, datetime.now().strftime('%Y-%m-%d'))

    canvas.restoreState()


def _create_executive_summary(
    num_variants: int,
    num_shortlisted: int,
    shortlist_enabled: bool,
    hdbscan_num_clusters: int,
    best_cluster_ids: List[int],
    best_cluster_stats: List[Dict[str, Any]],
    ranking_metric: str = 'sharpe_ratio',
    styles=None
) -> List:
    """
    Create the Executive Summary section.

    Returns list of flowable elements.
    """
    if styles is None:
        styles = _get_styles()

    elements = []
    elements.append(Paragraph("Executive Summary", styles['SectionHeader']))
    elements.append(Spacer(1, 0.1 * inch))

    # Key Findings
    elements.append(Paragraph("<b>Key Findings</b>", styles['SubsectionHeader']))

    findings = []
    findings.append(f"Analyzed <b>{num_variants:,}</b> strategy variants")

    if shortlist_enabled and num_shortlisted < num_variants:
        reduction_pct = (1 - num_shortlisted / num_variants) * 100
        findings.append(f"Shortlisting reduced variants by <b>{reduction_pct:.1f}%</b> to {num_shortlisted:,}")

    findings.append(f"HDBSCAN clustering identified <b>{hdbscan_num_clusters}</b> distinct clusters")

    if best_cluster_ids:
        findings.append(f"Top performing clusters: <b>{', '.join(map(str, best_cluster_ids))}</b>")

    for finding in findings:
        elements.append(Paragraph(f"• {finding}", styles['ExecutiveSummaryText']))

    elements.append(Spacer(1, 0.15 * inch))

    # Best Cluster Summary
    if best_cluster_stats and len(best_cluster_stats) > 0:
        elements.append(Paragraph("<b>Best Cluster Performance</b>", styles['SubsectionHeader']))

        for i, stats in enumerate(best_cluster_stats[:3]):  # Top 3
            cluster_id = best_cluster_ids[i] if i < len(best_cluster_ids) else i
            mean_val = stats.get('mean', 0)
            count = stats.get('count', 0)
            elements.append(Paragraph(
                f"• Cluster {cluster_id}: Mean {ranking_metric} = <b>{mean_val:.4f}</b> ({count} variants)",
                styles['ExecutiveSummaryText']
            ))

    elements.append(Spacer(1, 0.2 * inch))

    return elements


def _create_data_quality_section(
    num_variants: int,
    column_info: List[Dict[str, Any]],
    strategy_params: List[str],
    styles=None
) -> List:
    """
    Create the Data Quality Assessment section.

    Returns list of flowable elements.
    """
    if styles is None:
        styles = _get_styles()

    elements = []
    elements.append(Paragraph("Data Quality Assessment", styles['SectionHeader']))
    elements.append(Spacer(1, 0.1 * inch))

    # Dataset Overview
    elements.append(Paragraph("<b>Dataset Overview</b>", styles['SubsectionHeader']))
    elements.append(Paragraph(f"• Total variants analyzed: <b>{num_variants:,}</b>", styles['InfoText']))
    elements.append(Paragraph(f"• Strategy parameters: <b>{', '.join(strategy_params)}</b>", styles['InfoText']))

    # Missing Values Analysis
    if column_info:
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Paragraph("<b>Missing Values Summary</b>", styles['SubsectionHeader']))

        # Find columns with missing values
        cols_with_missing = [c for c in column_info if c.get('null_count', 0) > 0]

        if cols_with_missing:
            missing_data = [['Column', 'Missing Count', 'Missing %']]
            for col in cols_with_missing[:10]:  # Limit to 10
                pct = col['null_count'] / num_variants * 100 if num_variants > 0 else 0
                missing_data.append([col['name'], str(col['null_count']), f"{pct:.1f}%"])

            missing_table = Table(missing_data, colWidths=[3 * inch, 1.5 * inch, 1 * inch])
            missing_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(missing_table)
        else:
            elements.append(Paragraph("✓ No missing values detected in key columns", styles['InfoText']))

    elements.append(Spacer(1, 0.2 * inch))

    return elements


def _create_clustering_interpretation(
    best_cluster_ids: List[int],
    best_cluster_data: List[List[Dict[str, Any]]],
    strategy_params: List[str],
    styles=None
) -> List:
    """
    Create the Clustering Interpretation section explaining what makes each cluster unique.

    Returns list of flowable elements.
    """
    if styles is None:
        styles = _get_styles()

    elements = []
    elements.append(Paragraph("Clustering Interpretation", styles['SectionHeader']))
    elements.append(Spacer(1, 0.1 * inch))

    elements.append(Paragraph(
        "This section describes the characteristic features of each top-performing cluster, "
        "highlighting the parameter ranges and metric distributions that define them.",
        styles['InfoText']
    ))
    elements.append(Spacer(1, 0.15 * inch))

    for i, cluster_id in enumerate(best_cluster_ids):
        if i >= len(best_cluster_data):
            continue

        cluster_variants = best_cluster_data[i]
        if not cluster_variants:
            continue

        elements.append(Paragraph(f"<b>Cluster {cluster_id} Characteristics</b>", styles['SubsectionHeader']))

        # Convert to DataFrame for analysis
        df = pd.DataFrame(cluster_variants)

        # Analyze strategy parameter ranges
        param_summary = []
        for param in strategy_params:
            if param in df.columns:
                min_val = df[param].min()
                max_val = df[param].max()
                mean_val = df[param].mean()

                if min_val == max_val:
                    param_summary.append(f"• <b>{param}</b>: Fixed at {min_val}")
                else:
                    param_summary.append(f"• <b>{param}</b>: Range [{min_val:.4g} - {max_val:.4g}], Mean: {mean_val:.4g}")

        for summary in param_summary:
            elements.append(Paragraph(summary, styles['InfoText']))

        # Key metrics summary
        metrics = ['sharpe_ratio', 'sortino_ratio', 'profit_factor']
        metric_summaries = []
        for metric in metrics:
            if metric in df.columns:
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                metric_summaries.append(f"{metric}: {mean_val:.4f} (±{std_val:.4f})")

        if metric_summaries:
            elements.append(Paragraph(f"• <b>Metrics</b>: {', '.join(metric_summaries)}", styles['InfoText']))

        elements.append(Spacer(1, 0.1 * inch))

    return elements


def _create_statistical_validation_section(
    silhouette_score: Optional[float],
    noise_ratio: Optional[float],
    num_clusters: int,
    cluster_sizes: List[int],
    styles=None
) -> List:
    """
    Create the Statistical Validation section with clustering quality metrics.

    Returns list of flowable elements.
    """
    if styles is None:
        styles = _get_styles()

    elements = []
    elements.append(Paragraph("Statistical Validation", styles['SectionHeader']))
    elements.append(Spacer(1, 0.1 * inch))

    elements.append(Paragraph("<b>Clustering Quality Metrics</b>", styles['SubsectionHeader']))

    # Silhouette Score interpretation
    if silhouette_score is not None:
        score_quality = "poor"
        if silhouette_score > 0.7:
            score_quality = "strong"
        elif silhouette_score > 0.5:
            score_quality = "reasonable"
        elif silhouette_score > 0.25:
            score_quality = "weak"

        elements.append(Paragraph(
            f"• <b>Silhouette Score</b>: {silhouette_score:.4f} ({score_quality} cluster separation)",
            styles['InfoText']
        ))

    # Noise ratio
    if noise_ratio is not None:
        noise_quality = "low" if noise_ratio < 0.1 else ("moderate" if noise_ratio < 0.3 else "high")
        elements.append(Paragraph(
            f"• <b>Noise Ratio</b>: {noise_ratio * 100:.1f}% of points classified as noise ({noise_quality})",
            styles['InfoText']
        ))

    # Cluster count and sizes
    elements.append(Paragraph(f"• <b>Number of Clusters</b>: {num_clusters}", styles['InfoText']))

    if cluster_sizes:
        min_size = min(cluster_sizes)
        max_size = max(cluster_sizes)
        mean_size = sum(cluster_sizes) / len(cluster_sizes)
        elements.append(Paragraph(
            f"• <b>Cluster Sizes</b>: Min={min_size}, Max={max_size}, Mean={mean_size:.1f}",
            styles['InfoText']
        ))

    elements.append(Spacer(1, 0.15 * inch))

    # Interpretation guidance
    elements.append(Paragraph("<b>Interpretation Guide</b>", styles['SubsectionHeader']))
    elements.append(Paragraph(
        "• Silhouette Score ranges from -1 to 1. Values > 0.5 indicate well-separated clusters.",
        styles['InfoText']
    ))
    elements.append(Paragraph(
        "• Lower noise ratios suggest cleaner cluster boundaries and more robust parameter groupings.",
        styles['InfoText']
    ))
    elements.append(Paragraph(
        "• Balanced cluster sizes typically indicate more generalizable parameter insights.",
        styles['InfoText']
    ))

    elements.append(Spacer(1, 0.2 * inch))

    return elements


def generate_report(
    report_title: str,
    csv_path: str,
    strategy_params: List[str],
    x_param: str,
    y_param: str,
    const_param: Optional[str],
    shortlist_enabled: bool,
    shortlist_conditions: List[Dict[str, Any]],
    kmeans_k: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    core_probability_threshold: float,
    num_variants: int,
    num_shortlisted: int,
    initial_heatmaps: List[Dict[str, Any]],
    shortlisted_heatmaps: Optional[List[Dict[str, Any]]],
    pca_variance_chart: Dict[str, Any],
    explained_variance: List[float],
    cumulative_variance: List[float],
    kmeans_scatter: Dict[str, Any],
    kmeans_stats_sharpe: List[Dict[str, Any]],
    kmeans_stats_sortino: List[Dict[str, Any]],
    kmeans_stats_profit: List[Dict[str, Any]],
    kmeans_best_cluster_size: int,
    hdbscan_grid_chart_all: Dict[str, Any],
    hdbscan_grid_chart_core: Dict[str, Any],
    hdbscan_config_results: List[Dict[str, Any]],
    hdbscan_scatter_all: Dict[str, Any],
    hdbscan_scatter_core: Dict[str, Any],
    hdbscan_stats_sharpe: List[Dict[str, Any]],
    hdbscan_stats_sortino: List[Dict[str, Any]],
    hdbscan_stats_profit: List[Dict[str, Any]],
    hdbscan_num_clusters: int,
    hdbscan_num_core_points: int,
    best_cluster_ids: List[int],
    best_cluster_heatmaps: List[List[Dict[str, Any]]],
    best_cluster_scatter_plots: List[Dict[str, Any]],
    best_cluster_data: List[List[Dict[str, Any]]],
    best_cluster_core_counts: List[int],
    save_path: Optional[str] = None,
    filename: Optional[str] = None,
    silhouette_score: Optional[float] = None,
    noise_ratio: Optional[float] = None,
    column_info: Optional[List[Dict[str, Any]]] = None
) -> Tuple[bytes, str]:
    """Generate the complete PDF report."""
    # Reset figure counter for new report
    _reset_figure_counter()

    styles = _get_styles()
    buffer = io.BytesIO()

    # Create document - portrait only
    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1*cm,
        leftMargin=1*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm,
    )

    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        A4[0] - doc.leftMargin - doc.rightMargin,
        A4[1] - doc.topMargin - doc.bottomMargin,
        id='main'
    )

    # Create page template with header/footer
    def on_page(canvas, doc):
        _header_footer(canvas, doc, report_title)

    doc.addPageTemplates([PageTemplate(id='main', frames=frame, onPage=on_page)])

    story = []

    # Image dimensions
    img_width = 7 * inch
    img_height = 4.5 * inch

    # Rotated heatmap dimensions (height becomes width after rotation)
    rotated_heatmap_width = 7 * inch
    rotated_heatmap_height = 9 * inch  # Taller to fill page

    # ===== PAGE 1: Title Page =====
    story.append(Spacer(1, 2.5*inch))
    story.append(Paragraph(report_title, styles['ReportTitle']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Trading Strategy Parameter Optimization Report", styles['Subtitle']))
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['InfoText']))
    story.append(PageBreak())

    # ===== PAGE 2: Executive Summary (NEW) =====
    exec_summary_elements = _create_executive_summary(
        num_variants=num_variants,
        num_shortlisted=num_shortlisted,
        shortlist_enabled=shortlist_enabled,
        hdbscan_num_clusters=hdbscan_num_clusters,
        best_cluster_ids=best_cluster_ids,
        best_cluster_stats=hdbscan_stats_sharpe,
        ranking_metric='sharpe_ratio',
        styles=styles
    )
    story.extend(exec_summary_elements)
    story.append(PageBreak())

    # ===== PAGE 3: Configuration Summary =====
    story.append(Paragraph("Optimization Configuration", styles['SectionHeader']))
    story.append(Spacer(1, 0.15*inch))

    config_data = [
        ['Parameter', 'Value'],
        ['Data Source', Path(csv_path).name],
        ['Strategy Parameters', ', '.join(strategy_params)],
        ['Total Variants Backtested', str(num_variants)],
        ['Heatmap X-Axis', x_param],
        ['Heatmap Y-Axis', y_param],
        ['Heatmap Constant Parameter', const_param if const_param else 'None'],
        ['K-Means K Value', str(kmeans_k)],
        ['HDBSCAN Min Cluster Size', str(hdbscan_min_cluster_size)],
        ['HDBSCAN Min Samples', str(hdbscan_min_samples)],
        ['Core Probability Threshold', str(core_probability_threshold)],
    ]

    config_table = Table(config_data, colWidths=[2.5*inch, 4*inch])
    config_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f1f5f9')),
    ]))
    story.append(config_table)
    story.append(Spacer(1, 0.2*inch))

    if shortlist_enabled and shortlist_conditions:
        story.append(Paragraph("Shortlist Conditions Applied", styles['SubsectionHeader']))
        for cond in shortlist_conditions:
            story.append(Paragraph(f"  - {cond['metric']} {cond['operator']} {cond['value']}", styles['InfoText']))
        story.append(Paragraph(f"Variants after shortlist: {num_shortlisted}", styles['InfoText']))

    story.append(PageBreak())

    # ===== Data Quality Assessment (NEW) =====
    data_quality_elements = _create_data_quality_section(
        num_variants=num_variants,
        column_info=column_info or [],
        strategy_params=strategy_params,
        styles=styles
    )
    story.extend(data_quality_elements)
    story.append(PageBreak())

    # ===== SECTION: Initial Heatmaps (Rotated) =====
    story.append(Paragraph("Parameter Landscape - Sharpe Ratio Heatmaps", styles['SectionHeader']))

    sharpe_heatmaps = [hm for hm in initial_heatmaps
                       if hm.get('_metadata', {}).get('metric_col') == 'sharpe_ratio'
                       or hm.get('_metadata', {}).get('metric') == 'Sharpe Ratio']

    for i, hm in enumerate(sharpe_heatmaps):
        # Rotate heatmap 90 degrees clockwise
        img_bytes = _plotly_heatmap_to_rotated_image_bytes(hm, width=900, height=600)
        img = _create_image_from_bytes(img_bytes, rotated_heatmap_width, rotated_heatmap_height)
        story.append(img)

        # Add figure caption
        const_info = hm.get('_metadata', {}).get('const_value', '')
        caption_text = f"Sharpe Ratio Heatmap - {x_param} vs {y_param}"
        if const_info:
            caption_text += f" ({const_param}={const_info})"
        story.append(_create_figure_caption(caption_text, styles))

        if i < len(sharpe_heatmaps) - 1:
            story.append(PageBreak())

    story.append(PageBreak())

    # ===== SECTION: Shortlisted Heatmaps (if enabled) =====
    if shortlist_enabled and shortlisted_heatmaps:
        story.append(Paragraph("Shortlisted Variants - Highlighted Heatmaps", styles['SectionHeader']))

        sharpe_shortlisted = [hm for hm in shortlisted_heatmaps
                              if hm.get('_metadata', {}).get('metric_col') == 'sharpe_ratio'
                              or hm.get('_metadata', {}).get('metric') == 'Sharpe Ratio']

        for i, hm in enumerate(sharpe_shortlisted):
            img_bytes = _plotly_heatmap_to_rotated_image_bytes(hm, width=900, height=600)
            img = _create_image_from_bytes(img_bytes, rotated_heatmap_width, rotated_heatmap_height)
            story.append(img)

            # Add figure caption
            const_info = hm.get('_metadata', {}).get('const_value', '')
            caption_text = f"Shortlisted Variants Heatmap - {x_param} vs {y_param}"
            if const_info:
                caption_text += f" ({const_param}={const_info})"
            story.append(_create_figure_caption(caption_text, styles))

            if i < len(sharpe_shortlisted) - 1:
                story.append(PageBreak())

        story.append(PageBreak())

    # ===== SECTION: PCA Analysis =====
    story.append(Paragraph("PCA Analysis", styles['SectionHeader']))
    story.append(Paragraph("Variance Explained by Principal Components", styles['SubsectionHeader']))

    pca_data = [['Component', 'Variance Explained (%)', 'Cumulative (%)']]
    for i, (var, cum) in enumerate(zip(explained_variance, cumulative_variance)):
        pca_data.append([f'PC{i+1}', f'{var*100:.2f}%', f'{cum*100:.2f}%'])

    pca_table = Table(pca_data, colWidths=[1.5*inch, 2*inch, 2*inch])
    pca_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
    ]))
    story.append(pca_table)
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Variance Plot", styles['SubsectionHeader']))
    pca_img_bytes = _plotly_to_image_bytes(pca_variance_chart, width=700, height=350)
    pca_img = _create_image_from_bytes(pca_img_bytes, img_width, 3.5*inch)
    story.append(pca_img)
    story.append(_create_figure_caption("PCA Explained Variance by Principal Component", styles))
    story.append(PageBreak())

    # ===== SECTION: K-Means Clustering =====
    story.append(Paragraph("K-Means Clustering", styles['SectionHeader']))
    story.append(Paragraph(f"<b>K Used:</b> {kmeans_k} | <b>Best Cluster Size:</b> {kmeans_best_cluster_size}", styles['InfoText']))
    story.append(Spacer(1, 0.1*inch))

    kmeans_img_bytes = _plotly_to_image_bytes(kmeans_scatter, width=700, height=400)
    kmeans_img = _create_image_from_bytes(kmeans_img_bytes, img_width, 4*inch)
    story.append(kmeans_img)
    story.append(_create_figure_caption(f"K-Means Clustering (K={kmeans_k}) - PC1 vs PC2", styles))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("Top 5 Cluster Statistics", styles['SubsectionHeader']))
    kmeans_table = _create_combined_cluster_stats_table(
        kmeans_stats_sharpe, kmeans_stats_sortino, kmeans_stats_profit, top_n=5
    )
    story.append(kmeans_table)
    story.append(PageBreak())

    # ===== SECTION: HDBSCAN Grid Search - 3 pages =====
    # Page 1: All points grid (increased height for better subplot visibility)
    story.append(Paragraph("HDBSCAN Grid Search - All Points", styles['SectionHeader']))
    grid_all_img_bytes = _plotly_to_image_bytes(hdbscan_grid_chart_all, width=900, height=1100, scale=2.5)
    grid_all_img = _create_image_from_bytes(grid_all_img_bytes, img_width, 9.5*inch)
    story.append(grid_all_img)
    story.append(_create_figure_caption("HDBSCAN Grid Search Results - All Data Points", styles))
    story.append(PageBreak())

    # Page 2: Core only grid (increased height for better subplot visibility)
    story.append(Paragraph("HDBSCAN Grid Search - Core Points Only", styles['SectionHeader']))
    grid_core_img_bytes = _plotly_to_image_bytes(hdbscan_grid_chart_core, width=900, height=1100, scale=2.5)
    grid_core_img = _create_image_from_bytes(grid_core_img_bytes, img_width, 9.5*inch)
    story.append(grid_core_img)
    story.append(_create_figure_caption(f"HDBSCAN Grid Search Results - Core Points (prob ≥ {core_probability_threshold})", styles))
    story.append(PageBreak())

    # Page 3: Configuration results table
    story.append(Paragraph("HDBSCAN Configuration Results", styles['SectionHeader']))
    config_table = _hdbscan_config_to_table(hdbscan_config_results)
    story.append(config_table)
    story.append(PageBreak())

    # ===== SECTION: Final HDBSCAN - Both plots on same page =====
    story.append(Paragraph("Final HDBSCAN Clustering", styles['SectionHeader']))
    story.append(Paragraph(
        f"<b>Config:</b> min_cluster_size={hdbscan_min_cluster_size}, min_samples={hdbscan_min_samples} | "
        f"<b>Clusters:</b> {hdbscan_num_clusters} | <b>Core Points:</b> {hdbscan_num_core_points}",
        styles['InfoText']
    ))
    story.append(Spacer(1, 0.05*inch))

    hdbscan_scatter_all_grey = _update_scatter_with_grey_noise(hdbscan_scatter_all)
    hdbscan_scatter_core_grey = _update_scatter_with_grey_noise(hdbscan_scatter_core)

    story.append(Paragraph("All Points", styles['SubsectionHeader']))
    hdbscan_all_img_bytes = _plotly_to_image_bytes(hdbscan_scatter_all_grey, width=700, height=300)
    hdbscan_all_img = _create_image_from_bytes(hdbscan_all_img_bytes, img_width, 3*inch)
    story.append(hdbscan_all_img)
    story.append(_create_figure_caption(f"Final HDBSCAN Clustering - All Points ({hdbscan_num_clusters} clusters)", styles))

    story.append(Paragraph("Core Points Only", styles['SubsectionHeader']))
    hdbscan_core_img_bytes = _plotly_to_image_bytes(hdbscan_scatter_core_grey, width=700, height=300)
    hdbscan_core_img = _create_image_from_bytes(hdbscan_core_img_bytes, img_width, 3*inch)
    story.append(hdbscan_core_img)
    story.append(_create_figure_caption(f"Final HDBSCAN Clustering - Core Points Only ({hdbscan_num_core_points} points)", styles))

    story.append(Paragraph("Cluster Statistics", styles['SubsectionHeader']))
    hdbscan_table = _create_combined_cluster_stats_table(
        hdbscan_stats_sharpe, hdbscan_stats_sortino, hdbscan_stats_profit, top_n=10
    )
    story.append(hdbscan_table)
    story.append(PageBreak())

    # ===== SECTION: Clustering Interpretation =====
    clustering_interp_elements = _create_clustering_interpretation(
        best_cluster_ids=best_cluster_ids,
        best_cluster_data=best_cluster_data,
        strategy_params=strategy_params,
        styles=styles
    )
    story.extend(clustering_interp_elements)
    story.append(PageBreak())

    # ===== SECTION: Best Clusters =====
    story.append(Paragraph("Best Clusters Analysis", styles['SectionHeader']))
    story.append(Paragraph(f"<b>Best Cluster IDs:</b> {', '.join(map(str, best_cluster_ids))} | <b>Ranked by:</b> Mean Sharpe Ratio", styles['InfoText']))
    story.append(Spacer(1, 0.1*inch))

    for i, cluster_id in enumerate(best_cluster_ids):
        story.append(Paragraph(f"Cluster {cluster_id}", styles['SubsectionHeader']))
        core_count = best_cluster_core_counts[i] if i < len(best_cluster_core_counts) else 0
        cluster_variants = best_cluster_data[i]
        story.append(Paragraph(f"{len(cluster_variants)} variants ({core_count} core points)", styles['InfoText']))

        # 1. Variants table first
        story.append(Paragraph("Variants", styles['SubsectionHeader']))
        if cluster_variants:
            df = pd.DataFrame(cluster_variants)
            display_cols = ['variant_id'] + strategy_params + ['sharpe_ratio', 'sortino_ratio', 'profit_factor']
            display_cols = [c for c in display_cols if c in df.columns]
            df = df[display_cols]
            variants_table = _dataframe_to_table(df, max_rows=20)
            story.append(variants_table)
        else:
            story.append(Paragraph("No variant data available.", styles['InfoText']))

        # 2. PC1 vs PC2 scatter plot
        story.append(Paragraph("PC1 vs PC2 (Cluster Highlighted)", styles['SubsectionHeader']))
        scatter_img_bytes = _plotly_to_image_bytes(best_cluster_scatter_plots[i], width=700, height=350)
        scatter_img = _create_image_from_bytes(scatter_img_bytes, img_width, 3.5*inch)
        story.append(scatter_img)
        story.append(_create_figure_caption(f"Cluster {cluster_id} highlighted in PCA space", styles))
        story.append(PageBreak())

        # 3. Heatmaps (rotated)
        story.append(Paragraph(f"Cluster {cluster_id} - Sharpe Ratio Heatmaps (Highlighted)", styles['SectionHeader']))

        cluster_sharpe_heatmaps = [hm for hm in best_cluster_heatmaps[i]
                                   if hm.get('_metadata', {}).get('metric_col') == 'sharpe_ratio'
                                   or hm.get('_metadata', {}).get('metric') == 'Sharpe Ratio']

        for j, hm in enumerate(cluster_sharpe_heatmaps):
            hm_img_bytes = _plotly_heatmap_to_rotated_image_bytes(hm, width=900, height=600)
            hm_img = _create_image_from_bytes(hm_img_bytes, rotated_heatmap_width, rotated_heatmap_height)
            story.append(hm_img)

            # Add figure caption
            const_info = hm.get('_metadata', {}).get('const_value', '')
            caption_text = f"Cluster {cluster_id} Sharpe Ratio Heatmap"
            if const_info:
                caption_text += f" ({const_param}={const_info})"
            story.append(_create_figure_caption(caption_text, styles))

            if j < len(cluster_sharpe_heatmaps) - 1:
                story.append(PageBreak())

        if i < len(best_cluster_ids) - 1:
            story.append(PageBreak())

    # Build PDF
    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    # Generate filename
    if filename:
        if not filename.endswith('.pdf'):
            filename = filename + '.pdf'
    else:
        safe_title = "".join(c if c.isalnum() or c in ' -_' else '_' for c in report_title)
        safe_title = safe_title.replace(' ', '_')[:50]
        filename = f"optimization_report_{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        if save_path.is_dir():
            save_path = save_path / filename
        with open(save_path, 'wb') as f:
            f.write(pdf_bytes)

    return pdf_bytes, filename
