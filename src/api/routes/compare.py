"""API endpoints for the Compare Variants feature."""

import io
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.models.compare import (
    ChartType,
    CompareCalculateMetricsRequest,
    CompareCalculateMetricsResponse,
    CompareExportRequest,
    CompareGenerateChartRequest,
    CompareGenerateChartResponse,
    CompareLoadRequest,
    CompareLoadResponse,
    VariantInfo,
    VariantMetrics,
    AggregateStats,
)
from src.comparison.metrics import calculate_variant_metrics, calculate_aggregate_stats
from src.comparison.visualization import (
    generate_comparison_table,
    generate_grouped_bar_chart,
    generate_radar_chart,
    generate_cumulative_line_chart,
    generate_drawdown_chart,
    generate_ranking_table,
    get_metric_display,
    METRIC_DISPLAY,
    METRIC_SECTIONS,
)

router = APIRouter(prefix="/compare", tags=["compare-variants"])

# Session storage (in-memory)
sessions: Dict[str, Dict[str, Any]] = {}


def _parse_expiry_date(date_val: Any) -> Optional[pd.Timestamp]:
    """Parse expiry date from various formats."""
    if pd.isna(date_val):
        return None
    try:
        if isinstance(date_val, str):
            cleaned = date_val.strip().strip('"').strip("'")
            return pd.to_datetime(cleaned)
        return pd.to_datetime(date_val)
    except (ValueError, TypeError):
        return None


def align_variant_data_to_common_dates(
    variant_data: Dict[str, pd.DataFrame],
    expiry_column: str = "Expiry"
) -> Tuple[Dict[str, pd.DataFrame], Optional[str], Optional[str]]:
    """
    Align all variant DataFrames to a common date range.

    Finds the intersection of expiry dates across all variants:
    - Common start date = max of all min expiry dates
    - Common end date = min of all max expiry dates

    Args:
        variant_data: Dictionary mapping variant names to DataFrames
        expiry_column: Column name for expiry date

    Returns:
        Tuple of (filtered_variant_data, common_start_date, common_end_date)
    """
    if not variant_data:
        return {}, None, None

    # Parse expiry dates for all variants and find date ranges
    date_ranges = {}
    for name, df in variant_data.items():
        if expiry_column not in df.columns:
            continue

        parsed_dates = df[expiry_column].apply(_parse_expiry_date).dropna()
        if len(parsed_dates) > 0:
            date_ranges[name] = {
                "min": parsed_dates.min(),
                "max": parsed_dates.max()
            }

    if not date_ranges:
        # No valid dates found, return original data
        return variant_data, None, None

    # Find common date range (intersection)
    common_start = max(dr["min"] for dr in date_ranges.values())
    common_end = min(dr["max"] for dr in date_ranges.values())

    if common_start > common_end:
        # No overlapping dates, return original data with warning
        return variant_data, None, None

    # Filter each variant to the common date range
    filtered_data = {}
    for name, df in variant_data.items():
        if expiry_column not in df.columns:
            filtered_data[name] = df
            continue

        df_copy = df.copy()
        df_copy['_parsed_expiry'] = df_copy[expiry_column].apply(_parse_expiry_date)

        # Filter to common date range
        mask = (df_copy['_parsed_expiry'] >= common_start) & (df_copy['_parsed_expiry'] <= common_end)
        df_filtered = df_copy[mask].drop(columns=['_parsed_expiry']).reset_index(drop=True)

        filtered_data[name] = df_filtered

    return filtered_data, common_start.strftime('%Y-%m-%d'), common_end.strftime('%Y-%m-%d')


def get_session(session_id: str) -> Dict[str, Any]:
    """Get session data or raise 404."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}. Please load files first."
        )
    return sessions[session_id]


@router.get("/browse")
async def browse_directory(path: str = "") -> Dict[str, Any]:
    """
    Browse filesystem to select CSV files.

    Args:
        path: Directory path to browse (empty for home directory)

    Returns:
        Directory listing with files and subdirectories
    """
    if not path:
        path = os.path.expanduser("~")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

    try:
        entries = os.listdir(path)
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {path}")

    directories = []
    files = []

    for entry in sorted(entries):
        if entry.startswith('.'):
            continue  # Skip hidden files

        full_path = os.path.join(path, entry)

        if os.path.isdir(full_path):
            directories.append({
                "name": entry,
                "path": full_path
            })
        elif entry.lower().endswith('.csv'):
            try:
                size = os.path.getsize(full_path)
            except OSError:
                size = 0

            files.append({
                "name": entry,
                "path": full_path,
                "size": size,
                "size_display": _format_size(size)
            })

    # Get parent directory
    parent_path = str(Path(path).parent)

    return {
        "current_path": path,
        "parent_path": parent_path if parent_path != path else None,
        "directories": directories,
        "files": files
    }


def _format_size(size: int) -> str:
    """Format file size for display."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


@router.post("/load", response_model=CompareLoadResponse)
async def load_files(request: CompareLoadRequest) -> CompareLoadResponse:
    """
    Load and validate multiple CSV files for comparison.

    Each CSV file represents one variant to compare.

    Args:
        request: List of CSV file paths

    Returns:
        Session ID and variant information
    """
    variants = []
    variant_data = {}
    columns = None
    preview_data = {}

    for csv_path in request.csv_paths:
        if not os.path.exists(csv_path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {csv_path}"
            )

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error reading {csv_path}: {str(e)}"
            )

        # Extract variant name from filename
        name = Path(csv_path).stem

        # Store columns (use first file as reference)
        if columns is None:
            columns = df.columns.tolist()

        # Create variant info
        variants.append(VariantInfo(
            name=name,
            file_path=csv_path,
            num_trades=len(df)
        ))

        # Store data for later use
        variant_data[name] = df

        # Create preview (first 5 rows)
        preview_df = df.head(5)
        preview_records = []
        for _, row in preview_df.iterrows():
            record = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    record[col] = None
                elif isinstance(val, (np.integer, np.floating)):
                    record[col] = float(val) if isinstance(val, np.floating) else int(val)
                else:
                    record[col] = str(val) if not isinstance(val, str) else val
            preview_records.append(record)
        preview_data[name] = preview_records

    # Create session
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "csv_paths": request.csv_paths,
        "variants": [v.model_dump() for v in variants],
        "variant_data": variant_data,
        "columns": columns,
        "variant_metrics": None,
        "aggregate_stats": None,
    }

    return CompareLoadResponse(
        session_id=session_id,
        num_files=len(variants),
        variants=variants,
        columns=columns,
        preview_data=preview_data
    )


@router.post("/calculate-metrics", response_model=CompareCalculateMetricsResponse)
async def calculate_metrics(request: CompareCalculateMetricsRequest) -> CompareCalculateMetricsResponse:
    """
    Calculate performance metrics for all loaded variants.

    IMPORTANT: All variants are aligned to a common date range before calculating
    metrics to ensure fair comparison. The common range is the intersection of
    all variants' expiry dates (max of min dates to min of max dates).

    Args:
        request: Session ID and column mappings

    Returns:
        Calculated metrics for each variant and aggregate stats
    """
    session = get_session(request.session_id)
    variant_data = session["variant_data"]

    # Validate columns exist
    sample_df = next(iter(variant_data.values()))
    if request.pnl_column not in sample_df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"PnL column '{request.pnl_column}' not found. Available: {sample_df.columns.tolist()}"
        )

    # Align all variants to common date range
    aligned_data, common_start, common_end = align_variant_data_to_common_dates(
        variant_data, request.expiry_column
    )

    # Store aligned data and common date range in session for chart generation
    session["aligned_variant_data"] = aligned_data
    session["common_date_range"] = {
        "start": common_start,
        "end": common_end
    }

    variant_metrics = []

    for name, df in aligned_data.items():
        try:
            metrics = calculate_variant_metrics(
                df=df,
                name=name,
                pnl_column=request.pnl_column,
                trades_column=request.trades_column,
                min_notional_column=request.min_notional_column,
                expiry_column=request.expiry_column,
                threshold_x_pct=request.threshold_x_pct,
                threshold_y_pct=request.threshold_y_pct
            )
            variant_metrics.append(VariantMetrics(**metrics))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error calculating metrics for {name}: {str(e)}"
            )

    # Calculate aggregate stats
    metrics_dicts = [vm.model_dump() for vm in variant_metrics]
    agg_stats = calculate_aggregate_stats(metrics_dicts)

    # Store in session
    session["variant_metrics"] = metrics_dicts
    session["aggregate_stats"] = agg_stats
    session["pnl_column"] = request.pnl_column
    session["trades_column"] = request.trades_column
    session["min_notional_column"] = request.min_notional_column
    session["expiry_column"] = request.expiry_column
    session["threshold_x_pct"] = request.threshold_x_pct
    session["threshold_y_pct"] = request.threshold_y_pct

    # Build common date range response
    common_date_range = None
    if common_start and common_end:
        from src.models.compare import CommonDateRange
        common_date_range = CommonDateRange(start=common_start, end=common_end)

    return CompareCalculateMetricsResponse(
        session_id=request.session_id,
        variant_metrics=variant_metrics,
        aggregate_stats=AggregateStats(**agg_stats),
        common_date_range=common_date_range
    )


@router.post("/generate-chart", response_model=CompareGenerateChartResponse)
async def generate_chart(request: CompareGenerateChartRequest) -> CompareGenerateChartResponse:
    """
    Generate a comparison chart.

    Args:
        request: Session ID, chart type, and options

    Returns:
        Chart data (Plotly JSON or table data)
    """
    session = get_session(request.session_id)

    variant_metrics = session.get("variant_metrics")
    if variant_metrics is None:
        raise HTTPException(
            status_code=400,
            detail="Metrics not calculated yet. Call /calculate-metrics first."
        )

    # Use aligned variant data for charts (ensures same date range)
    variant_data = session.get("aligned_variant_data", session.get("variant_data", {}))
    pnl_column = session.get("pnl_column", "Pnl%")

    # Filter variants if specified
    if request.selected_variants:
        variant_metrics = [
            vm for vm in variant_metrics
            if vm["name"] in request.selected_variants
        ]
        variant_data = {
            k: v for k, v in variant_data.items()
            if k in request.selected_variants
        }

    chart_data = {}
    table_data = None
    alias_map = request.alias_map or {}

    # Get threshold values from session
    threshold_x = session.get("threshold_x_pct", -5.0)
    threshold_y = session.get("threshold_y_pct", -10.0)

    try:
        if request.chart_type == ChartType.TABLE:
            result = generate_comparison_table(
                variant_metrics, request.metrics, alias_map,
                threshold_x=threshold_x, threshold_y=threshold_y
            )
            table_data = result.get("table_data")
            chart_data = {}

        elif request.chart_type == ChartType.GROUPED_BAR:
            chart_data = generate_grouped_bar_chart(
                variant_metrics,
                request.metrics,
                request.normalize,
                alias_map
            )

        elif request.chart_type == ChartType.RADAR:
            chart_data = generate_radar_chart(variant_metrics, request.metrics, alias_map)

        elif request.chart_type == ChartType.RANKING_TABLE:
            result = generate_ranking_table(variant_metrics, request.metrics, alias_map)
            table_data = result.get("ranking_data")
            chart_data = {}

        elif request.chart_type == ChartType.LINE:
            chart_data = generate_cumulative_line_chart(
                variant_data,
                pnl_column,
                date_from=request.date_from,
                date_to=request.date_to,
                alias_map=alias_map
            )

        elif request.chart_type == ChartType.DRAWDOWN:
            chart_data = generate_drawdown_chart(
                variant_data,
                pnl_column,
                date_from=request.date_from,
                date_to=request.date_to,
                alias_map=alias_map
            )

        # New chart types will be added later
        elif request.chart_type in [
            ChartType.ROLLING_HEATMAP,
            ChartType.RISK_RETURN_SCATTER,
            ChartType.MONTHLY_HEATMAP,
            ChartType.UNDERWATER
        ]:
            raise HTTPException(
                status_code=501,
                detail=f"Chart type '{request.chart_type}' not yet implemented"
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown chart type: {request.chart_type}"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating chart: {str(e)}"
        )

    return CompareGenerateChartResponse(
        session_id=request.session_id,
        chart_type=request.chart_type,
        chart_data=chart_data,
        table_data=table_data
    )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str) -> Dict[str, str]:
    """Delete a session to free up memory."""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")


@router.get("/sessions")
async def list_sessions() -> Dict[str, List[str]]:
    """List all active session IDs."""
    return {"sessions": list(sessions.keys())}


@router.post("/export-csv")
async def export_csv(request: CompareExportRequest) -> StreamingResponse:
    """
    Export comparison table as CSV file.

    Args:
        request: Session ID and alias map

    Returns:
        CSV file as streaming response
    """
    session = get_session(request.session_id)

    variant_metrics = session.get("variant_metrics")
    if variant_metrics is None:
        raise HTTPException(
            status_code=400,
            detail="Metrics not calculated yet. Call /calculate-metrics first."
        )

    alias_map = request.alias_map or {}
    threshold_x = session.get("threshold_x_pct", -5.0)
    threshold_y = session.get("threshold_y_pct", -10.0)
    custom_title = request.title or "Variant Comparison Report"

    # Filter variant metrics by selected variants
    if request.selected_variants:
        variant_metrics = [vm for vm in variant_metrics if vm["name"] in request.selected_variants]

    # Get variant names (use aliases if provided)
    variant_names = []
    for vm in variant_metrics:
        original_name = vm["name"]
        display_name = alias_map.get(original_name, original_name)
        variant_names.append(display_name)

    # Build CSV data with title and section headers
    rows = []

    # Add title row
    title_row = {"Metric": custom_title}
    for vn in variant_names:
        title_row[vn] = ""
    rows.append(title_row)

    # Add timestamp row
    timestamp_display = datetime.now().strftime("%Y-%m-%d %H:%M")
    timestamp_row = {"Metric": f"Generated: {timestamp_display}"}
    for vn in variant_names:
        timestamp_row[vn] = ""
    rows.append(timestamp_row)

    # Add empty row for spacing
    empty_row = {"Metric": ""}
    for vn in variant_names:
        empty_row[vn] = ""
    rows.append(empty_row)

    for section in METRIC_SECTIONS:
        section_name = section["name"]
        section_metrics = section["metrics"]

        # Filter to valid metrics
        valid_metrics = [m for m in section_metrics if m in METRIC_DISPLAY or m in ["weeks_below_x_pct", "weeks_min_notional_below_y_pct"]]

        if not valid_metrics:
            continue

        # Add section header row (uppercase section name, empty cells for variants)
        section_row = {"Metric": f"--- {section_name.upper()} ---"}
        for vm in variant_metrics:
            variant_display = alias_map.get(vm["name"], vm["name"])
            section_row[variant_display] = ""
        rows.append(section_row)

        # Add metrics in this section
        for metric_key in valid_metrics:
            display_name, fmt = get_metric_display(metric_key, threshold_x, threshold_y)
            row = {"Metric": display_name}

            for vm in variant_metrics:
                variant_display = alias_map.get(vm["name"], vm["name"])
                value = vm.get(metric_key, 0)
                row[variant_display] = format(value, fmt)

            rows.append(row)

    # Create DataFrame and export to CSV
    df = pd.DataFrame(rows)

    # Write to buffer
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    # Use custom filename if provided, otherwise generate with timestamp
    if request.filename:
        filename = f"{request.filename}.csv"
    else:
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"compare_variants_{timestamp_file}.csv"

    return StreamingResponse(
        io.BytesIO(buffer.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.post("/export-pdf")
async def export_pdf(request: CompareExportRequest) -> StreamingResponse:
    """
    Export comparison table as PDF file (landscape orientation).
    Page 1: Return, Volatility, Drawdown & Risk, Risk-Adjusted metrics
    Page 2: Rolling Stats, Win/Loss, Trade Statistics, Threshold metrics

    Args:
        request: Session ID and alias map

    Returns:
        PDF file as streaming response
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak

    session = get_session(request.session_id)

    variant_metrics = session.get("variant_metrics")
    if variant_metrics is None:
        raise HTTPException(
            status_code=400,
            detail="Metrics not calculated yet. Call /calculate-metrics first."
        )

    alias_map = request.alias_map or {}
    threshold_x = session.get("threshold_x_pct", -5.0)
    threshold_y = session.get("threshold_y_pct", -10.0)
    custom_title = request.title or "Variant Comparison Report"

    # Filter variant metrics by selected variants
    if request.selected_variants:
        variant_metrics = [vm for vm in variant_metrics if vm["name"] in request.selected_variants]

    # Get variant names (use aliases if provided)
    variant_names = []
    for vm in variant_metrics:
        original_name = vm["name"]
        display_name = alias_map.get(original_name, original_name)
        variant_names.append(display_name)

    # Metrics where lower is better
    lower_better_metrics = {
        'max_drawdown', 'negative_annualized_sd', 'annualized_sd',
        'rolling_roi_std', 'ulcer_index',
        'weeks_below_x_pct', 'weeks_min_notional_below_y_pct',
        'avg_orders_per_cycle'
    }

    # Sections for page 1 (before Rolling Statistics)
    page1_sections = ["Return Metrics", "Volatility Metrics", "Drawdown & Risk Metrics", "Risk-Adjusted Metrics"]
    # Sections for page 2 (from Rolling Statistics onwards)
    page2_sections = ["Rolling Statistics", "Win/Loss Metrics", "Trade Statistics", "Threshold Metrics"]

    def build_table_data(sections_to_include):
        """Build table data for specified sections."""
        header_row = ["Metric"] + variant_names
        table_data = [header_row]
        section_rows = []
        best_worst_cells = []
        row_index = 0

        for section in METRIC_SECTIONS:
            section_name = section["name"]
            if section_name not in sections_to_include:
                continue

            section_metrics = section["metrics"]
            valid_metrics = [m for m in section_metrics if m in METRIC_DISPLAY or m in ["weeks_below_x_pct", "weeks_min_notional_below_y_pct"]]

            if not valid_metrics:
                continue

            section_row = [section_name] + [""] * len(variant_names)
            table_data.append(section_row)
            section_rows.append(row_index)
            row_index += 1

            for metric_key in valid_metrics:
                display_name, fmt = get_metric_display(metric_key, threshold_x, threshold_y)
                row = [display_name]

                raw_values = []
                for vm in variant_metrics:
                    value = vm.get(metric_key, 0)
                    raw_values.append(value)
                    row.append(format(value, fmt))

                is_lower_better = metric_key in lower_better_metrics
                if is_lower_better:
                    best_val = min(raw_values)
                    worst_val = max(raw_values)
                else:
                    best_val = max(raw_values)
                    worst_val = min(raw_values)

                best_count = raw_values.count(best_val)
                worst_count = raw_values.count(worst_val)

                for col_idx, val in enumerate(raw_values):
                    actual_row = row_index + 1
                    actual_col = col_idx + 1
                    if val == best_val and best_count == 1:
                        best_worst_cells.append((actual_row, actual_col, True))
                    elif val == worst_val and worst_count == 1:
                        best_worst_cells.append((actual_row, actual_col, False))

                table_data.append(row)
                row_index += 1

        return table_data, section_rows, best_worst_cells

    def create_styled_table(table_data, section_rows, best_worst_cells, col_widths):
        """Create a styled table with all formatting."""
        table = Table(table_data, colWidths=col_widths, repeatRows=1)

        style_commands = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
            ('TOPPADDING', (0, 1), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f1f5f9')),
        ]

        # Section header styling - light gray background with black text
        for section_row_idx in section_rows:
            actual_row = section_row_idx + 1
            style_commands.extend([
                ('BACKGROUND', (0, actual_row), (-1, actual_row), colors.HexColor('#e5e7eb')),
                ('TEXTCOLOR', (0, actual_row), (-1, actual_row), colors.black),
                ('FONTNAME', (0, actual_row), (-1, actual_row), 'Helvetica-Bold'),
                ('FONTSIZE', (0, actual_row), (-1, actual_row), 9),
                ('ALIGN', (0, actual_row), (-1, actual_row), 'LEFT'),
                ('SPAN', (0, actual_row), (-1, actual_row)),
            ])

        # Darker green and red backgrounds for best/worst cells with black text
        best_bg_color = colors.HexColor('#22c55e')    # Darker green background
        worst_bg_color = colors.HexColor('#ef4444')   # Darker red background

        for row_idx, col_idx, is_best in best_worst_cells:
            bg_color = best_bg_color if is_best else worst_bg_color
            style_commands.extend([
                ('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), bg_color),
                ('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx), colors.black),
                ('FONTNAME', (col_idx, row_idx), (col_idx, row_idx), 'Helvetica-Bold'),
            ])

        table.setStyle(TableStyle(style_commands))
        return table

    # Create PDF buffer
    buffer = io.BytesIO()

    # Create PDF document with landscape orientation
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )

    # Calculate column widths
    page_width = landscape(A4)[0] - inch
    header_row = ["Metric"] + variant_names
    num_cols = len(header_row)
    metric_col_width = 2.0 * inch
    remaining_width = page_width - metric_col_width
    variant_col_width = remaining_width / (num_cols - 1) if num_cols > 1 else remaining_width
    col_widths = [metric_col_width] + [variant_col_width] * (num_cols - 1)

    # Build styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20,
        alignment=1
    )

    elements = []

    # Page 1: Title and first set of metrics
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = Paragraph(f"{custom_title} - {timestamp}", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.25*inch))

    # Build and add page 1 table
    table_data1, section_rows1, best_worst1 = build_table_data(page1_sections)
    table1 = create_styled_table(table_data1, section_rows1, best_worst1, col_widths)
    elements.append(table1)

    # Page break
    elements.append(PageBreak())

    # Page 2: Title and remaining metrics
    title2 = Paragraph(f"{custom_title} (continued) - {timestamp}", title_style)
    elements.append(title2)
    elements.append(Spacer(1, 0.25*inch))

    # Build and add page 2 table
    table_data2, section_rows2, best_worst2 = build_table_data(page2_sections)
    table2 = create_styled_table(table_data2, section_rows2, best_worst2, col_widths)
    elements.append(table2)

    # Build PDF
    doc.build(elements)
    buffer.seek(0)

    # Use custom filename if provided, otherwise generate with timestamp
    if request.filename:
        filename = f"{request.filename}.pdf"
    else:
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"compare_variants_{timestamp_file}.pdf"

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
