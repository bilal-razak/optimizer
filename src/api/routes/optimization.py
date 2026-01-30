"""FastAPI routes for optimization endpoints."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.models.optimization import OptimizationRequest, OptimizationResponse
from src.optimization.pipeline import OptimizationPipeline

router = APIRouter(prefix="/optimization", tags=["optimization"])


@router.get("/browse", response_model=None)
async def browse_directory(
    path: Optional[str] = Query(default=None, description="Directory path to browse"),
    show_hidden: bool = Query(default=False, description="Show hidden files")
) -> Dict[str, Any]:
    """
    Browse local filesystem directories and files.

    Args:
        path: Directory path to browse (defaults to user's home directory)
        show_hidden: Whether to show hidden files (starting with .)

    Returns:
        Dictionary with current path, parent path, directories, and files
    """
    # Default to home directory if no path provided
    if path is None or path == "":
        path = str(Path.home())

    # Resolve the path
    try:
        current_path = Path(path).resolve()
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid path: {path}")

    if not current_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    if not current_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {path}")

    # Get parent path
    parent_path = str(current_path.parent) if current_path != current_path.parent else None

    directories = []
    files = []

    try:
        for item in sorted(current_path.iterdir(), key=lambda x: x.name.lower()):
            # Skip hidden files if not requested
            if not show_hidden and item.name.startswith('.'):
                continue

            if item.is_dir():
                directories.append({
                    "name": item.name,
                    "path": str(item)
                })
            elif item.is_file() and item.suffix.lower() == '.csv':
                # Only show CSV files
                try:
                    size = item.stat().st_size
                    size_str = _format_size(size)
                except Exception:
                    size_str = "Unknown"

                files.append({
                    "name": item.name,
                    "path": str(item),
                    "size": size_str
                })
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {path}")

    return {
        "current_path": str(current_path),
        "parent_path": parent_path,
        "directories": directories,
        "files": files
    }


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


@router.get("/columns", response_model=Dict[str, List[str]])
async def get_csv_columns(csv_path: str) -> Dict[str, List[str]]:
    """
    Get column names from a CSV file for parameter mapping UI.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Dictionary with 'columns' key containing list of column names
    """
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=404,
            detail=f"CSV file not found: {csv_path}"
        )

    try:
        df = pd.read_csv(csv_path, nrows=0)
        return {"columns": df.columns.tolist()}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading CSV file: {str(e)}"
        )


@router.get("/preview", response_model=None)
async def preview_csv(csv_path: str, rows: int = 5) -> Dict[str, Any]:
    """
    Preview first N rows of a CSV file.

    Args:
        csv_path: Path to the CSV file
        rows: Number of rows to preview (default: 5)

    Returns:
        Dictionary with columns and sample data
    """
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=404,
            detail=f"CSV file not found: {csv_path}"
        )

    try:
        df = pd.read_csv(csv_path, nrows=rows)
        return {
            "columns": df.columns.tolist(),
            "data": df.to_dict('records'),
            "total_rows_preview": len(df)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading CSV file: {str(e)}"
        )


@router.post("/run", response_model=OptimizationResponse)
async def run_optimization(request: OptimizationRequest) -> OptimizationResponse:
    """
    Run the full optimization pipeline.

    This endpoint:
    1. Loads and prepares the backtest data
    2. Generates initial parameter landscape heatmaps
    3. Applies optional shortlisting filters
    4. Performs feature engineering (scaling + PCA)
    5. Applies K-Means clustering
    6. Applies HDBSCAN clustering on filtered data
    7. Returns visualizations and best cluster data

    Args:
        request: OptimizationRequest with all configuration parameters

    Returns:
        OptimizationResponse with Plotly JSON visualizations and cluster data
    """
    # Validate CSV path exists
    if not os.path.exists(request.csv_path):
        raise HTTPException(
            status_code=404,
            detail=f"CSV file not found: {request.csv_path}"
        )

    try:
        # Run the pipeline
        pipeline = OptimizationPipeline(request)
        result = pipeline.run()
        return result

    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Column not found in CSV: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid value: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )
