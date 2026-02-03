"""API routes for the Combination Generator."""

import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.generator import (
    calculate_combinations_count,
    generate_combinations,
    split_into_chunks,
)
from src.models.generator import (
    GeneratorCountRequest,
    GeneratorCountResponse,
    GeneratorGenerateRequest,
    GeneratorPreviewRequest,
    GeneratorPreviewResponse,
    GeneratorResponse,
)

router = APIRouter(prefix="/generator", tags=["generator"])

# Store generated files temporarily for download
# Key: filename, Value: (file_path, creation_time)
_generated_files: Dict[str, tuple] = {}

# Temporary directory for generated files
TEMP_DIR = Path(tempfile.gettempdir()) / "optimizer_generator"
TEMP_DIR.mkdir(exist_ok=True)


def _cleanup_old_files():
    """Remove files older than 1 hour."""
    now = datetime.now()
    to_remove = []
    for filename, (file_path, created) in _generated_files.items():
        if (now - created).total_seconds() > 3600:  # 1 hour
            to_remove.append(filename)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass

    for filename in to_remove:
        del _generated_files[filename]


@router.post("/calculate-count", response_model=GeneratorCountResponse)
async def calculate_count(request: GeneratorCountRequest):
    """
    Calculate the total number of combinations without generating them.

    Returns the count and whether it's estimated (for complex dependencies).
    """
    try:
        total, is_estimated = calculate_combinations_count(
            request.parameters,
            request.dependencies
        )

        # If dependencies might filter, generate to get accurate count
        if is_estimated and total <= 1000000:  # Only for reasonable sizes
            df = generate_combinations(
                request.parameters,
                request.dependencies
            )
            total = len(df)
            is_estimated = False

        num_files = (total + 9999) // 10000  # Ceiling division

        return GeneratorCountResponse(
            total_combinations=total,
            num_files=num_files,
            estimated=is_estimated
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating count: {str(e)}")


@router.post("/preview", response_model=GeneratorPreviewResponse)
async def generate_preview(request: GeneratorPreviewRequest):
    """
    Generate a preview of the combinations (first N rows).
    """
    try:
        # Generate all combinations
        df = generate_combinations(
            request.parameters,
            request.dependencies,
            request.name_default,
            request.name_position
        )

        total = len(df)
        num_files = (total + 9999) // 10000

        # Get preview rows
        preview_df = df.head(request.preview_limit)

        return GeneratorPreviewResponse(
            total_combinations=total,
            preview_data=preview_df.to_dict(orient='records'),
            columns=list(df.columns),
            num_files=num_files
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating preview: {str(e)}")


@router.post("/generate", response_model=GeneratorResponse)
async def generate_csv(request: GeneratorGenerateRequest):
    """
    Generate the full CSV file(s) with all combinations.

    If save_path is provided, files are saved there.
    Files are also made available for download.
    """
    _cleanup_old_files()

    try:
        # Generate all combinations
        df = generate_combinations(
            request.parameters,
            request.dependencies,
            request.name_default,
            request.name_position
        )

        total = len(df)

        # Split into chunks if needed
        chunks = split_into_chunks(df, chunk_size=10000)
        num_files = len(chunks)

        # Generate unique session ID for this generation
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filenames = []
        download_urls = []
        saved_files = []

        for i, chunk in enumerate(chunks):
            if num_files == 1:
                filename = f"{request.filename_prefix}_{timestamp}.csv"
            else:
                filename = f"{request.filename_prefix}_{timestamp}_part{i + 1}.csv"

            # Save to temp directory for download
            temp_path = TEMP_DIR / f"{session_id}_{filename}"
            chunk.to_csv(temp_path, index=False)

            # Track for download
            _generated_files[f"{session_id}_{filename}"] = (str(temp_path), datetime.now())
            download_urls.append(f"/generator/download/{session_id}_{filename}")

            # Save to user-specified path if provided
            if request.save_path:
                save_dir = Path(request.save_path)
                if save_dir.exists() and save_dir.is_dir():
                    save_path = save_dir / filename
                    chunk.to_csv(save_path, index=False)
                    saved_files.append(str(save_path))

            filenames.append(filename)

        return GeneratorResponse(
            total_combinations=total,
            num_files=num_files,
            filenames=filenames,
            save_path=request.save_path if saved_files else None,
            download_urls=download_urls
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating CSV: {str(e)}")


@router.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download a generated CSV file.
    """
    if filename not in _generated_files:
        raise HTTPException(status_code=404, detail="File not found or expired")

    file_path, _ = _generated_files[filename]

    if not os.path.exists(file_path):
        del _generated_files[filename]
        raise HTTPException(status_code=404, detail="File not found")

    # Extract original filename (without session prefix)
    original_filename = '_'.join(filename.split('_')[1:])

    return FileResponse(
        path=file_path,
        filename=original_filename,
        media_type='text/csv'
    )
