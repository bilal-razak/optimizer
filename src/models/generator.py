"""Pydantic models for the Combination Generator."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ParameterConfig(BaseModel):
    """Configuration for a single parameter."""

    name: str = Field(..., description="Parameter name")
    min_value: float = Field(..., description="Minimum value")
    max_value: float = Field(..., description="Maximum value")
    step: float = Field(..., gt=0, description="Step increment (must be positive)")


class DependencyConfig(BaseModel):
    """Configuration for a dependency between parameters."""

    type: Literal["condition", "expression", "filter"] = Field(
        ...,
        description="Type of dependency: condition (filter rows), expression (compute value), or filter (post-generation filter)"
    )
    expression: str = Field(
        ...,
        description="Expression string, e.g., 'param2 > param1' or 'param2 = param1 + 5'"
    )


class GeneratorRequest(BaseModel):
    """Request model for generating combinations."""

    parameters: List[ParameterConfig] = Field(
        ...,
        min_length=1,
        description="List of parameter configurations"
    )
    dependencies: List[DependencyConfig] = Field(
        default_factory=list,
        description="List of dependencies between parameters"
    )
    name_prefix: str = Field(
        default="",
        description="Prefix for the __name__ column (prepended to indicator groups)"
    )
    name_postfix: str = Field(
        default="",
        description="Postfix for the __name__ column (appended after indicator groups)"
    )


class GeneratorCountRequest(BaseModel):
    """Request model for calculating combination count."""

    parameters: List[ParameterConfig] = Field(
        ...,
        min_length=1,
        description="List of parameter configurations"
    )
    dependencies: List[DependencyConfig] = Field(
        default_factory=list,
        description="List of dependencies between parameters"
    )


class GeneratorCountResponse(BaseModel):
    """Response model for combination count calculation."""

    total_combinations: int = Field(..., description="Total number of combinations")
    num_files: int = Field(..., description="Number of CSV files (if >10K combinations)")
    estimated: bool = Field(
        default=False,
        description="Whether the count is estimated (for large combinations)"
    )


class GeneratorPreviewRequest(BaseModel):
    """Request model for preview generation."""

    parameters: List[ParameterConfig] = Field(
        ...,
        min_length=1,
        description="List of parameter configurations"
    )
    dependencies: List[DependencyConfig] = Field(
        default_factory=list,
        description="List of dependencies between parameters"
    )
    name_prefix: str = Field(
        default="",
        description="Prefix for the __name__ column (prepended to indicator groups)"
    )
    name_postfix: str = Field(
        default="",
        description="Postfix for the __name__ column (appended after indicator groups)"
    )
    preview_limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of rows to preview"
    )


class GeneratorPreviewResponse(BaseModel):
    """Response model for preview generation."""

    total_combinations: int = Field(..., description="Total number of combinations")
    preview_data: List[Dict[str, Any]] = Field(..., description="Preview rows as list of dicts")
    columns: List[str] = Field(..., description="Column names in order")
    num_files: int = Field(..., description="Number of CSV files needed")


class GeneratorGenerateRequest(BaseModel):
    """Request model for full CSV generation."""

    parameters: List[ParameterConfig] = Field(
        ...,
        min_length=1,
        description="List of parameter configurations"
    )
    dependencies: List[DependencyConfig] = Field(
        default_factory=list,
        description="List of dependencies between parameters"
    )
    name_prefix: str = Field(
        default="",
        description="Prefix for the __name__ column (prepended to indicator groups)"
    )
    name_postfix: str = Field(
        default="",
        description="Postfix for the __name__ column (appended after indicator groups)"
    )
    save_path: Optional[str] = Field(
        default=None,
        description="Directory path to save the CSV file(s). If None, only download is available."
    )
    filename_prefix: str = Field(
        default="combinations",
        description="Prefix for the generated CSV filename(s)"
    )


class GeneratorResponse(BaseModel):
    """Response model for full CSV generation."""

    total_combinations: int = Field(..., description="Total number of combinations generated")
    num_files: int = Field(..., description="Number of CSV files generated")
    filenames: List[str] = Field(..., description="List of generated CSV filenames")
    save_path: Optional[str] = Field(default=None, description="Directory where files were saved")
    download_urls: List[str] = Field(
        default_factory=list,
        description="URLs for downloading the generated files"
    )
