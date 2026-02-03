"""Core logic for generating parameter combinations."""

import itertools
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.models.generator import DependencyConfig, ParameterConfig


def generate_parameter_ranges(params: List[ParameterConfig]) -> Dict[str, List[float]]:
    """
    Generate value ranges for each parameter.

    Args:
        params: List of parameter configurations

    Returns:
        Dictionary mapping parameter names to lists of values
    """
    ranges = {}
    for param in params:
        # Use numpy arange for float-safe range generation
        # Add small epsilon to max to include endpoint when it falls exactly on step
        values = np.arange(param.min_value, param.max_value + param.step * 0.5, param.step)
        # Filter to ensure we don't exceed max_value
        values = values[values <= param.max_value + 1e-10]
        # Round to avoid floating point precision issues
        values = np.round(values, decimals=10)
        ranges[param.name] = values.tolist()
    return ranges


def calculate_combinations_count(
    params: List[ParameterConfig],
    dependencies: List[DependencyConfig] = None
) -> Tuple[int, bool]:
    """
    Calculate total combinations without generating them.

    For simple cases (no dependencies), returns exact count.
    For dependencies with filters, returns estimated count.

    Args:
        params: List of parameter configurations
        dependencies: Optional list of dependencies

    Returns:
        Tuple of (count, is_estimated)
    """
    ranges = generate_parameter_ranges(params)

    # Calculate base count (Cartesian product)
    base_count = 1
    for values in ranges.values():
        base_count *= len(values)

    # If no dependencies, return exact count
    if not dependencies:
        return base_count, False

    # Check if any dependencies are filters (require post-generation filtering)
    has_filters = any(dep.type in ["condition", "filter"] for dep in dependencies)

    if has_filters:
        # For filters, we need to actually generate to get accurate count
        # Return base count as estimate
        return base_count, True

    return base_count, False


def _parse_expression(expression: str) -> Tuple[str, str, str]:
    """
    Parse an expression into (left_side, operator, right_side).

    Supports:
    - Conditions: "param2 > param1", "param2 >= 10"
    - Expressions: "param2 = param1 + 5"

    Args:
        expression: Expression string

    Returns:
        Tuple of (left_side, operator, right_side)
    """
    # Match assignment expressions first (param = expression)
    assignment_match = re.match(r'^(\w+)\s*=\s*(.+)$', expression.strip())
    if assignment_match:
        return assignment_match.group(1), '=', assignment_match.group(2).strip()

    # Match comparison operators
    comparison_ops = ['>=', '<=', '!=', '==', '>', '<']
    for op in comparison_ops:
        if op in expression:
            parts = expression.split(op, 1)
            if len(parts) == 2:
                return parts[0].strip(), op, parts[1].strip()

    raise ValueError(f"Could not parse expression: {expression}")


def _safe_eval(expr: str, context: Dict[str, float]) -> Any:
    """
    Safely evaluate an expression with restricted operations.

    Args:
        expr: Expression string
        context: Dictionary of variable names to values

    Returns:
        Evaluated result
    """
    # Define allowed operations
    allowed_names = {
        'abs': abs,
        'min': min,
        'max': max,
        'round': round,
        'int': int,
        'float': float,
    }

    # Add context variables
    safe_dict = {**allowed_names, **context}

    # Use eval with restricted globals
    try:
        return eval(expr, {"__builtins__": {}}, safe_dict)
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expr}': {str(e)}")


def apply_dependencies(
    df: pd.DataFrame,
    dependencies: List[DependencyConfig]
) -> pd.DataFrame:
    """
    Apply dependencies to filter/modify combinations.

    Args:
        df: DataFrame with parameter combinations
        dependencies: List of dependency configurations

    Returns:
        Filtered/modified DataFrame
    """
    if not dependencies:
        return df

    result = df.copy()

    for dep in dependencies:
        if dep.type == "expression":
            # Expression type: compute a value
            left, op, right = _parse_expression(dep.expression)
            if op == '=':
                # Apply expression to compute new column values
                def compute_row(row):
                    context = {col: row[col] for col in result.columns if col != '__name__'}
                    return _safe_eval(right, context)

                if left in result.columns:
                    result[left] = result.apply(compute_row, axis=1)

        elif dep.type in ["condition", "filter"]:
            # Condition/Filter type: filter rows
            left, op, right = _parse_expression(dep.expression)

            def evaluate_condition(row):
                context = {col: row[col] for col in result.columns if col != '__name__'}
                left_val = _safe_eval(left, context)
                right_val = _safe_eval(right, context)

                if op == '>':
                    return left_val > right_val
                elif op == '>=':
                    return left_val >= right_val
                elif op == '<':
                    return left_val < right_val
                elif op == '<=':
                    return left_val <= right_val
                elif op == '==':
                    return left_val == right_val
                elif op == '!=':
                    return left_val != right_val
                return True

            mask = result.apply(evaluate_condition, axis=1)
            result = result[mask].reset_index(drop=True)

    return result


def generate_name_column(
    df: pd.DataFrame,
    params: List[str],
    default: str,
    position: str
) -> pd.Series:
    """
    Generate __name__ column based on format.

    Format: {default}{param1_name}[{param1_value}]{param2_name}[{param2_value}]...
    or with default as postfix.

    Args:
        df: DataFrame with parameter values
        params: List of parameter names in order
        default: Default name string
        position: "prefix" or "postfix"

    Returns:
        Series with formatted names
    """
    def format_row(row):
        param_parts = ''.join(f"{p}[{row[p]}]" for p in params if p in df.columns)
        if position == "prefix":
            return f"{default}{param_parts}"
        else:
            return f"{param_parts}{default}"

    return df.apply(format_row, axis=1)


def generate_combinations(
    params: List[ParameterConfig],
    dependencies: List[DependencyConfig] = None,
    name_default: str = "Strategy",
    name_position: str = "prefix"
) -> pd.DataFrame:
    """
    Generate all valid parameter combinations.

    Args:
        params: List of parameter configurations
        dependencies: Optional list of dependencies
        name_default: Default part of __name__
        name_position: "prefix" or "postfix"

    Returns:
        DataFrame with all combinations
    """
    dependencies = dependencies or []

    # Generate ranges
    ranges = generate_parameter_ranges(params)
    param_names = [p.name for p in params]

    # Generate Cartesian product
    all_values = [ranges[name] for name in param_names]
    combinations = list(itertools.product(*all_values))

    # Create DataFrame
    df = pd.DataFrame(combinations, columns=param_names)

    # Apply dependencies
    df = apply_dependencies(df, dependencies)

    # Generate __name__ column
    df['__name__'] = generate_name_column(df, param_names, name_default, name_position)

    # Reorder columns to have __name__ first
    cols = ['__name__'] + param_names
    df = df[cols]

    return df


def split_into_chunks(
    df: pd.DataFrame,
    chunk_size: int = 10000
) -> List[pd.DataFrame]:
    """
    Split DataFrame into chunks for large datasets.

    Args:
        df: DataFrame to split
        chunk_size: Maximum rows per chunk

    Returns:
        List of DataFrames
    """
    if len(df) <= chunk_size:
        return [df]

    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        chunks.append(chunk)

    return chunks
