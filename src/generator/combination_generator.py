"""Core logic for generating parameter combinations."""

import itertools
import re
from collections import OrderedDict
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
    dependent_params = []  # Track dependent parameters created by expressions

    for dep in dependencies:
        if dep.type == "expression":
            # Expression type: create/compute a dependent parameter
            left, op, right = _parse_expression(dep.expression)
            if op == '=':
                # Create a closure to capture the current state of 'right'
                expr = right

                def make_compute_row(expression):
                    def compute_row(row):
                        context = {col: row[col] for col in result.columns if col != '__name__'}
                        return _safe_eval(expression, context)
                    return compute_row

                # Create or update the column with computed values
                result[left] = result.apply(make_compute_row(expr), axis=1)

                # Track this as a dependent parameter for __name__ generation
                if left not in dependent_params:
                    dependent_params.append(left)

        elif dep.type in ["condition", "filter"]:
            # Condition/Filter type: filter rows based on any parameters (including dependent ones)
            left, op, right = _parse_expression(dep.expression)

            # Create a closure to capture current state
            def make_evaluate_condition(l, o, r):
                def evaluate_condition(row):
                    context = {col: row[col] for col in result.columns if col != '__name__'}
                    left_val = _safe_eval(l, context)
                    right_val = _safe_eval(r, context)

                    if o == '>':
                        return left_val > right_val
                    elif o == '>=':
                        return left_val >= right_val
                    elif o == '<':
                        return left_val < right_val
                    elif o == '<=':
                        return left_val <= right_val
                    elif o == '==':
                        return left_val == right_val
                    elif o == '!=':
                        return left_val != right_val
                    return True
                return evaluate_condition

            mask = result.apply(make_evaluate_condition(left, op, right), axis=1)
            result = result[mask].reset_index(drop=True)

    return result, dependent_params


def generate_name_column(
    df: pd.DataFrame,
    params: List[str],
    prefix: str = "",
    postfix: str = ""
) -> pd.Series:
    """
    Generate __name__ column based on format.

    Groups parameters by indicator prefix (text before first underscore).
    Format: {prefix}<indicator1>[val1, val2, ...] + <indicator2>[val1, val2, ...] + ...{postfix}

    Example: Given params EMA_tf=5, EMA_len=10, ADX_tf=15, ADX_dilen=14
    With prefix="Strategy_", postfix="_v1"
    Output: Strategy_EMA[5, 10] + ADX[15, 14]_v1

    Args:
        df: DataFrame with parameter values
        params: List of parameter names in order (format: <indicator>_<param_name>)
        prefix: String to prepend to the name (empty string if not needed)
        postfix: String to append to the name (empty string if not needed)

    Returns:
        Series with formatted names
    """
    # Group parameters by indicator prefix
    def get_indicator_prefix(param_name: str) -> str:
        """Extract indicator prefix from parameter name (text before first underscore)."""
        if '_' in param_name:
            return param_name.split('_')[0]
        return param_name  # Use full name if no underscore

    # Build ordered dict of indicator -> list of param names
    indicator_params = OrderedDict()
    for param in params:
        if param in df.columns:
            indicator = get_indicator_prefix(param)
            if indicator not in indicator_params:
                indicator_params[indicator] = []
            indicator_params[indicator].append(param)

    def format_row(row):
        parts = []
        for indicator, param_list in indicator_params.items():
            values = [str(row[p]) for p in param_list]
            parts.append(f"{indicator}[{', '.join(values)}]")
        indicator_part = ' + '.join(parts)
        return f"{prefix}{indicator_part}{postfix}"

    return df.apply(format_row, axis=1)


def generate_combinations(
    params: List[ParameterConfig],
    dependencies: List[DependencyConfig] = None,
    name_prefix: str = "",
    name_postfix: str = ""
) -> pd.DataFrame:
    """
    Generate all valid parameter combinations.

    Args:
        params: List of parameter configurations
        dependencies: Optional list of dependencies (expressions create dependent params, conditions filter)
        name_prefix: Prefix for __name__ column (prepended to indicator groups)
        name_postfix: Postfix for __name__ column (appended after indicator groups)

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

    # Apply dependencies (expressions create dependent params, conditions filter rows)
    df, dependent_params = apply_dependencies(df, dependencies)

    # All params for __name__ generation (base + dependent)
    all_param_names = param_names + dependent_params

    # Generate __name__ column using all parameters
    df['__name__'] = generate_name_column(df, all_param_names, name_prefix, name_postfix)

    # Reorder columns to have __name__ first, then base params, then dependent params
    cols = ['__name__'] + param_names + dependent_params
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
