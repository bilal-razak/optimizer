"""Feature engineering functions for optimization pipeline."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# Standard metric columns expected in backtest results
METRIC_COLS = [
    'total_pnl', 'sortino_ratio', 'sharpe_ratio', 'avg_annual_roi',
    'win_ratio', 'max_draw_down', 'annualized_sd', 'negative_annualized_sd',
    'profit_factor', 'average_profit', 'average_loss', 'avg_order_per_cycle'
]


def prepare_data(df: pd.DataFrame, strategy_params: List[str]) -> pd.DataFrame:
    """
    Prepare backtest data by calculating derived metrics and extracting required columns.

    Args:
        df: Raw backtest results DataFrame
        strategy_params: List of strategy parameter column names

    Returns:
        DataFrame with variant_id as index, containing metrics and strategy params
    """
    df = df.copy()

    # Calculate derived metrics
    if 'non_negative_pnl_value' in df.columns and 'negative_pnl_value' in df.columns:
        df['profit_factor'] = df['non_negative_pnl_value'] / df['negative_pnl_value'].abs()

    if 'orders_counts' in df.columns and 'total_trades' in df.columns:
        df['avg_order_per_cycle'] = df['orders_counts'] / df['total_trades']

    # Handle infinite values in profit_factor
    df['profit_factor'] = df['profit_factor'].replace([np.inf, -np.inf], np.nan)

    # Select required columns
    available_metrics = [col for col in METRIC_COLS if col in df.columns]
    required_cols = ['variant_id'] + available_metrics + strategy_params

    # Filter to required columns and set index
    result = df[required_cols].copy()
    result = result.set_index('variant_id')

    return result


def feature_engineering(
    results: pd.DataFrame,
    param_cols: List[str],
    filter_mask: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Perform feature engineering: scaling and PCA dimensionality reduction.

    Args:
        results: DataFrame with variant_id as index
        param_cols: List of parameter column names
        filter_mask: Optional boolean mask to filter rows

    Returns:
        Tuple containing:
            - X_df: Combined scaled params + PCA components
            - params: Original parameter values
            - metrics: Original metric values
            - pca_df: PCA components
            - explained_variance: Explained variance ratio per component
            - cumulative_variance: Cumulative explained variance
    """
    # Apply filter if provided
    if filter_mask is not None:
        filtered_results = results[filter_mask].copy()
    else:
        filtered_results = results.copy()

    # Separate params and metrics
    params = filtered_results[param_cols].copy()
    metrics = filtered_results.drop(columns=param_cols).copy()

    # Handle missing values
    metrics = metrics.fillna(metrics.median())

    # Scale parameters with MinMaxScaler
    minmax_scaler = MinMaxScaler()
    params_scaled = pd.DataFrame(
        minmax_scaler.fit_transform(params),
        columns=params.columns,
        index=params.index
    )

    # Scale metrics with RobustScaler (handles outliers better)
    robust_scaler = RobustScaler()
    metrics_scaled = pd.DataFrame(
        robust_scaler.fit_transform(metrics),
        columns=metrics.columns,
        index=metrics.index
    )

    # Full PCA to get explained variance for visualization
    pca_full = PCA(n_components=None)
    pca_full.fit(metrics_scaled)
    explained_variance = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # PCA with 99% variance retention for clustering
    pca = PCA(n_components=0.99)
    principal_components = pca.fit_transform(metrics_scaled)
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i}' for i in range(1, principal_components.shape[1] + 1)],
        index=metrics.index
    )

    # Combine scaled params and PCA components
    X_df = pd.concat([params_scaled, pca_df], axis=1)

    return X_df, params, metrics, pca_df, explained_variance, cumulative_variance


def apply_shortlist_conditions(
    df: pd.DataFrame,
    conditions: List[dict]
) -> pd.Series:
    """
    Apply shortlist conditions to create a boolean mask.

    Args:
        df: DataFrame to filter
        conditions: List of condition dicts with 'metric', 'operator', 'value'

    Returns:
        Boolean Series mask
    """
    if not conditions:
        return pd.Series(True, index=df.index)

    mask = pd.Series(True, index=df.index)

    for condition in conditions:
        metric = condition['metric']
        operator = condition['operator']
        value = condition['value']

        if metric not in df.columns:
            continue

        if operator == '>':
            mask &= df[metric] > value
        elif operator == '>=':
            mask &= df[metric] >= value
        elif operator == '<':
            mask &= df[metric] < value
        elif operator == '<=':
            mask &= df[metric] <= value
        elif operator == '==':
            mask &= df[metric] == value

    return mask
