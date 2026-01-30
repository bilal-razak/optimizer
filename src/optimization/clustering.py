"""Clustering functions for optimization pipeline."""

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Suppress sklearn warnings
warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'",
    category=FutureWarning,
)


def kmeans_clustering(X_df: pd.DataFrame, k: int) -> np.ndarray:
    """
    Apply K-Means clustering.

    Args:
        X_df: Feature DataFrame (scaled params + PCA components)
        k: Number of clusters

    Returns:
        Array of cluster labels
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_df.values)
    return labels


def hdbscan_clustering(
    X_df: pd.DataFrame,
    min_cluster_size: int = 5,
    min_samples: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply HDBSCAN clustering.

    Args:
        X_df: Feature DataFrame
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples for core points

    Returns:
        Tuple of (labels, probabilities, persistence)
    """
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_epsilon=0,
        cluster_selection_method='eom',
        prediction_data=True
    )

    labels = clusterer.fit_predict(X_df.values)
    probabilities = clusterer.probabilities_
    persistence = clusterer.cluster_persistence_

    return labels, probabilities, persistence


def compute_cluster_stats(
    cluster_df: pd.DataFrame,
    metric: str,
    cluster_col: str = 'cluster',
    threshold_prob: float = 0.95
) -> pd.DataFrame:
    """
    Compute statistics for each cluster.

    Args:
        cluster_df: DataFrame with cluster assignments
        metric: Metric column to compute stats for
        cluster_col: Name of cluster column
        threshold_prob: Probability threshold for core cluster members

    Returns:
        DataFrame with cluster statistics
    """
    # Create mask for high-probability cluster members (or all if no probability column)
    if 'cluster_probability' in cluster_df.columns:
        mask = (
            ((cluster_df[cluster_col] != -1) & (cluster_df['cluster_probability'] >= threshold_prob)) |
            (cluster_df[cluster_col] == -1)
        )
    else:
        mask = pd.Series(True, index=cluster_df.index)

    filtered_df = cluster_df[mask]

    # Compute statistics per cluster
    stats = filtered_df.groupby(cluster_col)[metric].agg(['count', 'median', 'mean', 'std'])
    stats.columns = ['count', 'median', 'mean', 'std']
    stats = stats.reset_index()
    stats.columns = ['cluster_id', 'count', 'median', 'mean', 'std']

    return stats


def get_best_clusters(
    cluster_stats: pd.DataFrame,
    num_clusters: int,
    exclude_noise: bool = True
) -> List[int]:
    """
    Get the top N cluster IDs by mean metric value.

    Args:
        cluster_stats: DataFrame with cluster statistics (must have 'mean' column)
        num_clusters: Number of top clusters to return
        exclude_noise: Whether to exclude noise cluster (-1)

    Returns:
        List of best cluster IDs
    """
    df = cluster_stats.copy()

    # Exclude noise cluster if requested
    if exclude_noise:
        df = df[df['cluster_id'] != -1]

    # Sort by mean descending and get top N
    df = df.sort_values('mean', ascending=False)
    best_ids = df['cluster_id'].head(num_clusters).tolist()

    return best_ids


def calculate_auto_k(n_samples: int) -> int:
    """
    Calculate automatic k value for K-Means.

    Formula: k = 1 / 0.05 = 20 (as specified)

    Args:
        n_samples: Number of samples

    Returns:
        Calculated k value (minimum 2)
    """
    k = int(1 / 0.05)  # = 20
    return max(2, min(k, n_samples // 2))  # Ensure k is reasonable
