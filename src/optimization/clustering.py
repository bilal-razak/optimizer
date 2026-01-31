"""Clustering functions for optimization pipeline."""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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


def compute_cluster_validation_metrics(
    X: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Any]:
    """
    Compute clustering validation metrics including silhouette score and noise ratio.

    Args:
        X: Feature matrix used for clustering (n_samples, n_features)
        labels: Cluster labels array (-1 for noise in HDBSCAN)

    Returns:
        Dict with:
        - silhouette_score: float or None (if can't be computed)
        - noise_ratio: float (proportion of points labeled as noise)
        - num_clusters: int (excluding noise cluster)
        - cluster_sizes: list of cluster sizes
    """
    result = {
        'silhouette_score': None,
        'noise_ratio': 0.0,
        'num_clusters': 0,
        'cluster_sizes': []
    }

    n_samples = len(labels)
    if n_samples == 0:
        return result

    # Count noise points (label == -1)
    noise_mask = labels == -1
    noise_count = np.sum(noise_mask)
    result['noise_ratio'] = noise_count / n_samples

    # Get unique clusters (excluding noise)
    unique_labels = set(labels) - {-1}
    result['num_clusters'] = len(unique_labels)

    # Calculate cluster sizes
    result['cluster_sizes'] = [np.sum(labels == label) for label in sorted(unique_labels)]

    # Calculate silhouette score (requires at least 2 clusters and some non-noise points)
    non_noise_mask = ~noise_mask
    non_noise_count = np.sum(non_noise_mask)

    if result['num_clusters'] >= 2 and non_noise_count > result['num_clusters']:
        try:
            # Only compute silhouette for non-noise points
            X_non_noise = X[non_noise_mask]
            labels_non_noise = labels[non_noise_mask]

            result['silhouette_score'] = silhouette_score(X_non_noise, labels_non_noise)
        except Exception:
            # Silhouette calculation can fail in edge cases
            result['silhouette_score'] = None

    return result
