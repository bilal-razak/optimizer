"""Main optimization pipeline orchestrating all steps."""

from typing import Any, Dict, List, Optional

import pandas as pd

from src.models.optimization import (
    OptimizationRequest,
    OptimizationResponse,
    RankingMetric,
)
from src.optimization.clustering import (
    calculate_auto_k,
    compute_cluster_stats,
    get_best_clusters,
    hdbscan_clustering,
    kmeans_clustering,
)
from src.optimization.feature_engineering import (
    apply_shortlist_conditions,
    feature_engineering,
    prepare_data,
)
from src.optimization.visualization import (
    generate_cluster_scatter,
    generate_heatmaps,
    generate_pca_scatter,
    generate_pca_variance_chart,
)


class OptimizationPipeline:
    """
    Main optimization pipeline that orchestrates all processing steps.

    The pipeline follows this flow:
    1. Load and prepare data
    2. Generate initial heatmaps
    3. Apply shortlisting (optional)
    4. Feature engineering (scaling + PCA)
    5. K-Means clustering
    6. Filter to best K-Means cluster(s)
    7. HDBSCAN clustering on filtered data
    8. Generate final outputs with best clusters highlighted
    """

    def __init__(self, request: OptimizationRequest):
        """
        Initialize pipeline with request parameters.

        Args:
            request: OptimizationRequest containing all configuration
        """
        self.request = request
        self.raw_df: Optional[pd.DataFrame] = None
        self.results_df: Optional[pd.DataFrame] = None
        self.shortlisted_df: Optional[pd.DataFrame] = None

        # Feature engineering outputs
        self.X_df: Optional[pd.DataFrame] = None
        self.params: Optional[pd.DataFrame] = None
        self.metrics: Optional[pd.DataFrame] = None
        self.pca_df: Optional[pd.DataFrame] = None
        self.explained_variance = None
        self.cumulative_variance = None

        # Clustering outputs
        self.kmeans_df: Optional[pd.DataFrame] = None
        self.kmeans_filtered_df: Optional[pd.DataFrame] = None
        self.hdbscan_df: Optional[pd.DataFrame] = None
        self.best_cluster_ids: List[int] = []

        # Visualization outputs
        self.outputs: Dict[str, Any] = {}

    def run(self) -> OptimizationResponse:
        """
        Execute the full optimization pipeline.

        Returns:
            OptimizationResponse containing all results and visualizations
        """
        # Step 1: Load and prepare data
        self._load_data()
        self._prepare_data()

        # Step 2: Initial heatmaps
        self._generate_initial_heatmaps()

        # Step 3: Shortlisting (optional)
        self._apply_shortlisting()

        # Step 4: Feature engineering
        self._feature_engineering()

        # Step 5: K-Means clustering
        self._kmeans_clustering()

        # Step 6: HDBSCAN clustering
        self._hdbscan_clustering()

        # Step 7: Final outputs
        self._generate_final_outputs()

        return self._build_response()

    def _load_data(self) -> None:
        """Load CSV data from file path."""
        self.raw_df = pd.read_csv(self.request.csv_path, on_bad_lines='warn')

    def _prepare_data(self) -> None:
        """Prepare data by calculating derived metrics."""
        self.results_df = prepare_data(self.raw_df, self.request.strategy_params)

        # Sort by strategy params for consistent ordering
        self.results_df = self.results_df.sort_values(
            by=self.request.strategy_params,
            ascending=True
        )

    def _generate_initial_heatmaps(self) -> None:
        """Generate initial parameter landscape heatmaps."""
        hp = self.request.heatmap_params

        self.outputs['initial_heatmaps'] = generate_heatmaps(
            df=self.results_df.reset_index(),
            x_param=hp.x_param,
            y_param=hp.y_param,
            const_param1=hp.const_param1,
            const_param2=hp.const_param2
        )

    def _apply_shortlisting(self) -> None:
        """Apply optional shortlisting conditions."""
        config = self.request.shortlist_config

        if config is None or not config.enabled or not config.conditions:
            self.shortlisted_df = self.results_df.copy()
            self.outputs['shortlisted_heatmaps'] = None
            return

        # Apply conditions
        conditions = [c.model_dump() for c in config.conditions]
        mask = apply_shortlist_conditions(self.results_df, conditions)
        self.shortlisted_df = self.results_df[mask].copy()

        # Mark shortlisted in full results for heatmap highlighting
        results_with_shortlist = self.results_df.reset_index().copy()
        results_with_shortlist['shortlisted'] = mask.values.astype(int)

        hp = self.request.heatmap_params
        self.outputs['shortlisted_heatmaps'] = generate_heatmaps(
            df=results_with_shortlist,
            x_param=hp.x_param,
            y_param=hp.y_param,
            const_param1=hp.const_param1,
            const_param2=hp.const_param2,
            highlight_col='shortlisted',
            highlight_val=1
        )

    def _feature_engineering(self) -> None:
        """Perform feature engineering: scaling and PCA."""
        (
            self.X_df,
            self.params,
            self.metrics,
            self.pca_df,
            self.explained_variance,
            self.cumulative_variance
        ) = feature_engineering(
            self.shortlisted_df,
            self.request.strategy_params
        )

        # Generate PCA visualizations
        self.outputs['pca_variance_chart'] = generate_pca_variance_chart(
            self.explained_variance,
            self.cumulative_variance
        )

        # Prepare hover data
        hover_data = {}
        for col in self.request.strategy_params:
            if col in self.params.columns:
                hover_data[col] = self.params[col]

        ranking_col = self.request.ranking_metric.value
        if ranking_col in self.metrics.columns:
            hover_data[ranking_col] = self.metrics[ranking_col]

        self.outputs['pca_scatter'] = generate_pca_scatter(
            self.pca_df,
            hover_data
        )

    def _kmeans_clustering(self) -> None:
        """Apply K-Means clustering and filter to best cluster(s)."""
        # Calculate k
        if self.request.kmeans_k is not None:
            k = self.request.kmeans_k
        else:
            k = calculate_auto_k(len(self.X_df))

        # Apply K-Means
        kmeans_labels = kmeans_clustering(self.X_df, k)

        # Build clustering DataFrame
        self.kmeans_df = pd.concat([self.params, self.metrics, self.pca_df], axis=1)
        self.kmeans_df.index = self.X_df.index
        self.kmeans_df['kmeans_cluster'] = kmeans_labels

        # Generate scatter plot
        hover_cols = self.request.strategy_params + [self.request.ranking_metric.value]
        hover_cols = [c for c in hover_cols if c in self.kmeans_df.columns]

        self.outputs['kmeans_scatter'] = generate_cluster_scatter(
            self.kmeans_df,
            'kmeans_cluster',
            hover_cols,
            f'K-Means Clustering (k={k})'
        )

        # Compute cluster stats
        ranking_metric = self.request.ranking_metric.value
        kmeans_stats = compute_cluster_stats(
            self.kmeans_df,
            ranking_metric,
            cluster_col='kmeans_cluster',
            threshold_prob=0.0  # No probability threshold for K-Means
        )
        kmeans_stats = kmeans_stats.sort_values('mean', ascending=False)
        self.outputs['kmeans_cluster_stats'] = kmeans_stats.to_dict('records')

        # Get best K-Means cluster(s) - take top 1 for filtering
        best_kmeans_clusters = get_best_clusters(kmeans_stats, num_clusters=1)

        # Filter to best cluster
        mask = self.kmeans_df['kmeans_cluster'].isin(best_kmeans_clusters)
        self.kmeans_filtered_df = self.kmeans_df[mask].copy()

    def _hdbscan_clustering(self) -> None:
        """Apply HDBSCAN clustering on K-Means filtered data."""
        # Get filtered feature data
        filtered_X_df = self.X_df.loc[self.kmeans_filtered_df.index]

        # Apply HDBSCAN
        config = self.request.hdbscan_config
        labels, probabilities, persistence = hdbscan_clustering(
            filtered_X_df,
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples
        )

        # Build HDBSCAN DataFrame
        filtered_params = self.params.loc[self.kmeans_filtered_df.index]
        filtered_metrics = self.metrics.loc[self.kmeans_filtered_df.index]
        filtered_pca = self.pca_df.loc[self.kmeans_filtered_df.index]

        self.hdbscan_df = pd.concat([filtered_params, filtered_metrics, filtered_pca], axis=1)
        self.hdbscan_df['cluster'] = labels
        self.hdbscan_df['cluster_probability'] = probabilities

        # Generate scatter plot
        hover_cols = self.request.strategy_params + [self.request.ranking_metric.value]
        hover_cols = [c for c in hover_cols if c in self.hdbscan_df.columns]

        self.outputs['hdbscan_scatter'] = generate_cluster_scatter(
            self.hdbscan_df,
            'cluster',
            hover_cols,
            f'HDBSCAN Clustering (min_size={config.min_cluster_size}, min_samples={config.min_samples})'
        )

        # Compute cluster stats
        ranking_metric = self.request.ranking_metric.value
        hdbscan_stats = compute_cluster_stats(
            self.hdbscan_df,
            ranking_metric,
            cluster_col='cluster',
            threshold_prob=0.95
        )
        hdbscan_stats = hdbscan_stats.sort_values('mean', ascending=False)
        self.outputs['hdbscan_cluster_stats'] = hdbscan_stats.to_dict('records')

        # Get best HDBSCAN clusters
        self.best_cluster_ids = get_best_clusters(
            hdbscan_stats,
            num_clusters=self.request.num_best_clusters,
            exclude_noise=True
        )

    def _generate_final_outputs(self) -> None:
        """Generate final heatmaps and cluster data for best clusters."""
        hp = self.request.heatmap_params

        # Join cluster assignments back to full results
        full_results = self.results_df.reset_index().copy()
        full_results['cluster'] = -2  # Default: not in any cluster
        full_results['cluster_probability'] = 0.0

        # Map HDBSCAN cluster assignments
        for idx in self.hdbscan_df.index:
            mask = full_results['variant_id'] == idx
            if mask.any():
                full_results.loc[mask, 'cluster'] = self.hdbscan_df.loc[idx, 'cluster']
                full_results.loc[mask, 'cluster_probability'] = self.hdbscan_df.loc[idx, 'cluster_probability']

        # Generate heatmaps for each best cluster
        final_heatmaps = []
        best_clusters_data = []

        for cluster_id in self.best_cluster_ids:
            # Generate heatmap highlighting this cluster
            heatmaps = generate_heatmaps(
                df=full_results,
                x_param=hp.x_param,
                y_param=hp.y_param,
                const_param1=hp.const_param1,
                const_param2=hp.const_param2,
                highlight_col='cluster',
                highlight_val=cluster_id
            )
            final_heatmaps.append(heatmaps)

            # Get cluster data
            cluster_data = self.hdbscan_df[self.hdbscan_df['cluster'] == cluster_id].copy()
            cluster_data = cluster_data.reset_index()
            cluster_data = cluster_data.sort_values(
                by=self.request.strategy_params,
                ascending=True
            )
            best_clusters_data.append(cluster_data.to_dict('records'))

        self.outputs['final_heatmaps'] = final_heatmaps
        self.outputs['best_clusters_data'] = best_clusters_data

    def _build_response(self) -> OptimizationResponse:
        """Build the final response object."""
        return OptimizationResponse(
            initial_heatmaps=self.outputs['initial_heatmaps'],
            shortlisted_heatmaps=self.outputs['shortlisted_heatmaps'],
            pca_variance_chart=self.outputs['pca_variance_chart'],
            pca_scatter=self.outputs['pca_scatter'],
            kmeans_scatter=self.outputs['kmeans_scatter'],
            kmeans_cluster_stats=self.outputs['kmeans_cluster_stats'],
            hdbscan_scatter=self.outputs['hdbscan_scatter'],
            hdbscan_cluster_stats=self.outputs['hdbscan_cluster_stats'],
            final_heatmaps=self.outputs['final_heatmaps'],
            best_clusters_data=self.outputs['best_clusters_data'],
            num_variants_processed=len(self.results_df),
            num_variants_after_shortlist=len(self.shortlisted_df),
            best_cluster_ids=self.best_cluster_ids
        )
