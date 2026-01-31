"""Step-by-step optimization API endpoints with session state management."""

import os
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from fastapi.responses import StreamingResponse
import io

from src.models.optimization import (
    ColumnInfo,
    HDBSCANConfigResult,
    RankingMetric,
    StepBestClustersRequest,
    StepBestClustersResponse,
    StepHDBSCANFinalRequest,
    StepHDBSCANFinalResponse,
    StepHDBSCANGridRequest,
    StepHDBSCANGridResponse,
    StepHeatmapRequest,
    StepHeatmapResponse,
    StepKMeansRequest,
    StepKMeansResponse,
    StepLoadDataRequest,
    StepLoadDataResponse,
    StepPCARequest,
    StepPCAResponse,
    StepShortlistRequest,
    StepShortlistResponse,
    StepGenerateReportRequest,
    StepGenerateReportResponse,
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
    generate_hdbscan_core_grid,
    generate_hdbscan_grid,
    generate_heatmaps,
    generate_pca_scatter,
    generate_pca_variance_chart,
    HEATMAP_METRICS,
)

router = APIRouter(prefix="/steps", tags=["step-by-step"])

# Session storage (in-memory for simplicity)
# In production, use Redis or a database
sessions: Dict[str, Dict[str, Any]] = {}


def get_session(session_id: str) -> Dict[str, Any]:
    """Get session data or raise 404."""
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}. Please start from Step 1."
        )
    return sessions[session_id]


@router.post("/load-data", response_model=StepLoadDataResponse)
async def step_load_data(request: StepLoadDataRequest) -> StepLoadDataResponse:
    """
    Step 1: Load CSV data and show DataFrame head + info.

    This is the starting point of the optimization workflow.
    Supports loading multiple CSV files - they will be concatenated.
    All CSVs must have the same columns.
    Returns a session_id to use in subsequent steps.
    """
    # Validate all CSV paths exist
    for csv_path in request.csv_paths:
        if not os.path.exists(csv_path):
            raise HTTPException(
                status_code=404,
                detail=f"CSV file not found: {csv_path}"
            )

    try:
        # Load all CSV files
        dfs = []
        reference_columns = None

        for i, csv_path in enumerate(request.csv_paths):
            df = pd.read_csv(csv_path)

            # Validate columns match across all files
            if reference_columns is None:
                reference_columns = set(df.columns.tolist())
            else:
                current_columns = set(df.columns.tolist())
                if current_columns != reference_columns:
                    missing = reference_columns - current_columns
                    extra = current_columns - reference_columns
                    error_msg = f"Column mismatch in file: {os.path.basename(csv_path)}."
                    if missing:
                        error_msg += f" Missing columns: {list(missing)}."
                    if extra:
                        error_msg += f" Extra columns: {list(extra)}."
                    raise HTTPException(
                        status_code=400,
                        detail=error_msg
                    )

            dfs.append(df)

        # Concatenate all DataFrames
        raw_df = pd.concat(dfs, ignore_index=True)

        # Prepare data
        results_df = prepare_data(raw_df, request.strategy_params)
        results_df = results_df.sort_values(
            by=request.strategy_params,
            ascending=True
        )

        # Build column info
        column_info = []
        for col in raw_df.columns:
            non_null = raw_df[col].notna().sum()
            column_info.append(ColumnInfo(
                name=col,
                dtype=str(raw_df[col].dtype),
                non_null_count=int(non_null),
                null_count=int(len(raw_df) - non_null)
            ))

        # Get head data (first 10 rows)
        head_df = results_df.head(10).reset_index()
        # Convert to list of dicts, handling NaN
        head_data = []
        for _, row in head_df.iterrows():
            row_dict = {}
            for col in head_df.columns:
                val = row[col]
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (np.integer, np.floating)):
                    row_dict[col] = float(val) if isinstance(val, np.floating) else int(val)
                else:
                    row_dict[col] = val
            head_data.append(row_dict)

        # Determine available metrics (columns that exist and are numeric)
        metric_cols = [col for col, _, _ in HEATMAP_METRICS if col in results_df.columns]

        # Create session
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "csv_paths": request.csv_paths,
            "strategy_params": request.strategy_params,
            "raw_df": raw_df,
            "results_df": results_df,
            "shortlist_mask": None,
            "shortlisted_df": results_df.copy(),  # Initialize with full df
            "heatmap_params": None,
            "X_df": None,
            "params": None,
            "metrics": None,
            "pca_df": None,
            "explained_variance": None,
            "cumulative_variance": None,
            "kmeans_df": None,
            "kmeans_filtered_df": None,
            "filtered_X_df": None,
            "hdbscan_df": None,
            "hdbscan_grid_dfs": None,
            "hdbscan_grid_configs": None,
            "best_cluster_ids": [],
        }

        return StepLoadDataResponse(
            session_id=session_id,
            num_files=len(request.csv_paths),
            num_rows=len(results_df),
            num_columns=len(raw_df.columns),
            columns=raw_df.columns.tolist(),
            column_info=column_info,
            head_data=head_data,
            strategy_params=request.strategy_params,
            available_metrics=metric_cols
        )

    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Column not found in CSV: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in Step 1: {str(e)}"
        )


@router.post("/heatmap", response_model=StepHeatmapResponse)
async def step_heatmap(request: StepHeatmapRequest) -> StepHeatmapResponse:
    """
    Step 2: Generate heatmaps with configurable options.

    Can be called multiple times with different parameters.
    If shortlist has been applied, can optionally highlight shortlisted variants.
    """
    session = get_session(request.session_id)

    try:
        results_df = session["results_df"]
        shortlist_mask = session.get("shortlist_mask")

        # Prepare DataFrame for heatmap
        df_for_heatmap = results_df.reset_index().copy()

        # Add shortlist column if needed
        highlight_col = None
        highlight_val = None
        if request.show_shortlisted and shortlist_mask is not None:
            df_for_heatmap['_shortlisted'] = shortlist_mask.values.astype(int)
            highlight_col = '_shortlisted'
            highlight_val = 1

        # Filter by const_values if specified
        if request.const_param and request.const_values:
            df_for_heatmap = df_for_heatmap[
                df_for_heatmap[request.const_param].isin(request.const_values)
            ]

        # Determine which metrics to show
        available_metrics = [col for col, _, _ in HEATMAP_METRICS if col in df_for_heatmap.columns]
        if request.metrics:
            # Filter to requested metrics that exist
            metrics_to_show = [m for m in request.metrics if m in available_metrics]
        else:
            metrics_to_show = available_metrics

        # Build metrics list for generate_heatmaps
        metrics_tuples = [
            (col, title, rev) for col, title, rev in HEATMAP_METRICS
            if col in metrics_to_show
        ]

        # Generate heatmaps
        heatmaps = generate_heatmaps(
            df=df_for_heatmap,
            x_param=request.x_param,
            y_param=request.y_param,
            const_param1=request.const_param,
            const_param2=None,
            metrics=metrics_tuples if metrics_tuples else None,
            highlight_col=highlight_col,
            highlight_val=highlight_val
        )

        # Store heatmap params for later use in step 7
        session["heatmap_params"] = {
            "x_param": request.x_param,
            "y_param": request.y_param,
            "const_param1": request.const_param,
            "const_param2": None
        }

        return StepHeatmapResponse(
            session_id=request.session_id,
            heatmaps=heatmaps,
            num_heatmaps=len(heatmaps),
            x_param=request.x_param,
            y_param=request.y_param,
            const_param=request.const_param,
            metrics_shown=metrics_to_show
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating heatmaps: {str(e)}"
        )


@router.post("/shortlist", response_model=StepShortlistResponse)
async def step_shortlist(request: StepShortlistRequest) -> StepShortlistResponse:
    """
    Apply optional shortlisting conditions.

    If shortlisting is not enabled, clears any previous shortlist.
    The shortlist mask is stored so heatmaps can highlight shortlisted variants.
    """
    session = get_session(request.session_id)

    try:
        results_df = session["results_df"]
        config = request.shortlist_config

        if config is None or not config.enabled or not config.conditions:
            # No shortlisting - clear mask
            session["shortlist_mask"] = None
            session["shortlisted_df"] = results_df.copy()
            return StepShortlistResponse(
                session_id=request.session_id,
                shortlisted_heatmaps=None,
                num_variants_after_shortlist=len(results_df),
                shortlist_applied=False
            )

        # Apply conditions
        conditions = [c.model_dump() for c in config.conditions]
        mask = apply_shortlist_conditions(results_df, conditions)

        # Store mask for later use in heatmaps
        session["shortlist_mask"] = mask
        session["shortlisted_df"] = results_df[mask].copy()

        return StepShortlistResponse(
            session_id=request.session_id,
            shortlisted_heatmaps=None,  # User generates heatmaps via /heatmap endpoint
            num_variants_after_shortlist=int(mask.sum()),
            shortlist_applied=True
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error applying shortlist: {str(e)}"
        )


@router.post("/pca", response_model=StepPCAResponse)
async def step_pca(request: StepPCARequest) -> StepPCAResponse:
    """
    Step 3: Perform feature engineering (scaling + PCA).
    """
    session = get_session(request.session_id)

    # shortlisted_df is initialized to results_df, so should always exist
    if session["shortlisted_df"] is None:
        session["shortlisted_df"] = session["results_df"].copy()

    try:
        strategy_params = session["strategy_params"]
        shortlisted_df = session["shortlisted_df"]

        # Perform feature engineering
        (
            X_df,
            params,
            metrics,
            pca_df,
            explained_variance,
            cumulative_variance
        ) = feature_engineering(shortlisted_df, strategy_params)

        # Store in session
        session["X_df"] = X_df
        session["params"] = params
        session["metrics"] = metrics
        session["pca_df"] = pca_df
        session["explained_variance"] = explained_variance
        session["cumulative_variance"] = cumulative_variance

        # Generate visualizations
        pca_variance_chart = generate_pca_variance_chart(
            explained_variance,
            cumulative_variance
        )

        # Prepare hover data - strategy params
        hover_data = {}
        for col in strategy_params:
            if col in params.columns:
                hover_data[col] = params[col]

        # Prepare metrics data for hover - key metrics
        metrics_data = {}
        key_metrics = ['sharpe_ratio', 'sortino_ratio', 'profit_factor']
        for col in key_metrics:
            if col in metrics.columns:
                metrics_data[col] = metrics[col]

        pca_scatter = generate_pca_scatter(pca_df, hover_data, metrics_data)

        return StepPCAResponse(
            session_id=request.session_id,
            pca_variance_chart=pca_variance_chart,
            pca_scatter=pca_scatter,
            explained_variance_ratio=explained_variance.tolist()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in Step 3: {str(e)}"
        )


@router.post("/kmeans", response_model=StepKMeansResponse)
async def step_kmeans(request: StepKMeansRequest) -> StepKMeansResponse:
    """
    Step 4: Apply K-Means clustering.
    """
    session = get_session(request.session_id)

    if session["X_df"] is None:
        raise HTTPException(
            status_code=400,
            detail="Step 3 (PCA) must be completed first"
        )

    try:
        X_df = session["X_df"]
        params = session["params"]
        metrics = session["metrics"]
        pca_df = session["pca_df"]
        strategy_params = session["strategy_params"]

        # Calculate k
        if request.k is not None:
            k = request.k
        else:
            k = calculate_auto_k(len(X_df))

        # Apply K-Means
        kmeans_labels = kmeans_clustering(X_df, k)

        # Build clustering DataFrame
        kmeans_df = pd.concat([params, metrics, pca_df], axis=1)
        kmeans_df.index = X_df.index
        kmeans_df['kmeans_cluster'] = kmeans_labels

        session["kmeans_df"] = kmeans_df

        # Generate scatter plot with both strategy params and metrics for hover
        hover_cols = [c for c in strategy_params if c in kmeans_df.columns]
        metrics_cols = ['sharpe_ratio', 'sortino_ratio', 'profit_factor']
        metrics_cols = [c for c in metrics_cols if c in kmeans_df.columns]

        kmeans_scatter = generate_cluster_scatter(
            kmeans_df,
            'kmeans_cluster',
            hover_cols,
            f'K-Means Clustering (k={k})',
            metrics_cols=metrics_cols
        )

        # Compute cluster stats for all key metrics
        all_stats = {}
        key_metrics = ['sharpe_ratio', 'sortino_ratio', 'profit_factor']
        for metric in key_metrics:
            if metric in kmeans_df.columns:
                stats = compute_cluster_stats(
                    kmeans_df,
                    metric,
                    cluster_col='kmeans_cluster',
                    threshold_prob=0.0
                )
                stats = stats.sort_values('mean', ascending=False)
                all_stats[metric] = stats.to_dict('records')

        # Use sharpe_ratio as the default (primary selection metric)
        ranking_metric = 'sharpe_ratio'
        if ranking_metric in kmeans_df.columns:
            kmeans_cluster_stats = all_stats.get(ranking_metric, [])

            # Get best K-Means cluster and filter (always based on sharpe)
            sharpe_stats = compute_cluster_stats(
                kmeans_df,
                ranking_metric,
                cluster_col='kmeans_cluster',
                threshold_prob=0.0
            )
            sharpe_stats = sharpe_stats.sort_values('mean', ascending=False)
            best_kmeans_clusters = get_best_clusters(sharpe_stats, num_clusters=1)
            mask = kmeans_df['kmeans_cluster'].isin(best_kmeans_clusters)
            session["kmeans_filtered_df"] = kmeans_df[mask].copy()
            session["filtered_X_df"] = X_df.loc[session["kmeans_filtered_df"].index]
        else:
            kmeans_cluster_stats = []
            session["kmeans_filtered_df"] = kmeans_df.copy()
            session["filtered_X_df"] = X_df

        # Store all stats for later retrieval by metric
        session["kmeans_all_stats"] = all_stats

        return StepKMeansResponse(
            session_id=request.session_id,
            kmeans_scatter=kmeans_scatter,
            kmeans_cluster_stats=kmeans_cluster_stats,
            k_used=k,
            num_variants_in_best_kmeans=len(session["kmeans_filtered_df"]),
            all_cluster_stats=all_stats
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in Step 4: {str(e)}"
        )


@router.post("/hdbscan-grid", response_model=StepHDBSCANGridResponse)
async def step_hdbscan_grid(request: StepHDBSCANGridRequest) -> StepHDBSCANGridResponse:
    """
    Step 5: Run HDBSCAN with multiple configurations and display grid.

    This runs HDBSCAN for each combination of min_cluster_size and min_samples,
    generating a grid of scatter plots for comparison.
    """
    session = get_session(request.session_id)

    if session["kmeans_filtered_df"] is None:
        raise HTTPException(
            status_code=400,
            detail="Step 4 (K-Means) must be completed first"
        )

    try:
        filtered_X_df = session["filtered_X_df"]
        filtered_params = session["params"].loc[filtered_X_df.index]
        filtered_metrics = session["metrics"].loc[filtered_X_df.index]
        filtered_pca = session["pca_df"].loc[filtered_X_df.index]
        strategy_params = session["strategy_params"]
        ranking_metric = request.ranking_metric.value

        grid_config = request.grid_config
        min_cluster_sizes = grid_config.min_cluster_sizes
        min_sample_sizes = grid_config.min_sample_sizes
        threshold_prob = grid_config.threshold_cluster_prob

        cluster_dfs = []
        cluster_core_dfs = []
        configs = []
        config_results = []

        for min_cluster_size in min_cluster_sizes:
            for min_samples in min_sample_sizes:
                # Skip invalid configs (per notebook logic)
                if min_cluster_size < min_samples:
                    continue

                # Run HDBSCAN
                labels, probabilities, persistence = hdbscan_clustering(
                    filtered_X_df,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples
                )

                # Build cluster DataFrame
                cluster_df = pd.concat([filtered_params, filtered_metrics, filtered_pca], axis=1)
                cluster_df.index = filtered_X_df.index
                cluster_df['cluster'] = labels
                cluster_df['cluster_probability'] = probabilities

                cluster_dfs.append(cluster_df)
                configs.append((min_cluster_size, min_samples))

                # Core points (excluding noise, high probability)
                cluster_core = cluster_df[
                    (cluster_df['cluster'] != -1) &
                    (cluster_df['cluster_probability'] >= threshold_prob)
                ]
                cluster_core_dfs.append(cluster_core)

                # Compute stats for this config
                num_clusters = len([c for c in cluster_df['cluster'].unique() if c != -1])

                if ranking_metric in cluster_df.columns:
                    stats_df = compute_cluster_stats(
                        cluster_df,
                        ranking_metric,
                        cluster_col='cluster',
                        threshold_prob=threshold_prob
                    )
                    cluster_stats = stats_df.to_dict('records')
                else:
                    cluster_stats = []

                config_results.append(HDBSCANConfigResult(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    num_clusters=num_clusters,
                    cluster_stats=cluster_stats
                ))

        # Store for later use
        session["hdbscan_grid_dfs"] = cluster_dfs
        session["hdbscan_grid_core_dfs"] = cluster_core_dfs
        session["hdbscan_grid_configs"] = configs

        # Generate grid charts
        hdbscan_grid_chart = generate_hdbscan_grid(
            cluster_dfs=cluster_dfs,
            configs=configs,
            strategy_params=strategy_params,
            ranking_metric=ranking_metric,
            title="HDBSCAN Clustering Grid (All Points)"
        )

        hdbscan_core_grid_chart = generate_hdbscan_core_grid(
            cluster_core_dfs=cluster_core_dfs,
            configs=configs,
            strategy_params=strategy_params,
            ranking_metric=ranking_metric,
            title="HDBSCAN Clustering Grid (Core Points Only)"
        )

        return StepHDBSCANGridResponse(
            session_id=request.session_id,
            hdbscan_grid_chart=hdbscan_grid_chart,
            hdbscan_core_grid_chart=hdbscan_core_grid_chart,
            config_results=config_results,
            available_configs=[[c[0], c[1]] for c in configs]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in Step 5: {str(e)}"
        )


@router.post("/hdbscan-final", response_model=StepHDBSCANFinalResponse)
async def step_hdbscan_final(request: StepHDBSCANFinalRequest) -> StepHDBSCANFinalResponse:
    """
    Step 6: Run final HDBSCAN with user-selected configuration.
    """
    session = get_session(request.session_id)

    if session["filtered_X_df"] is None:
        raise HTTPException(
            status_code=400,
            detail="Step 4 (K-Means) must be completed first"
        )

    try:
        filtered_X_df = session["filtered_X_df"]
        filtered_params = session["params"].loc[filtered_X_df.index]
        filtered_metrics = session["metrics"].loc[filtered_X_df.index]
        filtered_pca = session["pca_df"].loc[filtered_X_df.index]
        strategy_params = session["strategy_params"]
        ranking_metric = request.ranking_metric.value

        # Run HDBSCAN with selected config
        labels, probabilities, persistence = hdbscan_clustering(
            filtered_X_df,
            min_cluster_size=request.min_cluster_size,
            min_samples=request.min_samples
        )

        # Build HDBSCAN DataFrame
        hdbscan_df = pd.concat([filtered_params, filtered_metrics, filtered_pca], axis=1)
        hdbscan_df.index = filtered_X_df.index
        hdbscan_df['cluster'] = labels
        hdbscan_df['cluster_probability'] = probabilities

        session["hdbscan_df"] = hdbscan_df

        # Generate scatter plot for ALL points
        hover_cols = [c for c in strategy_params if c in hdbscan_df.columns]
        metrics_cols = ['sharpe_ratio', 'sortino_ratio', 'profit_factor']
        metrics_cols = [c for c in metrics_cols if c in hdbscan_df.columns]

        hdbscan_scatter = generate_cluster_scatter(
            hdbscan_df,
            'cluster',
            hover_cols,
            f'HDBSCAN (min_size={request.min_cluster_size}, min_samples={request.min_samples})',
            metrics_cols=metrics_cols
        )

        # Generate scatter plot for CORE points only (default threshold 0.95)
        threshold_prob = 0.95
        core_df = hdbscan_df[
            (hdbscan_df['cluster'] != -1) &
            (hdbscan_df['cluster_probability'] >= threshold_prob)
        ]

        hdbscan_core_scatter = generate_cluster_scatter(
            core_df,
            'cluster',
            hover_cols,
            f'HDBSCAN Core Points (prob >= {threshold_prob})',
            metrics_cols=metrics_cols
        )

        # Compute stats
        num_clusters = len([c for c in hdbscan_df['cluster'].unique() if c != -1])
        num_core_points = len(core_df)

        if ranking_metric in hdbscan_df.columns:
            hdbscan_stats = compute_cluster_stats(
                hdbscan_df,
                ranking_metric,
                cluster_col='cluster',
                threshold_prob=threshold_prob
            )
            hdbscan_stats = hdbscan_stats.sort_values('mean', ascending=False)
            hdbscan_cluster_stats = hdbscan_stats.to_dict('records')
        else:
            hdbscan_cluster_stats = []

        return StepHDBSCANFinalResponse(
            session_id=request.session_id,
            hdbscan_scatter=hdbscan_scatter,
            hdbscan_core_scatter=hdbscan_core_scatter,
            hdbscan_cluster_stats=hdbscan_cluster_stats,
            num_clusters=num_clusters,
            num_core_points=num_core_points
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in Step 6: {str(e)}"
        )


@router.post("/best-clusters", response_model=StepBestClustersResponse)
async def step_best_clusters(request: StepBestClustersRequest) -> StepBestClustersResponse:
    """
    Step 7: Get best clusters and generate final heatmaps.
    """
    session = get_session(request.session_id)

    if session["hdbscan_df"] is None:
        raise HTTPException(
            status_code=400,
            detail="Step 6 (HDBSCAN Final) must be completed first"
        )

    try:
        results_df = session["results_df"]
        hdbscan_df = session["hdbscan_df"]
        strategy_params = session["strategy_params"]
        hp = session.get("heatmap_params", {
            "x_param": strategy_params[0] if len(strategy_params) > 0 else None,
            "y_param": strategy_params[1] if len(strategy_params) > 1 else None,
            "const_param1": strategy_params[2] if len(strategy_params) > 2 else None,
            "const_param2": strategy_params[3] if len(strategy_params) > 3 else None
        })
        ranking_metric = request.ranking_metric.value

        # Get cluster stats and find best clusters
        if ranking_metric in hdbscan_df.columns:
            hdbscan_stats = compute_cluster_stats(
                hdbscan_df,
                ranking_metric,
                cluster_col='cluster',
                threshold_prob=0.95
            )
            hdbscan_stats = hdbscan_stats.sort_values('mean', ascending=False)
            best_cluster_ids = get_best_clusters(
                hdbscan_stats,
                num_clusters=request.num_best_clusters,
                exclude_noise=True
            )
        else:
            best_cluster_ids = []

        session["best_cluster_ids"] = best_cluster_ids

        # Join cluster assignments back to full results for heatmap generation
        full_results = results_df.reset_index().copy()
        full_results['cluster'] = -2  # Default: not in any cluster
        full_results['cluster_probability'] = 0.0

        # Map HDBSCAN cluster assignments
        for idx in hdbscan_df.index:
            mask = full_results['variant_id'] == idx
            if mask.any():
                full_results.loc[mask, 'cluster'] = hdbscan_df.loc[idx, 'cluster']
                full_results.loc[mask, 'cluster_probability'] = hdbscan_df.loc[idx, 'cluster_probability']

        final_heatmaps = []
        final_core_heatmaps = []
        best_clusters_data = []
        cluster_const_values = []  # Store const values for each cluster
        cluster_core_counts = []  # Store core point counts

        # Core probability threshold
        core_threshold = 0.95

        for cluster_id in best_cluster_ids:
            # Get variants in this cluster
            cluster_variants = hdbscan_df[hdbscan_df['cluster'] == cluster_id].index.tolist()

            # Get core variants (high probability) in this cluster
            core_mask = (
                (hdbscan_df['cluster'] == cluster_id) &
                (hdbscan_df['cluster_probability'] >= core_threshold)
            )
            core_variants = hdbscan_df[core_mask].index.tolist()
            cluster_core_counts.append(len(core_variants))

            # Get unique const param values that exist in this cluster
            const_param = hp.get("const_param1")
            const_vals_in_cluster = []
            if const_param and const_param in full_results.columns:
                cluster_mask = full_results['variant_id'].isin(cluster_variants)
                const_vals_in_cluster = sorted(full_results.loc[cluster_mask, const_param].unique().tolist())

            cluster_const_values.append(const_vals_in_cluster)

            # Filter full_results to only include rows with const values present in this cluster
            if const_param and const_vals_in_cluster:
                df_for_heatmap = full_results[full_results[const_param].isin(const_vals_in_cluster)].copy()
            else:
                df_for_heatmap = full_results.copy()

            # Generate heatmaps with ALL cluster variants highlighted
            heatmaps = generate_heatmaps(
                df=df_for_heatmap,
                x_param=hp["x_param"],
                y_param=hp["y_param"],
                const_param1=const_param,
                const_param2=hp.get("const_param2"),
                highlight_col='cluster',
                highlight_val=cluster_id
            )
            final_heatmaps.append(heatmaps)

            # Generate heatmaps with only CORE cluster variants highlighted
            # Mark core variants in df_for_heatmap
            df_for_core_heatmap = df_for_heatmap.copy()
            df_for_core_heatmap['is_core'] = df_for_core_heatmap['variant_id'].isin(core_variants).astype(int)

            core_heatmaps = generate_heatmaps(
                df=df_for_core_heatmap,
                x_param=hp["x_param"],
                y_param=hp["y_param"],
                const_param1=const_param,
                const_param2=hp.get("const_param2"),
                highlight_col='is_core',
                highlight_val=1
            )
            final_core_heatmaps.append(core_heatmaps)

            # Get cluster data with metrics for the table
            cluster_data = hdbscan_df[hdbscan_df['cluster'] == cluster_id].copy()
            cluster_data = cluster_data.reset_index()
            cluster_data = cluster_data.sort_values(
                by=strategy_params,
                ascending=True
            )
            best_clusters_data.append(cluster_data.to_dict('records'))

        return StepBestClustersResponse(
            session_id=request.session_id,
            final_heatmaps=final_heatmaps,
            final_core_heatmaps=final_core_heatmaps,
            best_clusters_data=best_clusters_data,
            best_cluster_ids=best_cluster_ids,
            cluster_const_values=cluster_const_values,
            cluster_core_counts=cluster_core_counts
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in Step 7: {str(e)}"
        )


@router.get("/column-values/{session_id}/{column}")
async def get_column_unique_values(session_id: str, column: str) -> Dict[str, Any]:
    """
    Get unique values for a column in the session's DataFrame.

    Args:
        session_id: Session ID
        column: Column name

    Returns:
        Dictionary with unique values for the column
    """
    session = get_session(session_id)
    results_df = session["results_df"]

    if column not in results_df.columns:
        raise HTTPException(
            status_code=404,
            detail=f"Column not found: {column}"
        )

    try:
        unique_values = sorted(results_df[column].unique().tolist())
        return {
            "column": column,
            "values": unique_values,
            "count": len(unique_values)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting column values: {str(e)}"
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


@router.post("/generate-report")
async def step_generate_report(request: StepGenerateReportRequest):
    """
    Step 8: Generate PDF report with all optimization results.

    This endpoint re-runs the optimization pipeline with the provided configuration
    and generates a comprehensive PDF report.
    """
    from src.optimization.report_generator import (
        generate_report,
        generate_cluster_isolated_scatter,
    )
    from pathlib import Path

    session = get_session(request.session_id)

    try:
        # Get data from session
        results_df = session.get("results_df")
        if results_df is None:
            raise HTTPException(status_code=400, detail="Session data not found. Please run Step 1 first.")

        num_variants = len(results_df)

        # Get shortlist data
        shortlist_mask = session.get("shortlist_mask")
        shortlisted_df = session.get("shortlisted_df", results_df)
        num_shortlisted = len(shortlisted_df) if shortlisted_df is not None else num_variants

        # Convert shortlist conditions to dict format
        shortlist_conditions = [
            {"metric": c.metric, "operator": c.operator, "value": c.value}
            for c in request.shortlist_conditions
        ]

        # Get sharpe_ratio metric tuple
        sharpe_metric = [m for m in HEATMAP_METRICS if m[0] == 'sharpe_ratio']

        # Generate initial heatmaps (sharpe_ratio only)
        df_for_heatmap = results_df.reset_index().copy()
        initial_heatmaps = generate_heatmaps(
            df=df_for_heatmap,
            x_param=request.x_param,
            y_param=request.y_param,
            const_param1=request.const_param,
            const_param2=None,
            metrics=sharpe_metric
        )

        # Get shortlisted heatmaps if enabled
        shortlisted_heatmaps = None
        if request.shortlist_enabled and shortlist_mask is not None:
            df_with_shortlist = df_for_heatmap.copy()
            df_with_shortlist['_shortlisted'] = shortlist_mask.values.astype(int)
            shortlisted_heatmaps = generate_heatmaps(
                df=df_with_shortlist,
                x_param=request.x_param,
                y_param=request.y_param,
                const_param1=request.const_param,
                const_param2=None,
                metrics=sharpe_metric,
                highlight_col='_shortlisted',
                highlight_val=1
            )

        # Get PCA data from session
        explained_variance = session.get("explained_variance")
        cumulative_variance = session.get("cumulative_variance")
        if explained_variance is None:
            raise HTTPException(status_code=400, detail="PCA data not found. Please run Step 3 first.")

        pca_variance_chart = generate_pca_variance_chart(explained_variance, cumulative_variance)

        # Get K-Means data from session
        kmeans_df = session.get("kmeans_df")
        kmeans_filtered_df = session.get("kmeans_filtered_df")
        kmeans_all_stats = session.get("kmeans_all_stats", {})
        kmeans_stats_sharpe = kmeans_all_stats.get("sharpe_ratio", [])
        kmeans_stats_sortino = kmeans_all_stats.get("sortino_ratio", [])
        kmeans_stats_profit = kmeans_all_stats.get("profit_factor", [])

        if kmeans_df is None:
            raise HTTPException(status_code=400, detail="K-Means data not found. Please run Step 4 first.")

        # Regenerate K-Means scatter
        strategy_params = session.get("strategy_params", request.strategy_params)
        hover_cols = [c for c in strategy_params if c in kmeans_df.columns]
        metrics_cols = ['sharpe_ratio', 'sortino_ratio', 'profit_factor']
        metrics_cols = [c for c in metrics_cols if c in kmeans_df.columns]

        k_used = request.kmeans_k if request.kmeans_k else len(kmeans_df['kmeans_cluster'].unique())
        kmeans_scatter = generate_cluster_scatter(
            kmeans_df,
            'kmeans_cluster',
            hover_cols,
            f'K-Means Clustering (k={k_used})',
            metrics_cols=metrics_cols
        )

        # Get HDBSCAN grid data from session
        hdbscan_grid_dfs = session.get("hdbscan_grid_dfs", [])
        hdbscan_grid_configs = session.get("hdbscan_grid_configs", [])

        if not hdbscan_grid_dfs:
            raise HTTPException(status_code=400, detail="HDBSCAN grid data not found. Please run Step 5 first.")

        # Generate HDBSCAN grid charts - all points and core only
        hdbscan_grid_chart_all = generate_hdbscan_grid(
            cluster_dfs=hdbscan_grid_dfs,
            configs=hdbscan_grid_configs,
            strategy_params=strategy_params,
            ranking_metric='sharpe_ratio',
            title="HDBSCAN Clustering Grid - All Points"
        )

        # Generate core only DataFrames
        hdbscan_grid_core_dfs = []
        for cdf in hdbscan_grid_dfs:
            if 'cluster_probability' in cdf.columns:
                core_df = cdf[
                    (cdf['cluster'] != -1) &
                    (cdf['cluster_probability'] >= request.core_probability_threshold)
                ].copy()
            else:
                core_df = cdf[cdf['cluster'] != -1].copy()
            hdbscan_grid_core_dfs.append(core_df)

        # Generate core only grid
        from src.optimization.visualization import generate_hdbscan_core_grid
        hdbscan_grid_chart_core = generate_hdbscan_core_grid(
            cluster_core_dfs=hdbscan_grid_core_dfs,
            configs=hdbscan_grid_configs,
            strategy_params=strategy_params,
            ranking_metric='sharpe_ratio',
            title="HDBSCAN Clustering Grid - Core Points Only"
        )

        # Build config results for table
        hdbscan_config_results = []
        for idx, cdf in enumerate(hdbscan_grid_dfs):
            cfg = hdbscan_grid_configs[idx]
            num_clusters = len([c for c in cdf['cluster'].unique() if c != -1])
            hdbscan_config_results.append({
                'min_cluster_size': cfg[0],
                'min_samples': cfg[1],
                'num_clusters': num_clusters
            })

        # Get final HDBSCAN data from session
        hdbscan_df = session.get("hdbscan_df")
        if hdbscan_df is None:
            raise HTTPException(status_code=400, detail="HDBSCAN final data not found. Please run Step 6 first.")

        # Regenerate HDBSCAN scatters
        hdbscan_scatter_all = generate_cluster_scatter(
            hdbscan_df,
            'cluster',
            hover_cols,
            f'HDBSCAN (min_size={request.hdbscan_min_cluster_size}, min_samples={request.hdbscan_min_samples})',
            metrics_cols=metrics_cols
        )

        core_df = hdbscan_df[
            (hdbscan_df['cluster'] != -1) &
            (hdbscan_df['cluster_probability'] >= request.core_probability_threshold)
        ]
        hdbscan_scatter_core = generate_cluster_scatter(
            core_df,
            'cluster',
            hover_cols,
            f'HDBSCAN Core Points (prob >= {request.core_probability_threshold})',
            metrics_cols=metrics_cols
        )

        # HDBSCAN stats
        hdbscan_num_clusters = len([c for c in hdbscan_df['cluster'].unique() if c != -1])
        hdbscan_num_core_points = len(core_df)

        # Get stats for all three metrics
        hdbscan_stats_sharpe = []
        hdbscan_stats_sortino = []
        hdbscan_stats_profit = []

        if 'sharpe_ratio' in hdbscan_df.columns:
            hdbscan_stats_df = compute_cluster_stats(
                hdbscan_df,
                'sharpe_ratio',
                cluster_col='cluster',
                threshold_prob=request.core_probability_threshold
            )
            hdbscan_stats_sharpe = hdbscan_stats_df.to_dict('records')

        if 'sortino_ratio' in hdbscan_df.columns:
            sortino_stats_df = compute_cluster_stats(
                hdbscan_df,
                'sortino_ratio',
                cluster_col='cluster',
                threshold_prob=request.core_probability_threshold
            )
            hdbscan_stats_sortino = sortino_stats_df.to_dict('records')

        if 'profit_factor' in hdbscan_df.columns:
            pf_stats_df = compute_cluster_stats(
                hdbscan_df,
                'profit_factor',
                cluster_col='cluster',
                threshold_prob=request.core_probability_threshold
            )
            hdbscan_stats_profit = pf_stats_df.to_dict('records')

        # Get best clusters data from session
        best_cluster_ids = session.get("best_cluster_ids", [])
        if not best_cluster_ids:
            raise HTTPException(status_code=400, detail="Best clusters data not found. Please run Step 7 first.")

        # Get heatmap params
        hp = session.get("heatmap_params", {
            "x_param": request.x_param,
            "y_param": request.y_param,
            "const_param1": request.const_param
        })

        # Generate best cluster heatmaps and scatter plots
        best_cluster_heatmaps = []
        best_cluster_scatter_plots = []
        best_cluster_data = []
        best_cluster_core_counts = []

        full_results = results_df.reset_index().copy()
        full_results['cluster'] = -2
        for idx in hdbscan_df.index:
            mask = full_results['variant_id'] == idx
            if mask.any():
                full_results.loc[mask, 'cluster'] = hdbscan_df.loc[idx, 'cluster']

        for cluster_id in best_cluster_ids:
            cluster_variants = hdbscan_df[hdbscan_df['cluster'] == cluster_id].index.tolist()
            core_mask = (
                (hdbscan_df['cluster'] == cluster_id) &
                (hdbscan_df['cluster_probability'] >= request.core_probability_threshold)
            )
            core_count = int(core_mask.sum())
            best_cluster_core_counts.append(core_count)

            # Get const values in cluster
            const_param = hp.get("const_param1") or request.const_param
            const_vals = []
            if const_param and const_param in full_results.columns:
                cluster_mask = full_results['variant_id'].isin(cluster_variants)
                const_vals = sorted(full_results.loc[cluster_mask, const_param].unique().tolist())

            # Filter for heatmap
            if const_param and const_vals:
                df_for_hm = full_results[full_results[const_param].isin(const_vals)].copy()
            else:
                df_for_hm = full_results.copy()

            # Generate highlighted heatmaps
            cluster_heatmaps = generate_heatmaps(
                df=df_for_hm,
                x_param=hp.get("x_param", request.x_param),
                y_param=hp.get("y_param", request.y_param),
                const_param1=const_param,
                const_param2=None,
                metrics=sharpe_metric,
                highlight_col='cluster',
                highlight_val=cluster_id
            )
            best_cluster_heatmaps.append(cluster_heatmaps)

            # Generate isolated scatter plot
            scatter = generate_cluster_isolated_scatter(
                hdbscan_df,
                cluster_id,
                strategy_params
            )
            best_cluster_scatter_plots.append(scatter)

            # Get cluster data
            cluster_data = hdbscan_df[hdbscan_df['cluster'] == cluster_id].copy()
            cluster_data = cluster_data.reset_index()
            cluster_data = cluster_data.sort_values(by=strategy_params, ascending=True)
            best_cluster_data.append(cluster_data.to_dict('records'))

        # Generate the report
        pdf_bytes, filename = generate_report(
            report_title=request.report_title,
            csv_path=request.csv_path,
            strategy_params=strategy_params,
            x_param=request.x_param,
            y_param=request.y_param,
            const_param=request.const_param,
            shortlist_enabled=request.shortlist_enabled,
            shortlist_conditions=shortlist_conditions,
            kmeans_k=k_used,
            hdbscan_min_cluster_size=request.hdbscan_min_cluster_size,
            hdbscan_min_samples=request.hdbscan_min_samples,
            core_probability_threshold=request.core_probability_threshold,
            num_variants=num_variants,
            num_shortlisted=num_shortlisted,
            initial_heatmaps=initial_heatmaps,
            shortlisted_heatmaps=shortlisted_heatmaps,
            pca_variance_chart=pca_variance_chart,
            explained_variance=list(explained_variance),
            cumulative_variance=list(cumulative_variance),
            kmeans_scatter=kmeans_scatter,
            kmeans_stats_sharpe=kmeans_stats_sharpe,
            kmeans_stats_sortino=kmeans_stats_sortino,
            kmeans_stats_profit=kmeans_stats_profit,
            kmeans_best_cluster_size=len(kmeans_filtered_df) if kmeans_filtered_df is not None else 0,
            hdbscan_grid_chart_all=hdbscan_grid_chart_all,
            hdbscan_grid_chart_core=hdbscan_grid_chart_core,
            hdbscan_config_results=hdbscan_config_results,
            hdbscan_scatter_all=hdbscan_scatter_all,
            hdbscan_scatter_core=hdbscan_scatter_core,
            hdbscan_stats_sharpe=hdbscan_stats_sharpe,
            hdbscan_stats_sortino=hdbscan_stats_sortino,
            hdbscan_stats_profit=hdbscan_stats_profit,
            hdbscan_num_clusters=hdbscan_num_clusters,
            hdbscan_num_core_points=hdbscan_num_core_points,
            best_cluster_ids=best_cluster_ids,
            best_cluster_heatmaps=best_cluster_heatmaps,
            best_cluster_scatter_plots=best_cluster_scatter_plots,
            best_cluster_data=best_cluster_data,
            best_cluster_core_counts=best_cluster_core_counts,
            save_path=request.save_path,
            filename=request.report_filename
        )

        # Return success response with saved path
        save_path = Path(request.save_path)
        if save_path.is_dir():
            actual_path = str(save_path / filename)
        else:
            actual_path = str(save_path)

        return StepGenerateReportResponse(
            session_id=request.session_id,
            report_generated=True,
            report_path=actual_path,
            download_available=False,
            report_filename=filename,
            message=f"Report saved successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating report: {str(e)}"
        )
