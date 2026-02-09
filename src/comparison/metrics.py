"""Metric calculation functions for variant comparison.

IMPORTANT: Data is assumed to be WEEKLY returns (52 weeks/year).
All annualization uses factor of 52 for means and sqrt(52) for standard deviation.

Metric Formulas:
1. Total PnL% = sum(PnL%)
2. Avg PnL% = Total PnL% / Total no. of weeks
3. Avg Annual ROI = Avg PnL% * 52
4. Annualized SD = std(PnL%) * sqrt(52)
5. Negative Ann SD = std(PnL[PnL<0]) * sqrt(52)
6. Sharpe = (Avg Annual ROI - 7) / Annualized SD
7. Sortino = (Avg Annual ROI - 7) / Negative Ann SD
8. Max DD = max(CUMMAX(PnL%) - CUMSUM(PnL%))
9. Ulcer Index = std(CUMMAX(PnL%) - CUMSUM(PnL%))
10. Comfort Ratio = Avg Annual ROI / Ulcer Index
11. Max Loss = min(PnL%), Max Win = max(PnL%)
12. Avg Order per Cycle = sum(Trades) / len(PnL)
13. Win Rate = (len(PnL[PnL>=0]) / len(PnL)) * 100
14. 52W Rolling ROI Mean = PnL.rolling(52).sum().mean()
15. 52W Rolling ROI Std = PnL.rolling(52).sum().std()
16. 52W Rolling ROI Mean - Std = (14) - (15)
17. VaR 5% = np.percentile(PnL, 5)
18. Profit Factor = PnL[PnL>=0].sum() / abs(PnL[PnL<0].sum())
19. No. of Weeks < X% = len(PnL[PnL < X])
20. No. of Weeks with Notional loss < Y% = len(notional_min[notional_min < Y])
21. Last 52W PnL% = PnL[-52:].sum()
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Constants for weekly data annualization
WEEKS_PER_YEAR = 52
ROLLING_WINDOW = 52  # 52 weeks = 1 year for rolling stats
ANNUAL_RISK_FREE_RATE = 7.0  # Annual risk-free rate of 7%


def calculate_sharpe_ratio(
    avg_annual_roi: float,
    annualized_sd: float,
    risk_free_rate: float = ANNUAL_RISK_FREE_RATE
) -> float:
    """
    Calculate Sharpe ratio.

    Sharpe = (Avg Annual ROI - Rf) / Annualized SD

    Args:
        avg_annual_roi: Average annual ROI (Avg PnL% * 52)
        annualized_sd: Annualized standard deviation (std(PnL%) * sqrt(52))
        risk_free_rate: Annual risk-free rate (default 7%)

    Returns:
        Sharpe ratio
    """
    if annualized_sd == 0:
        return float('inf') if avg_annual_roi > risk_free_rate else 0.0

    sharpe = (avg_annual_roi - risk_free_rate) / annualized_sd
    return float(sharpe)


def calculate_sortino_ratio(
    avg_annual_roi: float,
    negative_annualized_sd: float,
    risk_free_rate: float = ANNUAL_RISK_FREE_RATE
) -> float:
    """
    Calculate Sortino ratio.

    Sortino = (Avg Annual ROI - Rf) / Negative Annualized SD

    Args:
        avg_annual_roi: Average annual ROI (Avg PnL% * 52)
        negative_annualized_sd: Annualized std of negative returns (std(PnL[PnL<0]) * sqrt(52))
        risk_free_rate: Annual risk-free rate (default 7%)

    Returns:
        Sortino ratio
    """
    if negative_annualized_sd == 0:
        return float('inf') if avg_annual_roi > risk_free_rate else 0.0

    sortino = (avg_annual_roi - risk_free_rate) / negative_annualized_sd
    return float(sortino)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown using simple cumulative sum.

    Max DD = max(CUMMAX(PnL%) - CUMSUM(PnL%))

    Args:
        returns: Series of weekly returns (as percentages)

    Returns:
        Maximum drawdown as a positive percentage
    """
    if len(returns) == 0:
        return 0.0

    # Calculate cumulative sum (simple, not compound)
    cumsum = returns.cumsum()

    # Calculate running maximum of cumulative sum
    cummax = cumsum.cummax()

    # Drawdown at each point = cummax - cumsum
    drawdown = cummax - cumsum

    # Return maximum drawdown (as positive value)
    return float(drawdown.max())


def calculate_ulcer_index(returns: pd.Series) -> float:
    """
    Calculate Ulcer Index using simple cumulative sum.

    Ulcer Index = std(CUMMAX(PnL%) - CUMSUM(PnL%))

    Args:
        returns: Series of weekly returns

    Returns:
        Ulcer Index value
    """
    if len(returns) == 0:
        return 0.0

    # Calculate cumulative sum (simple, not compound)
    cumsum = returns.cumsum()

    # Calculate running maximum of cumulative sum
    cummax = cumsum.cummax()

    # Drawdown at each point = cummax - cumsum
    drawdown = cummax - cumsum

    # Ulcer Index is std of drawdowns
    return float(drawdown.std())


def calculate_comfort_ratio(avg_annual_roi: float, ulcer_index: float) -> float:
    """
    Calculate Comfort Ratio = Avg Annual ROI / Ulcer Index.

    Higher values indicate better risk-adjusted performance.

    Args:
        avg_annual_roi: Average annual ROI (Avg PnL% * 52)
        ulcer_index: Ulcer Index value

    Returns:
        Comfort Ratio
    """
    if ulcer_index == 0:
        return float('inf') if avg_annual_roi > 0 else 0.0

    return float(avg_annual_roi / ulcer_index)


def calculate_annualized_sd(returns: pd.Series) -> float:
    """
    Calculate annualized standard deviation for weekly returns.

    Annualized SD = Weekly SD * sqrt(52)

    Args:
        returns: Series of weekly returns

    Returns:
        Annualized standard deviation
    """
    if len(returns) < 2:
        return 0.0

    return float(returns.std() * np.sqrt(WEEKS_PER_YEAR))


def calculate_negative_annualized_sd(returns: pd.Series) -> float:
    """
    Calculate annualized standard deviation of negative returns only.

    Measures downside volatility.

    Args:
        returns: Series of weekly returns

    Returns:
        Annualized negative standard deviation
    """
    negative_returns = returns[returns < 0]

    if len(negative_returns) < 2:
        return 0.0

    return float(negative_returns.std() * np.sqrt(WEEKS_PER_YEAR))


def calculate_rolling_stats(returns: pd.Series, window: int = ROLLING_WINDOW) -> Tuple[float, float]:
    """
    Calculate 52-week rolling ROI statistics.

    Args:
        returns: Series of weekly returns
        window: Rolling window size (default 52 weeks = 1 year)

    Returns:
        Tuple of (rolling_mean, rolling_std)
    """
    if len(returns) < window:
        # Not enough data for rolling window
        return float(returns.sum()), float(returns.std() * np.sqrt(len(returns)))

    # Calculate rolling sum (ROI over each 52-week period)
    rolling_roi = returns.rolling(window=window).sum()


    if len(rolling_roi.dropna()) == 0:
        return 0.0, 0.0

    return float(rolling_roi.mean()), float(rolling_roi.std())


def calculate_avg_annual_roi(returns: pd.Series) -> float:
    """
    Calculate average annual ROI.

    Avg Annual ROI = Avg PnL% * 52

    Args:
        returns: Series of weekly returns

    Returns:
        Average annual ROI
    """
    if len(returns) == 0:
        return 0.0

    avg_pnl = returns.mean()
    return float(avg_pnl * WEEKS_PER_YEAR)


def calculate_last_52w_pnl(returns: pd.Series) -> float:
    """
    Calculate sum of last 52 weeks PnL.

    Args:
        returns: Series of weekly returns

    Returns:
        Sum of last 52 weeks (or all if less than 52)
    """
    last_52 = returns.tail(52)
    return float(last_52.sum())


def calculate_profit_factor(returns: pd.Series) -> float:
    """
    Calculate Profit Factor = Sum(non-negative returns) / |Sum(negative returns)|.

    Profit Factor = PnL[PnL>=0].sum() / abs(PnL[PnL<0].sum())

    Args:
        returns: Series of weekly returns

    Returns:
        Profit factor (>1 means profitable overall)
    """
    wins = returns[returns >= 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return float('inf') if wins > 0 else 0.0

    return float(wins / losses)


def calculate_var_5pct(returns: pd.Series) -> float:
    """
    Calculate Value at Risk at 5th percentile.

    VaR represents the maximum expected loss at 95% confidence.

    Args:
        returns: Series of weekly returns

    Returns:
        5th percentile value (typically negative)
    """
    if len(returns) == 0:
        return 0.0

    return float(np.percentile(returns, 5))


def calculate_expected_shortfall_95(returns: pd.Series) -> float:
    """
    Calculate Expected Shortfall at 95% confidence level (ES 95%).

    Expected Shortfall (also known as Conditional VaR or CVaR) is the mean
    of all values at or below the 5th percentile. It measures the expected
    loss given that a loss exceeds the VaR threshold.

    Args:
        returns: Series of weekly returns

    Returns:
        Expected shortfall value (mean of worst 5% returns)
    """
    if len(returns) == 0:
        return 0.0

    var_5 = np.percentile(returns, 5)
    tail_returns = returns[returns <= var_5]

    if len(tail_returns) == 0:
        return float(var_5)

    return float(tail_returns.mean())


def calculate_max_profit(returns: pd.Series) -> float:
    """
    Calculate maximum single-period profit.

    Args:
        returns: Series of weekly returns

    Returns:
        Maximum profit value
    """
    if len(returns) == 0:
        return 0.0

    return float(returns.max())


def calculate_weeks_below_threshold(returns: pd.Series, threshold: float) -> int:
    """
    Count weeks where PnL% is below given threshold.

    Args:
        returns: Series of weekly returns
        threshold: Threshold percentage (e.g., -5.0 for -5%)

    Returns:
        Number of weeks below threshold
    """
    return int((returns < threshold).sum())


def calculate_weeks_min_notional_below(
    min_notional_series: pd.Series,
    threshold: float
) -> int:
    """
    Count weeks where Min Notional PnL is below given threshold.

    Args:
        min_notional_series: Series of min notional PnL values
        threshold: Threshold percentage

    Returns:
        Number of weeks below threshold
    """
    if min_notional_series is None or len(min_notional_series) == 0:
        return 0

    return int((min_notional_series < threshold).sum())


def _parse_expiry_date(date_val: Any) -> Optional[pd.Timestamp]:
    """Parse expiry date from various formats."""
    if pd.isna(date_val):
        return None
    try:
        # Handle string format like '"""2025-05-15T00:00:00.000Z"""'
        if isinstance(date_val, str):
            # Remove extra quotes and whitespace
            cleaned = date_val.strip().strip('"').strip("'")
            return pd.to_datetime(cleaned)
        return pd.to_datetime(date_val)
    except (ValueError, TypeError):
        return None


def calculate_variant_metrics(
    df: pd.DataFrame,
    name: str,
    pnl_column: str = "Pnl%",
    trades_column: str = "Trades",
    min_notional_column: str = "",
    expiry_column: str = "Expiry",
    threshold_x_pct: float = -5.0,
    threshold_y_pct: float = -10.0
) -> Dict[str, Any]:
    """
    Calculate all performance metrics for a single variant.

    IMPORTANT: Data is sorted by Expiry column before calculating metrics
    to ensure correct time-series calculations (cumulative, drawdown, rolling).

    Metrics are organized by nature:
    - Return metrics: Total PnL, Avg PnL, Avg Annual ROI, Last 52W PnL
    - Risk metrics: Annualized SD, Negative Ann SD, Max DD, Ulcer Index, VaR 5%
    - Risk-adjusted: Sharpe, Sortino, Comfort Ratio
    - Rolling stats: 52W Rolling Mean, Std, Mean-Std
    - Trade stats: Win Rate, Profit Factor, Max Win, Max Loss, Avg Orders
    - Count metrics: Num Periods, Weeks below thresholds

    Args:
        df: DataFrame containing the variant's trade data
        name: Variant name
        pnl_column: Column name for PnL percentage
        trades_column: Column name for number of trades
        min_notional_column: Column name for Min Notional PnL (optional)
        expiry_column: Column name for expiry date (used for sorting)
        threshold_x_pct: Threshold for "weeks below x%" calculation
        threshold_y_pct: Threshold for "weeks min notional below y%" calculation

    Returns:
        Dictionary of all calculated metrics
    """
    if pnl_column not in df.columns:
        raise ValueError(f"Column '{pnl_column}' not found in data")

    # Sort DataFrame by Expiry column before calculating metrics
    df_sorted = df.copy()
    if expiry_column in df_sorted.columns:
        df_sorted['_parsed_expiry'] = df_sorted[expiry_column].apply(_parse_expiry_date)
        df_sorted = df_sorted.sort_values('_parsed_expiry').reset_index(drop=True)
        df_sorted = df_sorted.drop(columns=['_parsed_expiry'])

    returns = df_sorted[pnl_column].dropna()
    num_periods = len(returns)

    # Handle infinity values for JSON serialization
    def safe_float(val: float) -> float:
        if np.isinf(val):
            return 999.99 if val > 0 else -999.99
        return val

    # ========== 1. RETURN METRICS ==========
    # 1. Total PnL% = sum(PnL%)
    total_pnl = float(returns.sum())

    # 2. Avg PnL% = Total PnL% / Total no. of weeks
    avg_pnl = float(returns.mean()) if num_periods > 0 else 0.0

    # 3. Avg Annual ROI = Avg PnL% * 52
    avg_annual_roi = calculate_avg_annual_roi(returns)

    # 21. Last 52W PnL% = PnL[-52:].sum()
    last_52w_pnl = calculate_last_52w_pnl(returns)

    # ========== 2. VOLATILITY METRICS ==========
    # 4. Annualized SD = std(PnL%) * sqrt(52)
    ann_sd = calculate_annualized_sd(returns)

    # 5. Negative Ann SD = std(PnL[PnL<0]) * sqrt(52)
    neg_ann_sd = calculate_negative_annualized_sd(returns)

    # ========== 3. DRAWDOWN METRICS ==========
    # 8. Max DD = max(CUMMAX(PnL%) - CUMSUM(PnL%))
    max_dd = calculate_max_drawdown(returns)

    # 9. Ulcer Index = std(CUMMAX(PnL%) - CUMSUM(PnL%))
    ulcer_index = calculate_ulcer_index(returns)

    # 17. VaR 5% = np.percentile(PnL, 5)
    var_5 = calculate_var_5pct(returns)

    # Expected Shortfall 95% = mean of all values <= VaR 5%
    es_95 = calculate_expected_shortfall_95(returns)

    # ========== 4. RISK-ADJUSTED METRICS ==========
    # 6. Sharpe = (Avg Annual ROI - 7) / Annualized SD
    sharpe = calculate_sharpe_ratio(avg_annual_roi, ann_sd)

    # 7. Sortino = (Avg Annual ROI - 7) / Negative Ann SD
    sortino = calculate_sortino_ratio(avg_annual_roi, neg_ann_sd)

    # 10. Comfort Ratio = Avg Annual ROI / Ulcer Index
    comfort = calculate_comfort_ratio(avg_annual_roi, ulcer_index)

    # ========== 5. ROLLING STATISTICS ==========
    # 14. 52W Rolling ROI Mean = PnL.rolling(52).sum().mean()
    # 15. 52W Rolling ROI Std = PnL.rolling(52).sum().std()
    rolling_mean, rolling_std = calculate_rolling_stats(returns)

    # 16. 52W Rolling ROI Mean - Std
    rolling_mean_minus_std = rolling_mean - rolling_std

    # ========== 6. WIN/LOSS METRICS ==========
    # 11. Max Loss = min(PnL%), Max Win = max(PnL%)
    max_loss = float(returns.min()) if num_periods > 0 else 0.0
    max_win = calculate_max_profit(returns)

    # 13. Win Rate = (len(PnL[PnL>=0]) / len(PnL)) * 100
    num_winning = int((returns >= 0).sum())
    num_losing = int((returns < 0).sum())
    win_rate = (num_winning / num_periods * 100) if num_periods > 0 else 0.0

    # 18. Profit Factor = PnL[PnL>=0].sum() / abs(PnL[PnL<0].sum())
    profit_factor = calculate_profit_factor(returns)

    # ========== 7. TRADE STATISTICS ==========
    # 12. Avg Order per Cycle = sum(Trades) / len(PnL)
    avg_orders = 0.0
    if trades_column in df_sorted.columns:
        trades = df_sorted[trades_column].dropna()
        if len(trades) > 0 and num_periods > 0:
            avg_orders = float(trades.sum() / num_periods)

    # ========== 8. COUNT METRICS ==========
    # 19. No. of Weeks < X% = len(PnL[PnL < X])
    weeks_below_x = calculate_weeks_below_threshold(returns, threshold_x_pct)

    # 20. No. of Weeks with Notional loss < Y%
    weeks_min_notional_below_y = 0
    if min_notional_column and min_notional_column in df_sorted.columns:
        min_notional_series = df_sorted[min_notional_column].dropna()
        weeks_min_notional_below_y = calculate_weeks_min_notional_below(
            min_notional_series, threshold_y_pct
        )

    return {
        "name": name,
        # Return metrics
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "avg_annual_roi": avg_annual_roi,
        "last_52w_pnl": last_52w_pnl,
        # Volatility metrics
        "annualized_sd": ann_sd,
        "negative_annualized_sd": neg_ann_sd,
        # Drawdown metrics
        "max_drawdown": max_dd,
        "ulcer_index": ulcer_index,
        "var_5pct": var_5,
        "expected_shortfall_95": es_95,
        # Risk-adjusted metrics
        "sharpe_ratio": safe_float(sharpe),
        "sortino_ratio": safe_float(sortino),
        "comfort_ratio": safe_float(comfort),
        # Rolling stats
        "rolling_roi_mean": rolling_mean,
        "rolling_roi_std": rolling_std,
        "rolling_roi_mean_minus_std": rolling_mean_minus_std,
        # Win/loss metrics
        "max_loss": max_loss,
        "max_profit": max_win,
        "win_rate": win_rate,
        "profit_factor": safe_float(profit_factor),
        # Trade stats
        "avg_orders_per_cycle": avg_orders,
        # Count metrics
        "num_periods": num_periods,
        "num_positive": num_winning,
        "num_negative": num_losing,
        "weeks_below_x_pct": weeks_below_x,
        "weeks_min_notional_below_y_pct": weeks_min_notional_below_y,
    }


def calculate_aggregate_stats(variant_metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate aggregate statistics across all variants.

    Args:
        variant_metrics: List of metric dictionaries for each variant

    Returns:
        Dictionary with mean, median, std, min, max for each metric
    """
    if not variant_metrics:
        return {
            "mean": {},
            "median": {},
            "std": {},
            "min": {},
            "max": {},
        }

    # Metrics to aggregate (organized by nature, exclude name and counts)
    numeric_metrics = [
        # Return metrics
        "total_pnl", "avg_pnl", "avg_annual_roi", "last_52w_pnl",
        # Volatility metrics
        "annualized_sd", "negative_annualized_sd",
        # Drawdown metrics
        "max_drawdown", "ulcer_index", "var_5pct", "expected_shortfall_95",
        # Risk-adjusted metrics
        "sharpe_ratio", "sortino_ratio", "comfort_ratio",
        # Rolling stats
        "rolling_roi_mean", "rolling_roi_std", "rolling_roi_mean_minus_std",
        # Win/loss metrics
        "max_loss", "max_profit", "win_rate", "profit_factor",
        # Trade stats
        "avg_orders_per_cycle",
        # Count metrics
        "weeks_below_x_pct", "weeks_min_notional_below_y_pct"
    ]

    result = {
        "mean": {},
        "median": {},
        "std": {},
        "min": {},
        "max": {},
    }

    for metric in numeric_metrics:
        values = [v[metric] for v in variant_metrics if metric in v]
        if values:
            arr = np.array(values)
            result["mean"][metric] = float(np.mean(arr))
            result["median"][metric] = float(np.median(arr))
            result["std"][metric] = float(np.std(arr))
            result["min"][metric] = float(np.min(arr))
            result["max"][metric] = float(np.max(arr))

    return result
