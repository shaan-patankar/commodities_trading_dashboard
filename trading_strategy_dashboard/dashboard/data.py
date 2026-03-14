"""Data loading and preprocessing for the trading dashboard.

Handles CSV ingestion, date normalization, portfolio aggregation,
and data resampling for multi-timeframe analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def read_strategy_csv(path: Path) -> pd.DataFrame:
    """Read a strategy CSV file and normalize to standard format.

    Expected format: one date column + one or more numeric PnL columns.
    Date column auto-detected from common names (date, Date, datetime, etc.).

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with ``"date"`` column (datetime) and numeric product columns.

    Raises:
        ValueError: If no date column is found.
    """
    df = pd.read_csv(path)

    # Auto-detect date column from a set of common names
    date_col = None
    for c in ["date", "Date", "datetime", "Timestamp", "Unnamed: 0"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"Could not find a date column in {path}. Columns={list(df.columns)}")

    df = df.rename(columns={date_col: "date"}).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Convert all product columns to numeric
    product_cols = [c for c in df.columns if c != "date"]
    for c in product_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[product_cols] = df[product_cols].fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# Strategy loading
# ---------------------------------------------------------------------------

def load_strategies(strategy_files: Dict[str, Path], logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """Load all strategy CSV files into a dict of DataFrames.

    Skips files that are missing or have invalid data, logging warnings.

    Args:
        strategy_files: Mapping of strategy name to CSV file path.
        logger:         Logger instance for warning messages.

    Returns:
        Mapping of strategy name to its loaded DataFrame.
    """
    strategies: Dict[str, pd.DataFrame] = {}
    for name, path in strategy_files.items():
        try:
            strategies[name] = read_strategy_csv(path)
        except FileNotFoundError:
            logger.warning("Missing CSV for %s: %s", name, path)
        except ValueError as exc:
            logger.warning("Skipping %s due to invalid data: %s", name, exc)
    return strategies


# ---------------------------------------------------------------------------
# Product / portfolio helpers
# ---------------------------------------------------------------------------

def products_for_strategy(strategies: Dict[str, pd.DataFrame], strategy: str) -> List[str]:
    """Get the list of product column names for a given strategy.

    Args:
        strategies: Full mapping of loaded strategy DataFrames.
        strategy:   Strategy key to look up.

    Returns:
        List of column names excluding ``"date"``, or an empty list if not found.
    """
    df = strategies.get(strategy)
    if df is None:
        return []
    return [c for c in df.columns if c != "date"]


def portfolio_dataframe(strategies: Dict[str, pd.DataFrame], selected_strategies: List[str]) -> pd.DataFrame:
    """Build a portfolio DataFrame by aggregating PnL across strategies.

    Each strategy's individual product PnLs are summed into a single column,
    then all strategies are merged on date (outer join).

    Args:
        strategies:          Full mapping of loaded strategy DataFrames.
        selected_strategies: Strategy names to include in the portfolio.

    Returns:
        Merged DataFrame with ``"date"`` and one column per strategy,
        filled with 0.0 where data is missing.
    """
    frames: List[pd.DataFrame] = []
    for strategy in selected_strategies:
        df = strategies.get(strategy)
        if df is None:
            continue
        products = products_for_strategy(strategies, strategy)
        frame = df[["date"]].copy()
        frame[strategy] = df[products].sum(axis=1)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["date"])

    merged = frames[0]
    for frame in frames[1:]:
        merged = pd.merge(merged, frame, on="date", how="outer")

    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged.fillna(0.0)
    return merged
