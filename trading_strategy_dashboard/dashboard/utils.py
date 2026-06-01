"""Utility functions for the trading dashboard.

Provides shared helper functions used across multiple dashboard modules,
such as display-name formatting for product labels.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd


def valid_product_columns(df: pd.DataFrame, products: Optional[List[str]] = None) -> List[str]:
    """Return the valid (non-``"date"``) value columns of *df*.

    Centralises the "which columns are real products" filter duplicated across
    the data, figures, and callbacks modules.

    Args:
        df:       DataFrame with a ``"date"`` column and value columns.
        products: Optional desired subset. When given, only those names that are
                  actual columns are returned (preserving the requested order).
                  When ``None``, all non-date columns are returned.

    Returns:
        Ordered list of column names. Empty if *df* has no value columns or no
        requested product matches.
    """
    if df is None or not hasattr(df, "columns"):
        return []
    non_date = [c for c in df.columns if c != "date"]
    if products is None:
        return non_date
    valid = set(non_date)
    return [p for p in products if p in valid]


def format_product_label(name: str) -> str:
    """Convert a snake_case product name to Title Case for display.

    Args:
        name: The raw column / product name (e.g. ``"crude_oil"``).

    Returns:
        A human-friendly label (e.g. ``"Crude Oil"``).
    """
    return name.replace("_", " ").title()
