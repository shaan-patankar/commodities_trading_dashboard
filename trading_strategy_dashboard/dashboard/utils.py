"""Utility functions for the trading dashboard.

Provides shared helper functions used across multiple dashboard modules,
such as display-name formatting for product labels.
"""

from __future__ import annotations


def format_product_label(name: str) -> str:
    """Convert a snake_case product name to Title Case for display.

    Args:
        name: The raw column / product name (e.g. ``"crude_oil"``).

    Returns:
        A human-friendly label (e.g. ``"Crude Oil"``).
    """
    return name.replace("_", " ").title()
