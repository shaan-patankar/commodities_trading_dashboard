"""Shared fixtures for the trading dashboard test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dashboard.analytics import SeriesPack, compute_series
from dashboard.config import DEFAULT_INITIAL_CAPITAL


@pytest.fixture
def daily_dates() -> pd.Series:
    """250 business days of dates (roughly 1 year)."""
    return pd.Series(pd.bdate_range("2024-01-02", periods=250, freq="B"))


@pytest.fixture
def weekly_dates() -> pd.Series:
    """52 weekly dates."""
    return pd.Series(pd.date_range("2024-01-01", periods=52, freq="W-FRI"))


@pytest.fixture
def monthly_dates() -> pd.Series:
    """24 monthly dates."""
    return pd.Series(pd.date_range("2024-01-31", periods=24, freq="ME"))


@pytest.fixture
def sample_df(daily_dates) -> pd.DataFrame:
    """DataFrame with two product columns and daily PnL values."""
    rng = np.random.default_rng(42)
    n = len(daily_dates)
    return pd.DataFrame({
        "date": daily_dates,
        "product_a": rng.normal(100, 500, n),
        "product_b": rng.normal(50, 300, n),
    })


@pytest.fixture
def sample_df_weekly(weekly_dates) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n = len(weekly_dates)
    return pd.DataFrame({
        "date": weekly_dates,
        "product_a": rng.normal(200, 1000, n),
    })


@pytest.fixture
def sample_df_monthly(monthly_dates) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = len(monthly_dates)
    return pd.DataFrame({
        "date": monthly_dates,
        "product_a": rng.normal(500, 2000, n),
    })


@pytest.fixture
def sample_series_pack(sample_df) -> SeriesPack:
    """SeriesPack computed from sample_df with default capital."""
    return compute_series(sample_df, ["product_a", "product_b"], DEFAULT_INITIAL_CAPITAL)


@pytest.fixture
def sample_strategy_df(daily_dates) -> pd.DataFrame:
    """Strategy PnL frame with two named products (mirrors a real strategy CSV)."""
    rng = np.random.default_rng(123)
    n = len(daily_dates)
    return pd.DataFrame({
        "date": daily_dates,
        "Brent_M3": rng.normal(100, 500, n),
        "Gasoil_M3": rng.normal(80, 400, n),
    })


@pytest.fixture
def sample_returns_df(daily_dates) -> pd.DataFrame:
    """Daily-returns frame aligned to sample_strategy_df (fractional returns)."""
    rng = np.random.default_rng(321)
    n = len(daily_dates)
    return pd.DataFrame({
        "date": daily_dates,
        "Brent_M3": rng.normal(0.0, 0.012, n),
        "Gasoil_M3": rng.normal(0.0, 0.009, n),
    })


@pytest.fixture
def empty_df() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "product_a": pd.Series(dtype="float64")})


@pytest.fixture
def single_row_df() -> pd.DataFrame:
    return pd.DataFrame({"date": [pd.Timestamp("2024-06-01")], "product_a": [1000.0]})
