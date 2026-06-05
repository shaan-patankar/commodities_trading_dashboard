"""Comprehensive tests for dashboard.analytics module."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from dashboard.analytics import (
    SeriesPack,
    annualize_factor_from_dates,
    compute_metrics,
    compute_series,
    filter_df_by_range,
    next_range_key,
    padded_date_range,
    range_cycle_label,
)
from dashboard.config import RANGE_OPTIONS


# ---------------------------------------------------------------------------
# compute_series
# ---------------------------------------------------------------------------
class TestComputeSeries:

    def test_equity_curve_is_cumulative(self, sample_df):
        sp = compute_series(sample_df, ["product_a", "product_b"])
        cumulative_pnl = sample_df[["product_a", "product_b"]].sum(axis=1).cumsum()
        expected = cumulative_pnl
        pd.testing.assert_series_equal(sp.equity, expected, check_names=False)

    def test_drawdown_always_le_zero(self, sample_df):
        sp = compute_series(sample_df, ["product_a", "product_b"])
        assert (sp.drawdown <= 0.0 + 1e-15).all(), "Drawdown must be <= 0"

    def test_drawdown_is_absolute(self, sample_df):
        """Drawdown is the absolute gap from the high-water mark (equity - hwm)."""
        sp = compute_series(sample_df, ["product_a"])
        equity = sample_df["product_a"].cumsum()
        hwm = equity.cummax()
        expected_dd = (equity - hwm).fillna(0.0)
        pd.testing.assert_series_equal(sp.drawdown, expected_dd, check_names=False, atol=1e-12)

    def test_returns_are_pnl_over_prev_equity(self):
        """Returns are PnL / previous-day equity (prev equity of 0 -> 0, unclamped)."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "product_a": [0.0, 100.0, -50.0, 0.0, 200.0],
        })
        sp = compute_series(df, ["product_a"])
        equity = pd.Series([0.0, 100.0, 50.0, 50.0, 250.0])
        prev_eq = equity.shift(1).replace(0, np.nan)
        expected = (df["product_a"] / prev_eq).fillna(0.0)
        pd.testing.assert_series_equal(sp.returns, expected, check_names=False, atol=1e-12)


    def test_compute_series_missing_date_column(self):
        """A frame without a 'date' column yields an empty pack, not a KeyError."""
        df = pd.DataFrame({"product_a": [1.0, 2.0, 3.0]})
        sp = compute_series(df, ["product_a"])
        assert len(sp.equity) == 0
        assert len(sp.pnl) == 0

    def test_compute_series_none_frame(self):
        sp = compute_series(None, ["product_a"])
        assert len(sp.equity) == 0

    def test_compute_series_unknown_product_ignored(self, sample_df):
        """Stale/unknown product names are filtered, not raised."""
        sp = compute_series(sample_df, ["product_a", "ghost_product"])
        expected = compute_series(sample_df, ["product_a"])
        pd.testing.assert_series_equal(sp.pnl, expected.pnl, check_names=False)


# ---------------------------------------------------------------------------
# Calmar guard (A1)
# ---------------------------------------------------------------------------
class TestCalmarGuard:
    @staticmethod
    def _calmar_value(rows: list[dict]) -> str:
        return next(r["Value"] for r in rows if r["Metric"] == "Calmar")

    @pytest.mark.xfail(
        strict=True, raises=ZeroDivisionError,
        reason="compute_metrics divides by -max_dd; with no drawdown max_dd == 0 "
               "so it raises ZeroDivisionError. Function left unchanged per request "
               "— this xfail documents the latent edge-case bug.",
    )
    def test_calmar_nan_when_no_drawdown(self):
        """Monotonically rising equity (no drawdown) currently raises in Calmar."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=60, freq="B"),
            "product_a": [1000.0] * 60,  # never a down day -> max_dd == 0
        })
        sp = compute_series(df, ["product_a"])
        rows = compute_metrics(sp, rf_annual=0.0)
        assert self._calmar_value(rows) == "—"

    def test_calmar_finite_with_drawdown(self, sample_series_pack):
        rows = compute_metrics(sample_series_pack, rf_annual=0.0)
        value = self._calmar_value(rows)
        if value != "—":
            assert math.isfinite(float(value))


# ---------------------------------------------------------------------------
# annualize_factor_from_dates
# ---------------------------------------------------------------------------
class TestAnnualizeFactorFromDates:
    """The factor is the mean number of observations in each *complete* calendar
    year (the trailing, partial year is dropped). It is therefore NaN for any
    series that does not span at least one full year before its final year."""

    def test_multi_year_daily_business_day_cadence(self):
        # A holiday-free business-day range averages ~260.5 obs/complete year
        # (52 weeks x 5 days, no holidays removed) — daily cadence.
        dates = pd.Series(pd.bdate_range("2021-01-01", "2023-12-31", freq="B"))
        assert annualize_factor_from_dates(dates) == pytest.approx(260, abs=4)

    def test_multi_year_weekly_approx_52(self):
        dates = pd.Series(pd.date_range("2021-01-01", periods=156, freq="W-FRI"))
        assert annualize_factor_from_dates(dates) == pytest.approx(52, abs=2)

    def test_multi_year_monthly_is_12(self, monthly_dates):
        # 24 monthly points span 2024 + 2025; the complete-year mean is exactly 12.
        assert annualize_factor_from_dates(monthly_dates) == pytest.approx(12)

    def test_single_year_span_is_nan(self, daily_dates):
        # daily_dates spans only 2024 -> the sole year is dropped -> NaN.
        assert math.isnan(annualize_factor_from_dates(daily_dates))

    def test_two_dates_same_year_is_nan(self):
        dates = pd.Series([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")])
        assert math.isnan(annualize_factor_from_dates(dates))

    def test_single_date_is_nan(self):
        dates = pd.Series([pd.Timestamp("2024-01-01")])
        assert math.isnan(annualize_factor_from_dates(dates))


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------
class TestComputeMetrics:
    def test_returns_list_of_dicts(self, sample_series_pack):
        rows = compute_metrics(sample_series_pack)
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)
        metric_names = [r["Metric"] for r in rows]
        assert "Total PnL" in metric_names
        assert "Sharpe" in metric_names
        assert "Sortino" in metric_names
        assert "Calmar" in metric_names
        assert "Max Drawdown" in metric_names

    def test_max_dd_is_cash_and_negative(self, sample_series_pack):
        """Max Drawdown comes from the absolute drawdown series (cash value, <= 0)."""
        rows = compute_metrics(sample_series_pack)
        dd_row = next(r for r in rows if r["Metric"] == "Max Drawdown")
        assert sample_series_pack.drawdown.min() <= 0.0
        val = dd_row["Value"]
        assert "%" not in val  # formatted as cash, not a percentage
        assert float(val.replace(",", "")) <= 0.0

    def test_sortino_present_with_downside(self):
        """With both up and down days the Sortino denominator is defined."""
        dates = pd.Series(pd.bdate_range("2024-01-02", periods=100))
        rng = np.random.default_rng(5)
        df = pd.DataFrame({"date": dates, "product_a": rng.normal(50, 800, len(dates))})
        sp = compute_series(df, ["product_a"])
        rows = compute_metrics(sp)
        sortino_row = next(r for r in rows if r["Metric"] == "Sortino")
        assert sortino_row["Value"] is not None

    def test_std_daily_pnl_present(self, sample_series_pack):
        """Std Daily PnL is a positive numeric cash value for mixed PnL."""
        rows = compute_metrics(sample_series_pack)
        std_row = next(r for r in rows if r["Metric"] == "Std Daily PnL")
        assert std_row["Value"] != "—"
        assert float(std_row["Value"].replace(",", "")) > 0.0

    def test_hit_rate_between_0_and_1(self, sample_series_pack):
        rows = compute_metrics(sample_series_pack)
        hr_row = next(r for r in rows if r["Metric"] == "Hit Rate")
        val_str = hr_row["Value"].replace("%", "").replace(",", "")
        hr = float(val_str) / 100.0
        assert 0.0 <= hr <= 1.0

    def test_profit_factor_positive_when_wins_and_losses(self, sample_series_pack):
        rows = compute_metrics(sample_series_pack)
        pf_row = next(r for r in rows if r["Metric"] == "Profit Factor")
        # With mixed pnl there should be a numeric profit factor
        assert pf_row["Value"] != "—"
        pf = float(pf_row["Value"].replace(",", ""))
        assert pf > 0

    @pytest.mark.xfail(
        strict=True, raises=ZeroDivisionError,
        reason="A single row has no drawdown (max_dd == 0) so compute_metrics "
               "raises ZeroDivisionError in Calmar. Function left unchanged per "
               "request — this xfail documents the latent edge-case bug.",
    )
    def test_single_row_does_not_crash(self, single_row_df):
        sp = compute_series(single_row_df, ["product_a"])
        rows = compute_metrics(sp)
        assert isinstance(rows, list)
        assert len(rows) > 0


# ---------------------------------------------------------------------------
# filter_df_by_range
# ---------------------------------------------------------------------------
class TestFilterDfByRange:
    @pytest.fixture
    def multi_year_df(self):
        dates = pd.date_range("2020-01-01", "2026-03-01", freq="B")
        return pd.DataFrame({
            "date": dates,
            "product_a": range(len(dates)),
        })

    def test_all_returns_full_df(self, multi_year_df):
        result = filter_df_by_range(multi_year_df, "All")
        assert len(result) == len(multi_year_df)

    def test_none_returns_full_df(self, multi_year_df):
        result = filter_df_by_range(multi_year_df, None)
        assert len(result) == len(multi_year_df)

    def test_1m_filters_last_month(self, multi_year_df):
        result = filter_df_by_range(multi_year_df, "1M")
        dates = pd.to_datetime(result["date"])
        span_days = (dates.max() - dates.min()).days
        assert span_days <= 31

    def test_3m_filters_last_3_months(self, multi_year_df):
        result = filter_df_by_range(multi_year_df, "3M")
        dates = pd.to_datetime(result["date"])
        span_days = (dates.max() - dates.min()).days
        assert span_days <= 93

    def test_ytd_starts_from_jan_1(self, multi_year_df):
        result = filter_df_by_range(multi_year_df, "YTD")
        dates = pd.to_datetime(result["date"])
        end_year = dates.max().year
        assert dates.min() >= pd.Timestamp(end_year, 1, 1)

    def test_1y_filters_last_year(self, multi_year_df):
        result = filter_df_by_range(multi_year_df, "1Y")
        dates = pd.to_datetime(result["date"])
        span_days = (dates.max() - dates.min()).days
        assert span_days <= 366

    def test_empty_df_returns_empty(self, empty_df):
        result = filter_df_by_range(empty_df, "1M")
        assert len(result) == 0

    def test_none_df_returns_none(self):
        result = filter_df_by_range(None, "1M")
        assert result is None

    def test_unknown_range_returns_full(self, multi_year_df):
        result = filter_df_by_range(multi_year_df, "UNKNOWN")
        assert len(result) == len(multi_year_df)


# ---------------------------------------------------------------------------
# next_range_key & range_cycle_label
# ---------------------------------------------------------------------------
class TestNextRangeKey:
    def test_cycles_through_options(self):
        key = "1M"
        visited = [key]
        for _ in range(len(RANGE_OPTIONS)):
            key = next_range_key(key)
            visited.append(key)
        # Should wrap around
        assert visited[0] == visited[len(RANGE_OPTIONS)]

    def test_none_treated_as_all(self):
        result = next_range_key(None)
        # "All" is last in RANGE_OPTIONS, so next should be first
        expected_idx = (RANGE_OPTIONS.index("All") + 1) % len(RANGE_OPTIONS)
        assert result == RANGE_OPTIONS[expected_idx]

    def test_unknown_wraps(self):
        result = next_range_key("INVALID")
        # Falls to last index, then wraps
        assert result == RANGE_OPTIONS[0]


class TestRangeCycleLabel:
    def test_all_same(self):
        assert range_cycle_label("1M", "1M", "1M") == "All Panels: 1M"

    def test_mixed(self):
        assert range_cycle_label("1M", "3M") == "All Panels: Mixed"

    def test_none_normalized_to_all(self):
        assert range_cycle_label(None, None) == "All Panels: All"

    def test_none_mixed_with_value(self):
        assert range_cycle_label(None, "1M") == "All Panels: Mixed"


# ---------------------------------------------------------------------------
# padded_date_range
# ---------------------------------------------------------------------------
class TestPaddedDateRange:
    def test_returns_tuple(self, daily_dates):
        result = padded_date_range(daily_dates)
        assert result is not None
        start, end = result
        assert start < daily_dates.min()
        assert end > daily_dates.max()

    def test_empty_returns_none(self):
        result = padded_date_range(pd.Series(dtype="datetime64[ns]"))
        assert result is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
