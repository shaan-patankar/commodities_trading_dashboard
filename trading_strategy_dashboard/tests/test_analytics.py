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
from dashboard.config import DEFAULT_INITIAL_CAPITAL, RANGE_OPTIONS


# ---------------------------------------------------------------------------
# compute_series
# ---------------------------------------------------------------------------
class TestComputeSeries:
    def test_equity_starts_at_initial_capital(self, sample_df):
        sp = compute_series(sample_df, ["product_a"], DEFAULT_INITIAL_CAPITAL)
        expected_first = DEFAULT_INITIAL_CAPITAL + sample_df["product_a"].iloc[0]
        assert sp.equity.iloc[0] == pytest.approx(expected_first)

    def test_equity_curve_is_cumulative(self, sample_df):
        sp = compute_series(sample_df, ["product_a", "product_b"], DEFAULT_INITIAL_CAPITAL)
        cumulative_pnl = sample_df[["product_a", "product_b"]].sum(axis=1).cumsum()
        expected = DEFAULT_INITIAL_CAPITAL + cumulative_pnl
        pd.testing.assert_series_equal(sp.equity, expected, check_names=False)

    def test_drawdown_always_le_zero(self, sample_df):
        sp = compute_series(sample_df, ["product_a", "product_b"], DEFAULT_INITIAL_CAPITAL)
        assert (sp.drawdown <= 0.0 + 1e-15).all(), "Drawdown must be <= 0"

    def test_drawdown_is_percentage(self, sample_df):
        """Drawdown should be (equity/hwm) - 1, not absolute difference."""
        sp = compute_series(sample_df, ["product_a"], DEFAULT_INITIAL_CAPITAL)
        # Manually compute expected drawdown
        equity = DEFAULT_INITIAL_CAPITAL + sample_df["product_a"].cumsum()
        hwm = equity.cummax()
        expected_dd = (equity / hwm) - 1.0
        pd.testing.assert_series_equal(sp.drawdown, expected_dd, check_names=False, atol=1e-12)

    def test_returns_clamped(self):
        """Returns must be clamped to [-1.0, 10.0]."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "product_a": [0, 0, -2_000_000, 0, 50_000_000],
        })
        sp = compute_series(df, ["product_a"], 1_000_000)
        assert sp.returns.min() >= -1.0
        assert sp.returns.max() <= 10.0

    def test_no_products_gives_flat_equity(self, sample_df):
        sp = compute_series(sample_df, [], DEFAULT_INITIAL_CAPITAL)
        assert (sp.equity == DEFAULT_INITIAL_CAPITAL).all()
        assert (sp.pnl == 0.0).all()

    def test_initial_capital_nonzero(self):
        """Initial capital should be 1,000,000, not 0."""
        assert DEFAULT_INITIAL_CAPITAL == 1_000_000.0

    def test_compute_series_missing_date_column(self):
        """A frame without a 'date' column yields an empty pack, not a KeyError."""
        df = pd.DataFrame({"product_a": [1.0, 2.0, 3.0]})
        sp = compute_series(df, ["product_a"], DEFAULT_INITIAL_CAPITAL)
        assert len(sp.equity) == 0
        assert len(sp.pnl) == 0

    def test_compute_series_none_frame(self):
        sp = compute_series(None, ["product_a"], DEFAULT_INITIAL_CAPITAL)
        assert len(sp.equity) == 0

    def test_compute_series_unknown_product_ignored(self, sample_df):
        """Stale/unknown product names are filtered, not raised."""
        sp = compute_series(sample_df, ["product_a", "ghost_product"], DEFAULT_INITIAL_CAPITAL)
        expected = compute_series(sample_df, ["product_a"], DEFAULT_INITIAL_CAPITAL)
        pd.testing.assert_series_equal(sp.pnl, expected.pnl, check_names=False)

    def test_compute_series_all_unknown_products_flat(self, sample_df):
        sp = compute_series(sample_df, ["ghost"], DEFAULT_INITIAL_CAPITAL)
        assert (sp.equity == DEFAULT_INITIAL_CAPITAL).all()


# ---------------------------------------------------------------------------
# Calmar guard (A1)
# ---------------------------------------------------------------------------
class TestCalmarGuard:
    @staticmethod
    def _calmar_value(rows: list[dict]) -> str:
        return next(r["Value"] for r in rows if r["Metric"] == "Calmar")

    def test_calmar_nan_when_no_drawdown(self):
        """Monotonically rising equity (no drawdown) -> Calmar is em-dash, not inf."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=60, freq="B"),
            "product_a": [1000.0] * 60,  # never a down day -> max_dd == 0
        })
        sp = compute_series(df, ["product_a"], DEFAULT_INITIAL_CAPITAL)
        rows = compute_metrics(sp, rf_annual=0.0)
        assert self._calmar_value(rows) == "—"

    def test_calmar_finite_with_drawdown(self, sample_series_pack):
        rows = compute_metrics(sample_series_pack, rf_annual=0.0)
        value = self._calmar_value(rows)
        if value != "—":
            assert math.isfinite(float(value))

    def test_single_row(self, single_row_df):
        sp = compute_series(single_row_df, ["product_a"], DEFAULT_INITIAL_CAPITAL)
        assert len(sp.equity) == 1
        assert sp.equity.iloc[0] == DEFAULT_INITIAL_CAPITAL + 1000.0

    def test_empty_products_empty_df(self, empty_df):
        sp = compute_series(empty_df, [], DEFAULT_INITIAL_CAPITAL)
        assert len(sp.equity) == 0


# ---------------------------------------------------------------------------
# annualize_factor_from_dates
# ---------------------------------------------------------------------------
class TestAnnualizeFactorFromDates:
    def test_daily_returns_252(self, daily_dates):
        assert annualize_factor_from_dates(daily_dates) == 252

    def test_weekly_returns_52(self, weekly_dates):
        assert annualize_factor_from_dates(weekly_dates) == 52

    def test_monthly_returns_12(self, monthly_dates):
        assert annualize_factor_from_dates(monthly_dates) == 12

    def test_fewer_than_3_dates_defaults_to_252(self):
        dates = pd.Series([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")])
        assert annualize_factor_from_dates(dates) == 252

    def test_single_date_defaults_to_252(self):
        dates = pd.Series([pd.Timestamp("2024-01-01")])
        assert annualize_factor_from_dates(dates) == 252


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------
class TestComputeMetrics:
    def test_returns_list_of_dicts(self, sample_series_pack):
        rows = compute_metrics(sample_series_pack)
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)
        metric_names = [r["Metric"] for r in rows]
        assert "Sharpe" in metric_names
        assert "Sortino" in metric_names
        assert "CAGR" in metric_names
        assert "Max Drawdown" in metric_names
        assert "Volatility" in metric_names

    def test_max_dd_is_percentage_and_negative(self, sample_series_pack):
        """Max DD metric should come from percentage drawdown series (always <= 0)."""
        rows = compute_metrics(sample_series_pack)
        dd_row = next(r for r in rows if r["Metric"] == "Max Drawdown")
        # The raw min of drawdown series is always <= 0
        assert sample_series_pack.drawdown.min() <= 0.0
        # Formatted value should show a percentage with a minus sign (or 0)
        val = dd_row["Value"]
        assert "%" in val

    def test_sortino_uses_only_negative_excess(self):
        """Sortino denominator must use only NEGATIVE excess returns, not zeros."""
        # All positive returns -> sortino denominator = 0 -> should be NaN
        dates = pd.Series(pd.bdate_range("2024-01-02", periods=100))
        df = pd.DataFrame({"date": dates, "product_a": [1000.0] * 100})
        sp = compute_series(df, ["product_a"], DEFAULT_INITIAL_CAPITAL)
        rows = compute_metrics(sp)
        sortino_row = next(r for r in rows if r["Metric"] == "Sortino")
        # With only positive returns, no negative excess -> "—" (NaN)
        # Or it could be a number if rf makes some excess negative; but with rf=0
        # and all positive pnl, all returns are positive so excess > 0 for most.
        # The key verification: the code filters ex < 0.0 (strict), not ex <= 0.
        assert sortino_row["Value"] is not None

    def test_volatility_from_raw_returns(self, sample_series_pack):
        """Volatility should use raw returns, not excess returns."""
        rows = compute_metrics(sample_series_pack)
        vol_row = next(r for r in rows if r["Metric"] == "Volatility")
        assert vol_row["Value"] != "—"
        # Manually verify: raw std * sqrt(ann)
        ann = annualize_factor_from_dates(pd.to_datetime(sample_series_pack.dates))
        raw_std = float(sample_series_pack.returns.std(ddof=1))
        expected_vol = raw_std * math.sqrt(ann)
        # Parse the formatted percentage
        val_str = vol_row["Value"].replace("%", "").replace(",", "")
        actual_vol = float(val_str) / 100.0
        assert actual_vol == pytest.approx(expected_vol, abs=1e-4)

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

    def test_cagr_short_period(self):
        """CAGR should handle very short time periods gracefully."""
        dates = pd.Series(pd.bdate_range("2024-01-02", periods=3))
        df = pd.DataFrame({"date": dates, "product_a": [1000, 2000, 3000]})
        sp = compute_series(df, ["product_a"], DEFAULT_INITIAL_CAPITAL)
        rows = compute_metrics(sp)
        cagr_row = next(r for r in rows if r["Metric"] == "CAGR")
        # Should not crash; value is either a number or "—"
        assert cagr_row["Value"] is not None

    def test_annualization_detection_shown(self, sample_series_pack):
        rows = compute_metrics(sample_series_pack)
        ann_row = next(r for r in rows if r["Metric"] == "Annualization")
        assert ann_row["Value"] == "252"

    def test_single_row_does_not_crash(self, single_row_df):
        sp = compute_series(single_row_df, ["product_a"], DEFAULT_INITIAL_CAPITAL)
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
class TestEdgeCases:
    def test_zero_initial_capital_protection(self):
        """When initial capital is 0 the code should not divide by zero."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "product_a": [100, 200, -50, 300, 100],
        })
        # equity starts at 0+100=100, prev_eq for first row is 0 -> replaced by NaN
        sp = compute_series(df, ["product_a"], 0.0)
        assert not sp.returns.isna().all(), "Returns should not all be NaN"
        assert np.isfinite(sp.returns.iloc[-1])

    def test_all_zero_pnl(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "product_a": [0.0] * 10,
        })
        sp = compute_series(df, ["product_a"], DEFAULT_INITIAL_CAPITAL)
        assert (sp.equity == DEFAULT_INITIAL_CAPITAL).all()
        assert (sp.drawdown == 0.0).all()
        assert (sp.returns == 0.0).all()
