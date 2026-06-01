"""Tests for dashboard.var_scaling (VaR-based position sizing)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from dashboard.analytics import compute_series
from dashboard.config import DEFAULT_INITIAL_CAPITAL, VAR_WINDOW, VAR_Z
from dashboard.data import portfolio_dataframe
from dashboard.var_scaling import (
    compute_var_scaled_frame,
    equal_weight_allocations,
    normalize_allocations,
    portfolio_effective_dataframe,
    validate_var_config,
    var_scaled_aggregate_series,
)


# ---------------------------------------------------------------------------
# Allocation helpers
# ---------------------------------------------------------------------------
class TestAllocationHelpers:
    def test_normalize_sums_to_100(self):
        out = normalize_allocations({"a": 30, "b": 10}, ["a", "b"])
        assert sum(out.values()) == pytest.approx(100.0)
        assert out["a"] == pytest.approx(75.0)

    def test_normalize_zero_falls_back_equal(self):
        out = normalize_allocations({"a": 0, "b": 0}, ["a", "b"])
        assert sum(out.values()) == pytest.approx(100.0)
        assert out["a"] == pytest.approx(50.0)

    def test_normalize_ignores_unknown_and_missing(self):
        out = normalize_allocations({"a": 50, "ghost": 50}, ["a", "b"])
        # only "a" has weight -> normalized to 100, b stays 0
        assert out["a"] == pytest.approx(100.0)
        assert out["b"] == pytest.approx(0.0)
        assert "ghost" not in out

    def test_equal_weight_distribution(self):
        out = equal_weight_allocations(["a", "b", "c"])
        assert sum(out.values()) == pytest.approx(100.0)
        assert out["a"] == pytest.approx(100.0 / 3, abs=1e-3)

    def test_equal_weight_empty(self):
        assert equal_weight_allocations([]) == {}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidateVarConfig:
    PRODUCTS = ["a", "b"]

    def test_valid(self):
        v = validate_var_config(10000, {"a": 60, "b": 40}, self.PRODUCTS)
        assert v["ok"] is True
        assert v["total_error"] is None
        assert v["row_errors"] == {}

    def test_total_not_a_number(self):
        v = validate_var_config("xyz", {"a": 60, "b": 40}, self.PRODUCTS)
        assert v["ok"] is False
        assert v["total_error"] is not None

    def test_total_zero_or_negative(self):
        assert validate_var_config(0, {"a": 60, "b": 40}, self.PRODUCTS)["ok"] is False
        assert validate_var_config(-5, {"a": 60, "b": 40}, self.PRODUCTS)["ok"] is False

    def test_row_not_a_number(self):
        v = validate_var_config(10000, {"a": "abc", "b": 40}, self.PRODUCTS)
        assert v["ok"] is False
        assert "a" in v["row_errors"]

    def test_row_out_of_range(self):
        assert validate_var_config(10000, {"a": -1, "b": 101}, self.PRODUCTS)["ok"] is False

    def test_sum_not_100(self):
        v = validate_var_config(10000, {"a": 30, "b": 30}, self.PRODUCTS)
        assert v["ok"] is False
        assert v["alloc_sum"] == pytest.approx(60.0)

    def test_missing_row_required(self):
        v = validate_var_config(10000, {"a": 100}, self.PRODUCTS)
        assert v["ok"] is False
        assert v["row_errors"].get("b") == "Required"


# ---------------------------------------------------------------------------
# compute_var_scaled_frame
# ---------------------------------------------------------------------------
class TestComputeVarScaledFrame:
    def test_warmup_and_lookahead_rows_zero(self, sample_strategy_df, sample_returns_df):
        scaled, _ = compute_var_scaled_frame(
            sample_strategy_df, sample_returns_df,
            ["Brent_M3", "Gasoil_M3"], 10000, {"Brent_M3": 50, "Gasoil_M3": 50},
        )
        # First `window` rows have no usable shifted notional -> exactly 0.
        for col in ("Brent_M3", "Gasoil_M3"):
            assert (scaled[col].iloc[:VAR_WINDOW] == 0.0).all()

    def test_sizing_math_matches_formula(self, daily_dates):
        # Constant-magnitude alternating returns so rolling std is well-defined.
        n = len(daily_dates)
        r = np.where(np.arange(n) % 2 == 0, 0.01, -0.01)
        strat = pd.DataFrame({"date": daily_dates, "p": np.ones(n)})
        rets = pd.DataFrame({"date": daily_dates, "p": r})
        scaled, diag = compute_var_scaled_frame(strat, rets, ["p"], 10000.0, {"p": 100})

        ret_series = pd.Series(r)
        sigma = ret_series.rolling(VAR_WINDOW, min_periods=VAR_WINDOW).std(ddof=1)
        notional = (10000.0 / (VAR_Z * sigma))
        expected = (notional.shift(1) * ret_series).where(notional.shift(1).notna(), 0.0).fillna(0.0)
        np.testing.assert_allclose(scaled["p"].to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-9)

    def test_sigma_zero_guard_no_blowup(self, daily_dates):
        n = len(daily_dates)
        strat = pd.DataFrame({"date": daily_dates, "p": np.ones(n)})
        rets = pd.DataFrame({"date": daily_dates, "p": np.full(n, 0.005)})  # constant -> std 0
        scaled, _ = compute_var_scaled_frame(strat, rets, ["p"], 10000.0, {"p": 100})
        assert np.isfinite(scaled["p"].to_numpy()).all()
        assert (scaled["p"] == 0.0).all()

    def test_missing_column_skipped(self, sample_strategy_df, sample_returns_df):
        rets = sample_returns_df.drop(columns=["Gasoil_M3"])
        scaled, diag = compute_var_scaled_frame(
            sample_strategy_df, rets,
            ["Brent_M3", "Gasoil_M3"], 10000, {"Brent_M3": 50, "Gasoil_M3": 50},
        )
        assert "Gasoil_M3" in diag["skipped"]
        assert "Gasoil_M3" not in scaled.columns
        assert "Brent_M3" in scaled.columns

    def test_date_inner_join_alignment(self, daily_dates):
        n = len(daily_dates)
        strat = pd.DataFrame({"date": daily_dates, "p": np.ones(n)})
        # returns cover only the first 100 dates
        rets = pd.DataFrame({"date": daily_dates.iloc[:100], "p": np.linspace(-0.01, 0.01, 100)})
        scaled, _ = compute_var_scaled_frame(strat, rets, ["p"], 10000, {"p": 100})
        assert len(scaled) == 100

    def test_60_40_notional_ratio(self, sample_strategy_df, sample_returns_df):
        # Same product cloned so sigma is identical -> notionals scale with allocation.
        rets = sample_returns_df.copy()
        rets["Gasoil_M3"] = rets["Brent_M3"]
        strat = sample_strategy_df.copy()
        _, diag = compute_var_scaled_frame(
            strat, rets, ["Brent_M3", "Gasoil_M3"], 10000, {"Brent_M3": 60, "Gasoil_M3": 40},
        )
        a = diag["latest"]["Brent_M3"]["notional"]
        b = diag["latest"]["Gasoil_M3"]["notional"]
        assert a / b == pytest.approx(60.0 / 40.0, rel=1e-6)

    def test_empty_intersection_returns_empty(self, daily_dates):
        strat = pd.DataFrame({"date": daily_dates, "p": np.ones(len(daily_dates))})
        rets = pd.DataFrame({"date": pd.bdate_range("2000-01-03", periods=30), "p": 0.01})
        scaled, _ = compute_var_scaled_frame(strat, rets, ["p"], 10000, {"p": 100})
        assert scaled.empty

    def test_feeds_compute_series(self, sample_strategy_df, sample_returns_df):
        scaled, _ = compute_var_scaled_frame(
            sample_strategy_df, sample_returns_df,
            ["Brent_M3", "Gasoil_M3"], 10000, {"Brent_M3": 50, "Gasoil_M3": 50},
        )
        sp = compute_series(scaled, ["Brent_M3", "Gasoil_M3"], DEFAULT_INITIAL_CAPITAL)
        assert len(sp.equity) == len(scaled)
        assert sp.equity.iloc[0] == pytest.approx(DEFAULT_INITIAL_CAPITAL + scaled[["Brent_M3", "Gasoil_M3"]].sum(axis=1).iloc[0])


# ---------------------------------------------------------------------------
# portfolio_effective_dataframe
# ---------------------------------------------------------------------------
class TestPortfolioEffectiveDataframe:
    def test_all_raw_matches_portfolio_dataframe(self, sample_strategy_df):
        strategies = {"S1": sample_strategy_df, "S2": sample_strategy_df}
        merged, breakdown = portfolio_effective_dataframe(strategies, {}, ["S1", "S2"], {})
        expected = portfolio_dataframe(strategies, ["S1", "S2"])
        pd.testing.assert_frame_equal(merged, expected, check_like=True)
        assert breakdown["S1"]["var_on"] is False

    def test_active_strategy_uses_scaled(self, sample_strategy_df, sample_returns_df):
        strategies = {"S1": sample_strategy_df}
        returns = {"S1": sample_returns_df}
        cfg = {"S1": {"total_var": 10000, "allocations": {"Brent_M3": 50, "Gasoil_M3": 50}, "active": True}}
        merged, breakdown = portfolio_effective_dataframe(strategies, returns, ["S1"], cfg)
        assert breakdown["S1"]["var_on"] is True
        assert "S1" in merged.columns

    def test_invalid_config_falls_back_to_raw(self, sample_strategy_df, sample_returns_df):
        strategies = {"S1": sample_strategy_df}
        returns = {"S1": sample_returns_df}
        # allocations don't sum to 100 -> invalid -> raw
        cfg = {"S1": {"total_var": 10000, "allocations": {"Brent_M3": 10, "Gasoil_M3": 10}, "active": True}}
        merged, breakdown = portfolio_effective_dataframe(strategies, returns, ["S1"], cfg)
        assert breakdown["S1"]["var_on"] is False
        raw = sample_strategy_df[["Brent_M3", "Gasoil_M3"]].sum(axis=1)
        np.testing.assert_allclose(merged["S1"].to_numpy(), raw.to_numpy())
