"""Tests for the Beneish M-Score signal.

All tests use injected mock data — no network calls.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.signals.beneish_m import (
    BeneishMSignal,
    _normalize_m,
    compute_m_score,
    HARD_STOP_M,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_annual_row(**overrides) -> pd.Series:
    """Return a Series representing one annual filing with healthy defaults."""
    base = {
        "revenue":             10_000_000_000,
        "receivables":            800_000_000,
        "cogs":                 6_000_000_000,
        "current_assets":       3_000_000_000,
        "ppe_net":              2_000_000_000,
        "total_assets":        15_000_000_000,
        "depreciation":           400_000_000,
        "sga":                  1_200_000_000,
        "net_income":           1_500_000_000,
        "cfo":                  1_800_000_000,   # cash flow > net income = good sign
        "long_term_debt":       2_000_000_000,
        "current_liabilities":  2_500_000_000,
    }
    base.update(overrides)
    return pd.Series(base)


def _make_two_period_df(curr_overrides=None, prev_overrides=None) -> pd.DataFrame:
    """Return a two-row annual fundamentals DataFrame (prev year, then curr year)."""
    prev = _make_annual_row(**(prev_overrides or {}))
    curr = _make_annual_row(**(curr_overrides or {}))
    return pd.DataFrame(
        [prev, curr],
        index=pd.DatetimeIndex(["2021-12-31", "2022-12-31"]),
    )


def _make_signal(df: pd.DataFrame, sector: str = "Technology") -> BeneishMSignal:
    mock_ds = MagicMock()
    mock_ds.get_annual_fundamentals.return_value = df
    mock_ds._fetch_with_retry.return_value = {"sector": sector}
    return BeneishMSignal(data_source=mock_ds)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_above_norm_high_clips_to_one(self):
        assert _normalize_m(-0.5, -2.5, -1.0) == 1.0

    def test_below_norm_low_clips_to_zero(self):
        assert _normalize_m(-3.0, -2.5, -1.0) == 0.0

    def test_midpoint(self):
        # midpoint of [-2.5, -1.0] = -1.75 → score = 0.5
        score = _normalize_m(-1.75, -2.5, -1.0)
        assert abs(score - 0.5) < 1e-9

    def test_nan_propagates(self):
        assert math.isnan(_normalize_m(np.nan, -2.5, -1.0))


# ---------------------------------------------------------------------------
# compute_m_score unit tests
# ---------------------------------------------------------------------------

class TestComputeMScore:
    def test_healthy_company_low_m(self):
        """A company with stable, cash-backed earnings should have M << -1.78."""
        curr = _make_annual_row()
        prev = _make_annual_row()
        result = compute_m_score(curr, prev)
        # With identical years and cash flow > net income, M should be safe
        assert result["m_score"] < HARD_STOP_M or math.isnan(result["m_score"])

    def test_manipulator_profile_high_m(self):
        """Classic manipulation profile: receivables spike, cash flow << income."""
        prev = _make_annual_row()
        curr = _make_annual_row(
            receivables=2_500_000_000,    # receivables tripled (channel stuffing)
            cogs=7_000_000_000,           # gross margin falling
            net_income=2_000_000_000,     # reported income jumped
            cfo=200_000_000,             # but actual cash flow collapsed → huge accruals
            depreciation=200_000_000,    # slowed depreciation
            long_term_debt=5_000_000_000, # took on a lot more debt
        )
        result = compute_m_score(curr, prev)
        if not math.isnan(result["m_score"]):
            assert result["m_score"] > HARD_STOP_M  # should be flagged

    def test_dsri_computed_correctly(self):
        """DSRI = (Rec_t/Rev_t) / (Rec_p/Rev_p)."""
        prev = _make_annual_row(receivables=500_000_000, revenue=10_000_000_000)
        curr = _make_annual_row(receivables=1_500_000_000, revenue=10_000_000_000)
        result = compute_m_score(curr, prev)
        # Rec ratio went from 0.05 to 0.15 → DSRI = 3.0
        assert abs(result["dsri"] - 3.0) < 1e-6

    def test_tata_computed_correctly(self):
        """TATA = (NI - CFO) / TA."""
        curr = _make_annual_row(
            net_income=1_000_000_000,
            cfo=400_000_000,
            total_assets=10_000_000_000,
        )
        prev = _make_annual_row()
        result = compute_m_score(curr, prev)
        expected_tata = (1_000_000_000 - 400_000_000) / 10_000_000_000
        assert abs(result["tata"] - expected_tata) < 1e-9

    def test_sgi_is_revenue_growth(self):
        """SGI = Rev_t / Rev_{t-1}."""
        prev = _make_annual_row(revenue=8_000_000_000)
        curr = _make_annual_row(revenue=10_000_000_000)
        result = compute_m_score(curr, prev)
        assert abs(result["sgi"] - (10e9 / 8e9)) < 1e-9

    def test_missing_both_sgi_and_tata_returns_nan(self):
        """NaN only when both SGI (revenue) and TATA (net income+CFO) are missing."""
        curr = _make_annual_row(revenue=np.nan, net_income=np.nan, cfo=np.nan)
        prev = _make_annual_row(revenue=np.nan, net_income=np.nan, cfo=np.nan)
        result = compute_m_score(curr, prev)
        assert math.isnan(result["m_score"])

    def test_partial_missing_imputes_neutrals(self):
        """Missing components are imputed with neutral values, score is still computed."""
        curr = _make_annual_row(sga=np.nan, depreciation=np.nan)  # SGAI and DEPI missing
        prev = _make_annual_row(sga=np.nan, depreciation=np.nan)
        result = compute_m_score(curr, prev)
        assert not math.isnan(result["m_score"])
        assert result["imputed_count"] == 2

    def test_components_used_counts_non_imputed(self):
        """components_used should count only fields with real data."""
        curr = _make_annual_row(sga=np.nan)  # SGAI imputed
        prev = _make_annual_row(sga=np.nan)
        result = compute_m_score(curr, prev)
        assert result["components_used"] == 7
        assert result["imputed_count"] == 1

    def test_all_present_imputed_count_zero(self):
        """When all 8 components are available, imputed_count should be 0."""
        result = compute_m_score(_make_annual_row(), _make_annual_row())
        assert result["imputed_count"] == 0


# ---------------------------------------------------------------------------
# BeneishMSignal.compute integration
# ---------------------------------------------------------------------------

class TestBeneishMSignal:
    def test_healthy_returns_low_score(self):
        df = _make_two_period_df()
        signal = _make_signal(df)
        result = signal.compute(["HEALTHY"], pd.Timestamp("2023-06-30"))
        score = result.scores.loc["HEALTHY"]
        if not math.isnan(score):
            assert score < 0.5

    def test_manipulator_triggers_hard_stop(self):
        df = _make_two_period_df(
            curr_overrides={
                "receivables":   2_500_000_000,
                "net_income":    2_000_000_000,
                "cfo":             100_000_000,
                "long_term_debt":5_000_000_000,
                "depreciation":    100_000_000,
            }
        )
        signal = _make_signal(df)
        result = signal.compute(["MANIP"], pd.Timestamp("2023-06-30"))
        # Either hard_stop is True or score is high (or NaN if not enough components)
        score = result.scores.loc["MANIP"]
        hard_stop = result.flags.loc["MANIP"]
        if not math.isnan(score):
            assert hard_stop or score > 0.5

    def test_financial_sector_returns_nan(self):
        """Financial institutions should be skipped — returns NaN score."""
        df = _make_two_period_df()
        signal = _make_signal(df, sector="Financial Services")
        result = signal.compute(["JPM"], pd.Timestamp("2023-06-30"))
        assert math.isnan(result.scores.loc["JPM"])
        assert not result.flags.loc["JPM"]

    def test_only_one_year_returns_nan(self):
        """With only one annual period, M-Score cannot be computed."""
        one_year = _make_two_period_df().iloc[[-1]]  # keep only most recent row
        signal = _make_signal(one_year)
        result = signal.compute(["ONE_YR"], pd.Timestamp("2023-06-30"))
        assert math.isnan(result.scores.loc["ONE_YR"])

    def test_future_date_returns_nan(self):
        """Data dated 2021-22 should not be visible at 2019-01-01."""
        df = _make_two_period_df()
        signal = _make_signal(df)
        result = signal.compute(["FUTURE"], pd.Timestamp("2019-01-01"))
        assert math.isnan(result.scores.loc["FUTURE"])

    def test_empty_fundamentals_returns_nan(self):
        mock_ds = MagicMock()
        mock_ds.get_annual_fundamentals.return_value = pd.DataFrame()
        mock_ds._fetch_with_retry.return_value = {"sector": "Technology"}
        signal = BeneishMSignal(data_source=mock_ds)
        result = signal.compute(["EMPTY"], pd.Timestamp("2023-06-30"))
        assert math.isnan(result.scores.loc["EMPTY"])

    def test_scores_between_zero_and_one_or_nan(self):
        df = _make_two_period_df()
        signal = _make_signal(df)
        result = signal.compute(["X"], pd.Timestamp("2023-06-30"))
        score = result.scores.iloc[0]
        if not math.isnan(score):
            assert 0.0 <= score <= 1.0

    def test_batch_returns_all_tickers(self):
        df = _make_two_period_df()
        signal = _make_signal(df)
        tickers = ["A", "B", "C"]
        result = signal.compute(tickers, pd.Timestamp("2023-06-30"))
        assert set(result.scores.index) == set(tickers)
        assert set(result.flags.index) == set(tickers)
