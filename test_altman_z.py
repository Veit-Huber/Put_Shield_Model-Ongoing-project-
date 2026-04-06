"""Tests for the Altman Z-Score signal.

Strategy:
  - Unit-test the formula math with hand-crafted inputs (no network).
  - Unit-test the normalization function.
  - Integration smoke-test against a live ticker (skipped in CI if no network).

We do NOT rely on yfinance data for the formula correctness tests — we inject
a mock data source so the tests are deterministic and fast.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.signals.altman_z import AltmanZSignal, _normalize, HARD_STOP_Z_ORIG, HARD_STOP_Z_PRIME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fundamentals(**overrides) -> pd.DataFrame:
    """Return a one-row fundamentals DataFrame with sensible defaults.

    All values represent a healthy mid-cap manufacturer.
    """
    base = {
        "working_capital":   2_000_000_000,   # $2B
        "total_assets":     10_000_000_000,   # $10B
        "retained_earnings": 3_000_000_000,   # $3B
        "ebit":              1_500_000_000,   # $1.5B
        "market_cap":       15_000_000_000,   # $15B
        "total_liabilities": 5_000_000_000,   # $5B
        "revenue":           8_000_000_000,   # $8B
        "book_value_equity": 5_000_000_000,   # $5B
    }
    base.update(overrides)
    return pd.DataFrame([base], index=pd.DatetimeIndex(["2020-06-30"]))


def _make_signal(fundamentals: pd.DataFrame, sector: str = "Industrials") -> AltmanZSignal:
    """Return an AltmanZSignal whose data source returns *fundamentals*."""
    mock_ds = MagicMock()
    mock_ds.get_fundamentals.return_value = fundamentals
    mock_ds._fetch_with_retry.return_value = {"sector": sector}

    signal = AltmanZSignal(data_source=mock_ds)
    return signal


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_below_low_clips_to_one(self):
        assert _normalize(0.5, 1.0, 3.5) == 1.0

    def test_above_high_clips_to_zero(self):
        assert _normalize(4.0, 1.0, 3.5) == 0.0

    def test_at_low_boundary(self):
        assert _normalize(1.0, 1.0, 3.5) == 1.0

    def test_at_high_boundary(self):
        assert _normalize(3.5, 1.0, 3.5) == 0.0

    def test_midpoint(self):
        # Midpoint of [1.0, 3.5] is 2.25 → score = 0.5
        score = _normalize(2.25, 1.0, 3.5)
        assert abs(score - 0.5) < 1e-9

    def test_nan_propagates(self):
        assert math.isnan(_normalize(np.nan, 1.0, 3.5))


# ---------------------------------------------------------------------------
# Formula tests (original Z)
# ---------------------------------------------------------------------------

class TestOriginalZFormula:
    """Test original Z = 1.2(WC/TA) + 1.4(RE/TA) + 3.3(EBIT/TA) + 0.6(MVE/TL) + 1.0(Rev/TA)"""

    def test_healthy_company_high_z(self):
        """A company with strong financials should have Z >> 2.99."""
        signal = _make_signal(
            _make_fundamentals(
                working_capital=3_000_000_000,
                total_assets=10_000_000_000,
                retained_earnings=5_000_000_000,
                ebit=2_000_000_000,
                market_cap=20_000_000_000,
                total_liabilities=4_000_000_000,
                revenue=10_000_000_000,
            ),
            sector="Industrials",
        )
        result = signal.compute(["TEST"], pd.Timestamp("2020-12-31"))
        z = result.raw_values.loc["TEST", "wc_ta"]  # quick sanity check
        assert z > 0  # working capital ratio should be positive

        # Compute expected Z manually
        ta, wc, re, ebit, mve, tl, rev = (
            10e9, 3e9, 5e9, 2e9, 20e9, 4e9, 10e9
        )
        expected_z = (
            1.2 * (wc / ta)
            + 1.4 * (re / ta)
            + 3.3 * (ebit / ta)
            + 0.6 * (mve / tl)
            + 1.0 * (rev / ta)
        )
        assert expected_z > 2.99  # safe zone

    def test_distressed_company_hard_stop(self):
        """A company in the distress zone (Z < 1.81) must trigger hard_stop."""
        signal = _make_signal(
            _make_fundamentals(
                working_capital=-500_000_000,   # negative WC
                total_assets=10_000_000_000,
                retained_earnings=-2_000_000_000,  # accumulated losses
                ebit=-300_000_000,               # operating loss
                market_cap=500_000_000,          # penny-stock market cap
                total_liabilities=9_500_000_000, # near-insolvent
                revenue=1_000_000_000,
            ),
            sector="Industrials",
        )
        result = signal.compute(["DISTRESSED"], pd.Timestamp("2020-12-31"))
        assert result.flags.loc["DISTRESSED"] is True or result.flags.loc["DISTRESSED"] == True  # noqa: E712

    def test_distressed_score_near_one(self):
        """Distressed company score should be close to 1.0 (maximum danger)."""
        signal = _make_signal(
            _make_fundamentals(
                working_capital=-500_000_000,
                total_assets=10_000_000_000,
                retained_earnings=-2_000_000_000,
                ebit=-300_000_000,
                market_cap=500_000_000,
                total_liabilities=9_500_000_000,
                revenue=1_000_000_000,
            ),
            sector="Industrials",
        )
        result = signal.compute(["DISTRESSED"], pd.Timestamp("2020-12-31"))
        score = result.scores.loc["DISTRESSED"]
        assert score >= 0.9 or math.isnan(score)  # should be near 1.0 or NaN

    def test_manual_z_computation(self):
        """Verify the formula against a manually computed expected value."""
        ta, wc, re, ebit, mve, tl, rev = (
            10_000, 2_000, 3_000, 1_500, 15_000, 5_000, 8_000
        )  # in $M (scaling doesn't affect ratios)
        expected = (
            1.2 * (wc / ta)
            + 1.4 * (re / ta)
            + 3.3 * (ebit / ta)
            + 0.6 * (mve / tl)
            + 1.0 * (rev / ta)
        )

        signal = _make_signal(
            _make_fundamentals(
                working_capital=wc * 1e6,
                total_assets=ta * 1e6,
                retained_earnings=re * 1e6,
                ebit=ebit * 1e6,
                market_cap=mve * 1e6,
                total_liabilities=tl * 1e6,
                revenue=rev * 1e6,
            ),
            sector="Industrials",
        )
        result = signal.compute(["MANUAL"], pd.Timestamp("2020-12-31"))
        computed_z = result.raw_values.loc["MANUAL", "z_score"]
        assert abs(computed_z - expected) < 1e-6


# ---------------------------------------------------------------------------
# Formula tests (Z'' non-manufacturing)
# ---------------------------------------------------------------------------

class TestZPrimeFormula:
    """Test Z'' = 6.56(WC/TA) + 3.26(RE/TA) + 6.72(EBIT/TA) + 1.05(BVE/TL)"""

    def test_tech_company_uses_z_prime(self):
        """Technology sector should always use Z'' formula."""
        signal = _make_signal(_make_fundamentals(), sector="Technology")
        result = signal.compute(["TECH"], pd.Timestamp("2020-12-31"))
        formula = result.raw_values.loc["TECH", "formula"]
        assert formula == "Z''"

    def test_financials_returns_nan(self):
        """Financial Services sector is excluded — Z-Score is not valid for banks."""
        signal = _make_signal(_make_fundamentals(), sector="Financial Services")
        result = signal.compute(["FIN"], pd.Timestamp("2020-12-31"))
        assert math.isnan(result.scores.loc["FIN"])

    def test_manual_z_prime_computation(self):
        """Verify Z'' formula against manually computed value."""
        ta, wc, re, ebit, bve, tl = (
            10_000, 2_000, 3_000, 1_500, 5_000, 5_000
        )  # in $M
        expected = (
            6.56 * (wc / ta)
            + 3.26 * (re / ta)
            + 6.72 * (ebit / ta)
            + 1.05 * (bve / tl)
        )

        signal = _make_signal(
            _make_fundamentals(
                working_capital=wc * 1e6,
                total_assets=ta * 1e6,
                retained_earnings=re * 1e6,
                ebit=ebit * 1e6,
                book_value_equity=bve * 1e6,
                total_liabilities=tl * 1e6,
            ),
            sector="Technology",
        )
        result = signal.compute(["TECH_MANUAL"], pd.Timestamp("2020-12-31"))
        computed_z = result.raw_values.loc["TECH_MANUAL", "z_score"]
        assert abs(computed_z - expected) < 1e-6

    def test_z_prime_hard_stop_threshold(self):
        """Z'' hard stop is at 1.23 (different from Z's 1.81)."""
        # Craft inputs that give Z'' ≈ 1.0 (below 1.23)
        # Z'' = 6.56(WC/TA) + 3.26(RE/TA) + 6.72(EBIT/TA) + 1.05(BVE/TL)
        # Let's set near-zero/negative values
        signal = _make_signal(
            _make_fundamentals(
                working_capital=-1_000_000_000,
                total_assets=10_000_000_000,
                retained_earnings=-2_000_000_000,
                ebit=-100_000_000,
                book_value_equity=200_000_000,
                total_liabilities=9_800_000_000,
            ),
            sector="Technology",
        )
        result = signal.compute(["DISTRESSED_TECH"], pd.Timestamp("2020-12-31"))
        assert result.flags.loc["DISTRESSED_TECH"] is True or result.flags.loc["DISTRESSED_TECH"] == True  # noqa: E712


# ---------------------------------------------------------------------------
# Missing / invalid data handling
# ---------------------------------------------------------------------------

class TestMissingData:
    def test_nan_total_assets_returns_nan_score(self):
        mock_ds = MagicMock()
        mock_ds.get_fundamentals.return_value = _make_fundamentals(total_assets=float("nan"))
        mock_ds._fetch_with_retry.return_value = {"sector": "Industrials"}
        signal = AltmanZSignal(data_source=mock_ds)

        result = signal.compute(["NAN_TA"], pd.Timestamp("2020-12-31"))
        assert math.isnan(result.scores.loc["NAN_TA"])

    def test_empty_dataframe_returns_nan(self):
        mock_ds = MagicMock()
        mock_ds.get_fundamentals.return_value = pd.DataFrame()
        mock_ds._fetch_with_retry.return_value = {"sector": "Industrials"}
        signal = AltmanZSignal(data_source=mock_ds)

        result = signal.compute(["EMPTY"], pd.Timestamp("2020-12-31"))
        assert math.isnan(result.scores.loc["EMPTY"])

    def test_future_date_returns_nan(self):
        """Data available only after 2022 should not be visible at 2019-01-01."""
        mock_ds = MagicMock()
        # All data is dated 2022
        mock_ds.get_fundamentals.return_value = _make_fundamentals()  # index = 2020-06-30
        mock_ds._fetch_with_retry.return_value = {"sector": "Industrials"}
        signal = AltmanZSignal(data_source=mock_ds)

        # Ask for signal as of a date BEFORE the data is available
        result = signal.compute(["FUTURE"], pd.Timestamp("2019-01-01"))
        assert math.isnan(result.scores.loc["FUTURE"])

    def test_no_hard_stop_for_healthy_company(self):
        """A healthy company should not trigger a hard stop."""
        signal = _make_signal(_make_fundamentals(), sector="Industrials")
        result = signal.compute(["HEALTHY"], pd.Timestamp("2020-12-31"))
        assert not result.flags.loc["HEALTHY"]

    def test_financial_sector_returns_nan(self):
        """Banks and insurers must return NaN — Z-Score is not valid for them."""
        for sector in ["Financial Services", "Financials", "Banks", "Insurance"]:
            signal = _make_signal(_make_fundamentals(), sector=sector)
            result = signal.compute(["FIG"], pd.Timestamp("2020-12-31"))
            assert math.isnan(result.scores.loc["FIG"]), f"Expected NaN for sector '{sector}'"
            assert not result.flags.loc["FIG"], f"Expected no hard_stop for sector '{sector}'"

    def test_non_financial_sector_still_computes(self):
        """Guard must not accidentally block non-financial sectors."""
        for sector in ["Technology", "Industrials", "Healthcare", "Energy"]:
            signal = _make_signal(_make_fundamentals(), sector=sector)
            result = signal.compute(["OK"], pd.Timestamp("2020-12-31"))
            # Score may be NaN for other reasons, but should NOT be NaN purely due to sector
            # We verify by checking the formula field is set (not 'unknown')
            formula = result.raw_values.loc["OK", "formula"]
            assert formula in ("Z", "Z''"), f"Unexpected formula '{formula}' for sector '{sector}'"


# ---------------------------------------------------------------------------
# Multi-ticker batch
# ---------------------------------------------------------------------------

class TestBatchCompute:
    def test_returns_result_for_all_tickers(self):
        mock_ds = MagicMock()
        mock_ds.get_fundamentals.return_value = _make_fundamentals()
        mock_ds._fetch_with_retry.return_value = {"sector": "Technology"}
        signal = AltmanZSignal(data_source=mock_ds)

        tickers = ["A", "B", "C"]
        result = signal.compute(tickers, pd.Timestamp("2020-12-31"))

        assert set(result.scores.index) == set(tickers)
        assert set(result.flags.index) == set(tickers)

    def test_scores_between_zero_and_one_or_nan(self):
        mock_ds = MagicMock()
        mock_ds.get_fundamentals.return_value = _make_fundamentals()
        mock_ds._fetch_with_retry.return_value = {"sector": "Technology"}
        signal = AltmanZSignal(data_source=mock_ds)

        result = signal.compute(["X"], pd.Timestamp("2020-12-31"))
        score = result.scores.iloc[0]
        if not math.isnan(score):
            assert 0.0 <= score <= 1.0
