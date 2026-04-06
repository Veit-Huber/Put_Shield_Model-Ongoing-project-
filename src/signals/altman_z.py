"""Altman Z-Score bankruptcy predictor signal.

Two variants are implemented:
  - Original Z  (manufacturing firms, SIC 2000-3999):
      Z = 1.2(WC/TA) + 1.4(RE/TA) + 3.3(EBIT/TA) + 0.6(MVE/TL) + 1.0(Sales/TA)
      Hard stop: Z < 1.81; safe zone: Z > 2.99

  - Z'' (non-manufacturing / private firms):
      Z'' = 6.56(WC/TA) + 3.26(RE/TA) + 6.72(EBIT/TA) + 1.05(BVE/TL)
      Hard stop: Z'' < 1.23; safe zone: Z'' > 2.60

Sector classification (Yahoo Finance sector string → formula):
  - Industrials, Basic Materials, Energy, Consumer Cyclical,
    Consumer Defensive, Healthcare, Real Estate  → use Z'' (non-mfg)
    unless SIC says otherwise
  - Technology, Communication Services, Financial Services, Utilities
    → always use Z'' (no reliable sales/assets ratio)

The module defaults to Z'' for any ticker whose sector cannot be determined.

Normalization:
  Score = 1.0 when Z ≤ norm_low  (most dangerous)
  Score = 0.0 when Z ≥ norm_high (safest)
  Linear interpolation between norm_low and norm_high
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.yahoo import YahooDataSource
from src.signals.base import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

# Sectors that should use the original Z (manufacturing-heavy)
MANUFACTURING_SECTORS = {
    "industrials",
    "basic materials",
    "consumer cyclical",  # includes auto / consumer goods mfg
}

# Sectors for which the Z-Score is not meaningful and should return NaN.
# Banks and insurers carry structural leverage that makes every ratio misleading.
# When more sector-specific signals are added (Tier 1 capital, NPL ratio, etc.)
# this guard can be relaxed for those signals specifically.
EXCLUDED_SECTORS = {
    "financial services",
    "financials",
    "banks",
    "insurance",
}

# Hard-stop thresholds
HARD_STOP_Z_ORIG = 1.81
HARD_STOP_Z_PRIME = 1.23

# Default normalization range — can be overridden via config
DEFAULT_NORM_LOW = 1.0
DEFAULT_NORM_HIGH = 3.5


def _normalize(z: float, norm_low: float, norm_high: float) -> float:
    """Map Z-Score to [0, 1] danger score (1 = most dangerous)."""
    if np.isnan(z):
        return np.nan
    if z <= norm_low:
        return 1.0
    if z >= norm_high:
        return 0.0
    return 1.0 - (z - norm_low) / (norm_high - norm_low)


class AltmanZSignal(BaseSignal):
    """Computes Altman Z-Score and normalizes to a 0–1 danger score."""

    name = "altman_z"

    def __init__(
        self,
        data_source: YahooDataSource | None = None,
        cache_dir: str | Path = "data/raw",
        hard_stop_z: float = HARD_STOP_Z_ORIG,
        norm_low: float = DEFAULT_NORM_LOW,
        norm_high: float = DEFAULT_NORM_HIGH,
    ) -> None:
        self.ds = data_source or YahooDataSource(cache_dir=cache_dir)
        self.hard_stop_z = hard_stop_z
        self.norm_low = norm_low
        self.norm_high = norm_high

    # ------------------------------------------------------------------
    # BaseSignal interface
    # ------------------------------------------------------------------

    def compute(self, tickers: list[str], date: pd.Timestamp) -> SignalResult:
        """Compute Altman Z-Score for each ticker as of *date*."""
        records = []
        for ticker in tickers:
            rec = self._compute_one(ticker, date)
            rec["ticker"] = ticker
            records.append(rec)

        df = pd.DataFrame(records).set_index("ticker")

        scores = df["score"]
        flags = df["hard_stop"].fillna(False).astype(bool)
        raw_values = df.drop(columns=["score", "hard_stop"])

        return SignalResult(
            scores=scores,
            raw_values=raw_values,
            flags=flags,
            signal_name=self.name,
            as_of_date=date,
        )

    # ------------------------------------------------------------------
    # Per-ticker computation
    # ------------------------------------------------------------------

    def _compute_one(self, ticker: str, date: pd.Timestamp) -> dict:
        """Return a dict with z_score, score, hard_stop, and raw inputs."""
        empty = {
            "z_score": np.nan,
            "score": np.nan,
            "hard_stop": False,
            "formula": "unknown",
            "wc_ta": np.nan,
            "re_ta": np.nan,
            "ebit_ta": np.nan,
            "mve_tl": np.nan,
            "rev_ta": np.nan,
        }

        if self._is_financial(ticker):
            logger.debug("Skipping Altman Z-Score for financial sector ticker %s", ticker)
            return empty

        try:
            fundamentals = self.ds.get_fundamentals(ticker)
        except Exception as exc:
            logger.warning("Failed to fetch fundamentals for %s: %s", ticker, exc)
            return empty

        if fundamentals is None or fundamentals.empty:
            logger.debug("No fundamental data for %s", ticker)
            return empty

        # Use the most recent filing available on or before *date*
        available = fundamentals[fundamentals.index <= date]
        if available.empty:
            logger.debug("No data available on or before %s for %s", date, ticker)
            return empty

        row = available.iloc[-1]

        # Determine which formula to use
        sector = self._get_sector(ticker)
        use_manufacturing = sector.lower() in MANUFACTURING_SECTORS

        ta = row.get("total_assets", np.nan)
        if pd.isna(ta) or ta == 0:
            logger.debug("total_assets missing or zero for %s", ticker)
            return empty

        wc = row.get("working_capital", np.nan)
        re = row.get("retained_earnings", np.nan)
        ebit = row.get("ebit", np.nan)
        tl = row.get("total_liabilities", np.nan)
        rev = row.get("revenue", np.nan)
        mve = row.get("market_cap", np.nan)
        bve = row.get("book_value_equity", np.nan)

        if use_manufacturing and pd.notna(mve) and pd.notna(rev):
            z, formula = self._z_original(wc, re, ebit, mve, rev, tl, ta)
        else:
            equity = bve if pd.notna(bve) else mve  # prefer book, fall back to market
            z, formula = self._z_prime(wc, re, ebit, equity, tl, ta)

        if np.isnan(z):
            return empty

        hard_stop_threshold = HARD_STOP_Z_ORIG if formula == "Z" else HARD_STOP_Z_PRIME
        hard_stop = bool(z < hard_stop_threshold)
        score = _normalize(z, self.norm_low, self.norm_high)

        return {
            "z_score": z,
            "score": score,
            "hard_stop": hard_stop,
            "formula": formula,
            "wc_ta": wc / ta if pd.notna(wc) else np.nan,
            "re_ta": re / ta if pd.notna(re) else np.nan,
            "ebit_ta": ebit / ta if pd.notna(ebit) else np.nan,
            "mve_tl": (mve if formula == "Z" else bve) / tl
                      if (pd.notna(tl) and tl != 0) else np.nan,
            "rev_ta": rev / ta if pd.notna(rev) else np.nan,
        }

    # ------------------------------------------------------------------
    # Formula implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _z_original(
        wc: float, re: float, ebit: float,
        mve: float, rev: float, tl: float, ta: float,
    ) -> tuple[float, str]:
        """Original Altman Z (1968) — manufacturing firms."""
        required = [wc, re, ebit, mve, rev, tl, ta]
        if any(np.isnan(v) for v in required) or ta == 0 or tl == 0:
            return np.nan, "Z"
        z = (
            1.2 * (wc / ta)
            + 1.4 * (re / ta)
            + 3.3 * (ebit / ta)
            + 0.6 * (mve / tl)
            + 1.0 * (rev / ta)
        )
        return z, "Z"

    @staticmethod
    def _z_prime(
        wc: float, re: float, ebit: float,
        equity: float, tl: float, ta: float,
    ) -> tuple[float, str]:
        """Altman Z'' (1995) — non-manufacturing / service / financial firms."""
        required = [wc, re, ebit, equity, tl, ta]
        if any(np.isnan(v) for v in required) or ta == 0 or tl == 0:
            return np.nan, "Z''"
        z = (
            6.56 * (wc / ta)
            + 3.26 * (re / ta)
            + 6.72 * (ebit / ta)
            + 1.05 * (equity / tl)
        )
        return z, "Z''"

    # ------------------------------------------------------------------
    # Sector lookup
    # ------------------------------------------------------------------

    def _get_sector(self, ticker: str) -> str:
        """Return the Yahoo Finance sector string for *ticker*, or '' on failure."""
        try:
            info = self.ds._fetch_with_retry(
                lambda: yf.Ticker(ticker).info, ticker  # noqa: F821
            )
            if info:
                return info.get("sector", "")
        except Exception:
            pass
        return ""

    def _is_financial(self, ticker: str) -> bool:
        """Return True if this ticker is in a sector where Z-Score is not meaningful."""
        return self._get_sector(ticker).lower() in EXCLUDED_SECTORS


# Lazy import to keep module importable without yfinance installed at parse time
try:
    import yfinance as yf  # noqa: F401 — used inside _get_sector lambda
except ImportError:
    yf = None  # type: ignore[assignment]
