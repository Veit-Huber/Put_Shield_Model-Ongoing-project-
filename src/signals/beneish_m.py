"""Beneish M-Score earnings manipulation detector.

Messod Beneish (1999) identified 8 financial ratios that, combined, predict
whether a company is manipulating its reported earnings.  Enron, WorldCom,
and Valeant all had M-Scores well above the threshold before they collapsed.

Formula:
  M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
            + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

Components (all are year-over-year ratios — requires two annual periods):
  DSRI  Days Sales Receivables Index  — receivables growing faster than revenue?
  GMI   Gross Margin Index            — gross margin deteriorating?
  AQI   Asset Quality Index           — shift toward non-productive assets?
  SGI   Sales Growth Index            — high growth (can pressure manipulation)
  DEPI  Depreciation Index            — slowing depreciation to boost earnings?
  SGAI  SGA Expense Index             — overhead growing faster than sales?
  TATA  Total Accruals to Total Assets— gap between reported income and cash flow
  LVGI  Leverage Index                — taking on more debt?

Hard stop:  M > -1.78  (Beneish's original threshold — likely manipulator)
Caution:    M > -2.22  (wider net, catches more borderline cases)

Normalization:
  Score = 1.0 when M ≥ norm_high  (most dangerous, clear manipulator)
  Score = 0.0 when M ≤ norm_low   (safe, clearly not manipulating)
  Linear interpolation between norm_low and norm_high

Important limitations:
  - Requires TWO consecutive annual periods of data.  Returns NaN if only
    one year is available.
  - NOT designed for financial institutions (banks, insurers).  Their
    balance sheets are structured differently — receivables and leverage
    mean something completely different.  The signal returns NaN for
    financials rather than a misleading score.
  - SGAI is often NaN because yfinance doesn't reliably report SGA separately.
    When SGAI is missing it is dropped from the formula (partial M-Score).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.yahoo import YahooDataSource
from src.signals.base import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

# Hard-stop threshold (Beneish original)
HARD_STOP_M = -1.78
# Normalization range:  M ≤ -2.5 → safe (0.0),  M ≥ -1.0 → max danger (1.0)
DEFAULT_NORM_LOW  = -2.5
DEFAULT_NORM_HIGH = -1.0

# Sectors for which the M-Score is not meaningful
EXCLUDED_SECTORS = {
    "financial services",
    "financials",
    "banks",
    "insurance",
}


def _normalize_m(m: float, norm_low: float, norm_high: float) -> float:
    """Map M-Score to [0, 1] danger score.  Higher M = more dangerous."""
    if np.isnan(m):
        return np.nan
    if m >= norm_high:
        return 1.0
    if m <= norm_low:
        return 0.0
    return (m - norm_low) / (norm_high - norm_low)


def _safe_ratio(num: float, den: float) -> float:
    """Return num/den, or NaN if either is NaN or denominator is zero."""
    if np.isnan(num) or np.isnan(den) or den == 0:
        return np.nan
    return num / den


def compute_m_score(curr: pd.Series, prev: pd.Series) -> dict:
    """Compute Beneish M-Score components from two annual filing rows.

    Parameters
    ----------
    curr : pd.Series
        Most recent annual filing (the 't' year).
    prev : pd.Series
        Prior annual filing (the 't-1' year).

    Returns
    -------
    dict with keys: m_score, dsri, gmi, aqi, sgi, depi, sgai, tata, lvgi,
                    components_used (int — how many of 8 were non-NaN)
    """
    def g(row: pd.Series, key: str) -> float:
        v = row.get(key, np.nan)
        return float(v) if pd.notna(v) else np.nan

    # Convenience aliases
    rev_t   = g(curr, "revenue");       rev_p   = g(prev, "revenue")
    rec_t   = g(curr, "receivables");   rec_p   = g(prev, "receivables")
    cogs_t  = g(curr, "cogs");          cogs_p  = g(prev, "cogs")
    ca_t    = g(curr, "current_assets");ca_p    = g(prev, "current_assets")
    ppe_t   = g(curr, "ppe_net");       ppe_p   = g(prev, "ppe_net")
    ta_t    = g(curr, "total_assets");  ta_p    = g(prev, "total_assets")
    dep_t   = g(curr, "depreciation");  dep_p   = g(prev, "depreciation")
    sga_t   = g(curr, "sga");           sga_p   = g(prev, "sga")
    ni_t    = g(curr, "net_income")
    cfo_t   = g(curr, "cfo")
    ltd_t   = g(curr, "long_term_debt");ltd_p   = g(prev, "long_term_debt")
    cl_t    = g(curr, "current_liabilities"); cl_p = g(prev, "current_liabilities")

    # ── DSRI: Days Sales Receivables Index ───────────────────────────────────
    # (Rec_t / Rev_t) / (Rec_{t-1} / Rev_{t-1})
    # > 1 means receivables grew faster than sales — possible channel stuffing
    dsri = _safe_ratio(
        _safe_ratio(rec_t, rev_t),
        _safe_ratio(rec_p, rev_p),
    )

    # ── GMI: Gross Margin Index ───────────────────────────────────────────────
    # [(Rev_{t-1} - COGS_{t-1}) / Rev_{t-1}] / [(Rev_t - COGS_t) / Rev_t]
    # > 1 means gross margin deteriorated — pressure to manipulate
    gm_t = _safe_ratio(rev_t - cogs_t, rev_t) if pd.notna(cogs_t) and pd.notna(rev_t) else np.nan
    gm_p = _safe_ratio(rev_p - cogs_p, rev_p) if pd.notna(cogs_p) and pd.notna(rev_p) else np.nan
    gmi  = _safe_ratio(gm_p, gm_t)

    # ── AQI: Asset Quality Index ──────────────────────────────────────────────
    # [1 - (CA_t + PPE_t) / TA_t] / [1 - (CA_{t-1} + PPE_{t-1}) / TA_{t-1}]
    # > 1 means more assets are intangible / low-quality
    def _aq(ca, ppe, ta):
        if any(np.isnan(v) for v in [ca, ppe, ta]) or ta == 0:
            return np.nan
        return 1.0 - (ca + ppe) / ta

    aqi = _safe_ratio(_aq(ca_t, ppe_t, ta_t), _aq(ca_p, ppe_p, ta_p))

    # ── SGI: Sales Growth Index ───────────────────────────────────────────────
    # Rev_t / Rev_{t-1}
    # High growth companies are under pressure to keep it up — can lead to manipulation
    sgi = _safe_ratio(rev_t, rev_p)

    # ── DEPI: Depreciation Index ──────────────────────────────────────────────
    # [Dep_{t-1} / (PPE_{t-1} + Dep_{t-1})] / [Dep_t / (PPE_t + Dep_t)]
    # > 1 means depreciation rate slowed — boosts reported earnings artificially
    def _dep_rate(dep, ppe):
        if any(np.isnan(v) for v in [dep, ppe]):
            return np.nan
        denom = ppe + dep
        return dep / denom if denom != 0 else np.nan

    depi = _safe_ratio(_dep_rate(dep_p, ppe_p), _dep_rate(dep_t, ppe_t))

    # ── SGAI: SGA Expense Index ───────────────────────────────────────────────
    # (SGA_t / Rev_t) / (SGA_{t-1} / Rev_{t-1})
    # > 1 means overhead growing faster than sales — operating leverage concern
    sgai = _safe_ratio(
        _safe_ratio(sga_t, rev_t),
        _safe_ratio(sga_p, rev_p),
    )

    # ── TATA: Total Accruals to Total Assets ──────────────────────────────────
    # (Net Income_t - CFO_t) / TA_t
    # Large positive value means reported income >> actual cash flow — red flag
    if pd.notna(ni_t) and pd.notna(cfo_t) and pd.notna(ta_t) and ta_t != 0:
        tata = (ni_t - cfo_t) / ta_t
    else:
        tata = np.nan

    # ── LVGI: Leverage Index ──────────────────────────────────────────────────
    # [(LTD_t + CL_t) / TA_t] / [(LTD_{t-1} + CL_{t-1}) / TA_{t-1}]
    # > 1 means leverage increased — possible sign of financial stress
    def _lev(ltd, cl, ta):
        if any(np.isnan(v) for v in [ltd, cl, ta]) or ta == 0:
            return np.nan
        return (ltd + cl) / ta

    lvgi = _safe_ratio(_lev(ltd_t, cl_t, ta_t), _lev(ltd_p, cl_p, ta_p))

    # ── M-Score ───────────────────────────────────────────────────────────────
    # Coefficients from Beneish (1999).
    #
    # Missing component strategy: impute with the neutral value so the
    # original -1.78 threshold remains valid:
    #   - Ratio components (DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI): neutral = 1.0
    #     (means "no change year-over-year")
    #   - TATA: neutral = 0.0
    #     (means "net income equals cash flow — no accruals")
    #
    # We track how many were imputed. When > 2 components are imputed the
    # score is flagged as low-confidence (imputed_count exposed in raw_values).
    # If the core revenue fields (revenue, total_assets) are missing entirely,
    # we still return NaN — the score would be meaningless.
    NEUTRAL: dict[str, float] = {
        "dsri": 1.0, "gmi": 1.0, "aqi": 1.0, "sgi": 1.0,
        "depi": 1.0, "sgai": 1.0, "tata": 0.0, "lvgi": 1.0,
    }

    raw_components: dict[str, float] = {
        "dsri": dsri, "gmi": gmi, "aqi": aqi, "sgi": sgi,
        "depi": depi, "sgai": sgai, "tata": tata, "lvgi": lvgi,
    }

    # Require at least SGI (needs revenue) and TATA (needs net income + CFO)
    # as a bare minimum — without these the score is uninformative.
    if np.isnan(sgi) and np.isnan(tata):
        return {
            "m_score": np.nan, "components_used": 0, "imputed_count": 8,
            **{k: np.nan for k in raw_components},
        }

    coefs = {
        "dsri": 0.920, "gmi": 0.528, "aqi": 0.404, "sgi": 0.892,
        "depi": 0.115, "sgai": -0.172, "tata": 4.679, "lvgi": -0.327,
    }

    intercept = -4.84
    score_sum = intercept
    used = 0
    imputed = 0
    for key, coef in coefs.items():
        val = raw_components[key]
        if np.isnan(val):
            val = NEUTRAL[key]
            imputed += 1
        else:
            used += 1
        score_sum += coef * val

    m_score = score_sum  # threshold -1.78 always applies

    return {
        "m_score":         m_score,
        "components_used": used,
        "imputed_count":   imputed,
        "dsri":  dsri,
        "gmi":   gmi,
        "aqi":   aqi,
        "sgi":   sgi,
        "depi":  depi,
        "sgai":  sgai,
        "tata":  tata,
        "lvgi":  lvgi,
    }


class BeneishMSignal(BaseSignal):
    """Computes Beneish M-Score and normalizes to a 0–1 danger score."""

    name = "beneish_m"

    def __init__(
        self,
        data_source: YahooDataSource | None = None,
        cache_dir: str | Path = "data/raw",
        hard_stop_m: float = HARD_STOP_M,
        norm_low: float = DEFAULT_NORM_LOW,
        norm_high: float = DEFAULT_NORM_HIGH,
    ) -> None:
        self.ds = data_source or YahooDataSource(cache_dir=cache_dir)
        self.hard_stop_m = hard_stop_m
        self.norm_low = norm_low
        self.norm_high = norm_high

    def compute(self, tickers: list[str], date: pd.Timestamp) -> SignalResult:
        records = []
        for ticker in tickers:
            rec = self._compute_one(ticker, date)
            rec["ticker"] = ticker
            records.append(rec)

        df = pd.DataFrame(records).set_index("ticker")
        scores = df["score"]
        flags  = df["hard_stop"].fillna(False).astype(bool)
        raw    = df.drop(columns=["score", "hard_stop"])

        return SignalResult(
            scores=scores,
            raw_values=raw,
            flags=flags,
            signal_name=self.name,
            as_of_date=date,
        )

    def _compute_one(self, ticker: str, date: pd.Timestamp) -> dict:
        empty = {
            "m_score": np.nan, "score": np.nan, "hard_stop": False,
            "dsri": np.nan, "gmi": np.nan, "aqi": np.nan, "sgi": np.nan,
            "depi": np.nan, "sgai": np.nan, "tata": np.nan, "lvgi": np.nan,
            "components_used": 0, "imputed_count": 0,
        }

        # Skip financial institutions — M-Score is not valid for them
        if self._is_financial(ticker):
            logger.debug("Skipping Beneish M-Score for financial sector ticker %s", ticker)
            return empty

        try:
            annual = self.ds.get_annual_fundamentals(ticker)
        except Exception as exc:
            logger.warning("Failed to fetch annual fundamentals for %s: %s", ticker, exc)
            return empty

        if annual is None or annual.empty:
            return empty

        # Only use filings available on or before the as-of date
        available = annual[annual.index <= date].sort_index()
        if len(available) < 2:
            logger.debug("Need 2 annual periods for Beneish, only %d available for %s", len(available), ticker)
            return empty

        curr = available.iloc[-1]
        prev = available.iloc[-2]

        result = compute_m_score(curr, prev)
        m = result["m_score"]

        hard_stop = bool(not np.isnan(m) and m > self.hard_stop_m)
        score = _normalize_m(m, self.norm_low, self.norm_high)

        imputed = result["imputed_count"]
        if imputed > 2:
            logger.debug(
                "%s: Beneish M-Score used %d imputed neutral values — treat as low-confidence",
                ticker, imputed,
            )

        return {
            "m_score":         m,
            "score":           score,
            "hard_stop":       hard_stop,
            "dsri":            result["dsri"],
            "gmi":             result["gmi"],
            "aqi":             result["aqi"],
            "sgi":             result["sgi"],
            "depi":            result["depi"],
            "sgai":            result["sgai"],
            "tata":            result["tata"],
            "lvgi":            result["lvgi"],
            "components_used": result["components_used"],
            "imputed_count":   imputed,
        }

    def _is_financial(self, ticker: str) -> bool:
        try:
            info = self.ds._fetch_with_retry(lambda: yf.Ticker(ticker).info, ticker)
            if info:
                sector = info.get("sector", "").lower()
                return sector in EXCLUDED_SECTORS
        except Exception:
            pass
        return False


try:
    import yfinance as yf  # noqa: F401
except ImportError:
    yf = None  # type: ignore[assignment]
