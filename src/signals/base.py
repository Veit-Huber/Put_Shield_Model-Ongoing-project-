"""Base classes for all signals."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SignalResult:
    """Standardized output from every signal.

    Attributes
    ----------
    scores : pd.Series
        Normalized danger scores, indexed by ticker.
        0.0 = safe, 1.0 = maximum danger.
        NaN means insufficient data — excluded from composite average.
    raw_values : pd.DataFrame
        Raw metric values for transparency / debugging.
        Each column is a metric; rows are tickers.
    flags : pd.Series[bool]
        Hard-stop flags, indexed by ticker.
        True = this ticker must be excluded regardless of composite score.
    signal_name : str
        Human-readable name of the signal.
    as_of_date : pd.Timestamp | None
        The date the signal was computed for.
    """

    scores: pd.Series
    raw_values: pd.DataFrame
    flags: pd.Series
    signal_name: str
    as_of_date: pd.Timestamp | None = None
    metadata: dict = field(default_factory=dict)


class BaseSignal(ABC):
    """Abstract base class for all put-shield signals.

    Every signal must:
    1. Inherit from BaseSignal.
    2. Implement ``compute(tickers, date) -> SignalResult``.
    3. Return normalized scores (0=safe, 1=danger).
    4. Return raw metric values for transparency.
    5. Return hard-stop boolean flags.
    """

    name: str = "base"

    @abstractmethod
    def compute(self, tickers: list[str], date: pd.Timestamp) -> SignalResult:
        """Compute the signal for each ticker as of *date*.

        Parameters
        ----------
        tickers:
            List of ticker symbols to evaluate.
        date:
            The "as-of" date — only data available on or before this date
            may be used (no look-ahead).

        Returns
        -------
        SignalResult
        """
