"""Signal aggregator — combines multiple signals into a composite danger score.

Reads configs/strategy.yaml to determine which signals are enabled and their
weights.  Adding a new signal only requires:
  1. Implementing it in src/signals/
  2. Adding an entry in strategy.yaml
  3. Registering it in SIGNAL_REGISTRY below.

Hard stops from ANY signal override the composite score.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.signals.altman_z import AltmanZSignal
from src.signals.base import BaseSignal, SignalResult
from src.signals.beneish_m import BeneishMSignal

logger = logging.getLogger(__name__)

# Registry mapping yaml key → signal class
# Add new signals here as they are implemented.
SIGNAL_REGISTRY: dict[str, type[BaseSignal]] = {
    "altman_z":  AltmanZSignal,
    "beneish_m": BeneishMSignal,
    # "piotroski_f": PiotroskiFSignal, # stage 2
    # "piotroski_f": PiotroskiFSignal, # stage 2
    # "iv_skew": IVSkewSignal,         # stage 3
    # "short_interest": ShortInterestSignal,  # stage 3
}


def load_config(config_path: str | Path = "configs/strategy.yaml") -> dict[str, Any]:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


class SignalCombiner:
    """Instantiates enabled signals and produces composite danger scores.

    Parameters
    ----------
    config_path:
        Path to strategy.yaml.
    cache_dir:
        Directory for data caching; passed through to each signal.
    """

    def __init__(
        self,
        config_path: str | Path = "configs/strategy.yaml",
        cache_dir: str | Path = "data/raw",
    ) -> None:
        self.config = load_config(config_path)
        self.cache_dir = Path(cache_dir)
        self.signals: dict[str, tuple[BaseSignal, float]] = {}  # name → (signal, weight)
        self._load_signals()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_signals(self) -> None:
        signals_cfg = self.config.get("signals", {})
        for name, cls in SIGNAL_REGISTRY.items():
            cfg = signals_cfg.get(name, {})
            if not cfg.get("enabled", False):
                logger.info("Signal '%s' is disabled — skipping.", name)
                continue
            weight = float(cfg.get("weight", 1.0))
            # Pass signal-specific config as kwargs where the signal accepts them
            kwargs = self._signal_kwargs(name, cfg)
            instance = cls(cache_dir=self.cache_dir, **kwargs)
            self.signals[name] = (instance, weight)
            logger.info("Loaded signal '%s' (weight=%.2f)", name, weight)

    def _signal_kwargs(self, name: str, cfg: dict) -> dict:
        """Extract constructor kwargs from yaml config for a specific signal."""
        kwargs: dict[str, Any] = {}
        if "norm_low" in cfg:
            kwargs["norm_low"] = cfg["norm_low"]
        if "norm_high" in cfg:
            kwargs["norm_high"] = cfg["norm_high"]
        if name == "altman_z" and "hard_stop_z" in cfg:
            kwargs["hard_stop_z"] = cfg["hard_stop_z"]
        if name == "beneish_m" and "hard_stop_m" in cfg:
            kwargs["hard_stop_m"] = cfg["hard_stop_m"]
        return kwargs

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def compute(
        self, tickers: list[str], date: pd.Timestamp
    ) -> pd.DataFrame:
        """Compute composite danger scores for *tickers* as of *date*.

        Returns
        -------
        pd.DataFrame
            One row per ticker, columns:
              - composite_score   : weighted average of enabled signal scores
                                    (NaN signals excluded from average)
              - hard_stop         : True if ANY signal raises a hard stop
              - flagged           : True if composite_score > danger_threshold
                                    OR hard_stop is True
              - <signal>_score    : individual signal scores
              - <signal>_hard_stop: individual signal flags
        """
        if not self.signals:
            raise RuntimeError("No signals are enabled. Check strategy.yaml.")

        danger_threshold = self.config.get("thresholds", {}).get("danger", 0.65)

        results: dict[str, SignalResult] = {}
        for name, (signal, _) in self.signals.items():
            try:
                results[name] = signal.compute(tickers, date)
            except Exception as exc:
                logger.error("Signal '%s' failed for %s: %s", name, date, exc)

        # Assemble output DataFrame
        out = pd.DataFrame(index=tickers)

        weighted_scores: list[pd.Series] = []
        weights: list[float] = []

        for name, result in results.items():
            _, weight = self.signals[name]
            out[f"{name}_score"] = result.scores
            out[f"{name}_hard_stop"] = result.flags
            weighted_scores.append(result.scores)  # store raw scores; weights applied below
            weights.append(weight)

        # Composite = weighted mean, ignoring NaN
        if weighted_scores:
            score_df = pd.concat(weighted_scores, axis=1)
            # For each row, sum non-NaN (score * weight) / sum of corresponding weights
            def _weighted_mean(row: pd.Series) -> float:
                valid_mask = ~row.isna()
                if not valid_mask.any():
                    return np.nan
                valid_weights = np.array(weights)[valid_mask.values]
                if valid_weights.sum() == 0:
                    return np.nan
                return (row[valid_mask].values * valid_weights).sum() / valid_weights.sum()

            out["composite_score"] = score_df.apply(_weighted_mean, axis=1)
        else:
            out["composite_score"] = np.nan

        # Hard stop: True if ANY signal flagged the ticker
        hard_stop_cols = [c for c in out.columns if c.endswith("_hard_stop")]
        if hard_stop_cols:
            out["hard_stop"] = out[hard_stop_cols].any(axis=1)
        else:
            out["hard_stop"] = False

        out["flagged"] = (out["composite_score"] > danger_threshold) | out["hard_stop"]
        out["as_of_date"] = date

        return out

    def compute_panel(
        self, tickers: list[str], dates: list[pd.Timestamp]
    ) -> pd.DataFrame:
        """Run compute() for multiple dates and stack results.

        Returns a DataFrame with a MultiIndex (date, ticker).
        """
        frames = []
        for date in dates:
            df = self.compute(tickers, date)
            df["date"] = date
            df.index.name = "ticker"
            frames.append(df.reset_index().set_index(["date", "ticker"]))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames)
