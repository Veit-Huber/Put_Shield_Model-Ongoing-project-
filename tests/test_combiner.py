"""Tests for SignalCombiner aggregation logic."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml

from src.signals.base import BaseSignal, SignalResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_signal_result(
    tickers: list[str],
    scores: list[float],
    flags: list[bool],
    name: str = "mock_signal",
) -> SignalResult:
    idx = pd.Index(tickers)
    return SignalResult(
        scores=pd.Series(scores, index=idx),
        raw_values=pd.DataFrame({"raw": scores}, index=idx),
        flags=pd.Series(flags, index=idx),
        signal_name=name,
        as_of_date=pd.Timestamp("2021-01-01"),
    )


class MockSignal(BaseSignal):
    """A signal that returns pre-canned results."""

    name = "mock_signal"

    def __init__(self, result: SignalResult, **kwargs):
        self._result = result

    def compute(self, tickers, date):
        return self._result


def _write_temp_config(tmp_path: Path, signals_cfg: dict) -> Path:
    config = {
        "universe": {"tickers": ["A", "B", "C"]},
        "backtest": {"start_date": "2018-01-01", "end_date": "2023-12-31"},
        "thresholds": {"danger": 0.65, "caution": 0.40},
        "signals": signals_cfg,
        "data": {"cache_dir": str(tmp_path / "data")},
    }
    path = tmp_path / "strategy.yaml"
    path.write_text(yaml.dump(config))
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCombinerWeighting:
    def test_single_signal_composite_equals_signal_score(self, tmp_path):
        config_path = _write_temp_config(
            tmp_path,
            {"altman_z": {"enabled": True, "weight": 1.0}},
        )
        canned = _make_signal_result(
            ["A", "B", "C"], [0.2, 0.7, 0.9], [False, False, True]
        )
        with patch("src.aggregator.combiner.SIGNAL_REGISTRY", {"altman_z": lambda **_: MockSignal(canned)}):
            from src.aggregator.combiner import SignalCombiner
            combiner = SignalCombiner.__new__(SignalCombiner)
            combiner.config = yaml.safe_load(config_path.read_text())
            combiner.cache_dir = tmp_path / "data"
            combiner.signals = {"altman_z": (MockSignal(canned), 1.0)}

        result = combiner.compute(["A", "B", "C"], pd.Timestamp("2021-01-01"))

        assert abs(result.loc["A", "composite_score"] - 0.2) < 1e-9
        assert abs(result.loc["B", "composite_score"] - 0.7) < 1e-9
        assert abs(result.loc["C", "composite_score"] - 0.9) < 1e-9

    def test_hard_stop_propagates_to_flagged(self, tmp_path):
        config_path = _write_temp_config(
            tmp_path,
            {"altman_z": {"enabled": True, "weight": 1.0}},
        )
        # C has low composite score but hard_stop=True
        canned = _make_signal_result(
            ["A", "B", "C"], [0.2, 0.3, 0.1], [False, False, True]
        )
        config = yaml.safe_load(config_path.read_text())
        from src.aggregator.combiner import SignalCombiner
        combiner = SignalCombiner.__new__(SignalCombiner)
        combiner.config = config
        combiner.cache_dir = tmp_path / "data"
        combiner.signals = {"altman_z": (MockSignal(canned), 1.0)}

        result = combiner.compute(["A", "B", "C"], pd.Timestamp("2021-01-01"))

        assert not result.loc["A", "hard_stop"]
        assert not result.loc["B", "hard_stop"]
        assert result.loc["C", "hard_stop"]
        assert result.loc["C", "flagged"]

    def test_high_composite_score_sets_flagged(self, tmp_path):
        canned = _make_signal_result(
            ["A", "B"], [0.8, 0.3], [False, False]
        )
        config = {
            "thresholds": {"danger": 0.65},
            "signals": {},
            "data": {"cache_dir": str(tmp_path)},
        }
        from src.aggregator.combiner import SignalCombiner
        combiner = SignalCombiner.__new__(SignalCombiner)
        combiner.config = config
        combiner.cache_dir = tmp_path
        combiner.signals = {"altman_z": (MockSignal(canned), 1.0)}

        result = combiner.compute(["A", "B"], pd.Timestamp("2021-01-01"))
        assert result.loc["A", "flagged"]
        assert not result.loc["B", "flagged"]

    def test_nan_score_excluded_from_weighted_average(self, tmp_path):
        """If one signal returns NaN, it should be excluded from the average."""
        canned1 = _make_signal_result(["A"], [0.6], [False], name="sig1")
        canned2 = _make_signal_result(["A"], [np.nan], [False], name="sig2")

        mock_sig1 = MockSignal(canned1)
        mock_sig1.name = "sig1"
        mock_sig2 = MockSignal(canned2)
        mock_sig2.name = "sig2"

        config = {
            "thresholds": {"danger": 0.65},
            "signals": {},
            "data": {"cache_dir": str(tmp_path)},
        }
        from src.aggregator.combiner import SignalCombiner
        combiner = SignalCombiner.__new__(SignalCombiner)
        combiner.config = config
        combiner.cache_dir = tmp_path
        combiner.signals = {
            "sig1": (mock_sig1, 1.0),
            "sig2": (mock_sig2, 1.0),
        }

        result = combiner.compute(["A"], pd.Timestamp("2021-01-01"))
        # Composite should equal sig1's score (0.6) since sig2 is NaN
        assert abs(result.loc["A", "composite_score"] - 0.6) < 1e-9

    def test_two_signals_weighted_average(self, tmp_path):
        """Composite should be weight-normalized average of two signals."""
        canned1 = _make_signal_result(["A"], [0.4], [False], name="sig1")
        canned2 = _make_signal_result(["A"], [0.8], [False], name="sig2")

        config = {
            "thresholds": {"danger": 0.65},
            "signals": {},
            "data": {"cache_dir": str(tmp_path)},
        }
        from src.aggregator.combiner import SignalCombiner
        combiner = SignalCombiner.__new__(SignalCombiner)
        combiner.config = config
        combiner.cache_dir = tmp_path
        # weights: sig1=1.0, sig2=2.0 → weighted mean = (0.4*1 + 0.8*2) / 3 = 2.0/3 ≈ 0.667
        combiner.signals = {
            "sig1": (MockSignal(canned1), 1.0),
            "sig2": (MockSignal(canned2), 2.0),
        }

        result = combiner.compute(["A"], pd.Timestamp("2021-01-01"))
        expected = (0.4 * 1.0 + 0.8 * 2.0) / (1.0 + 2.0)
        assert abs(result.loc["A", "composite_score"] - expected) < 1e-9


class TestCombinerPanelCompute:
    def test_panel_multi_index(self, tmp_path):
        canned = _make_signal_result(["A", "B"], [0.3, 0.7], [False, False])
        config = {
            "thresholds": {"danger": 0.65},
            "signals": {},
            "data": {"cache_dir": str(tmp_path)},
        }
        from src.aggregator.combiner import SignalCombiner
        combiner = SignalCombiner.__new__(SignalCombiner)
        combiner.config = config
        combiner.cache_dir = tmp_path
        combiner.signals = {"altman_z": (MockSignal(canned), 1.0)}

        dates = [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-02-01")]
        panel = combiner.compute_panel(["A", "B"], dates)

        assert panel.index.names == ["date", "ticker"]
        assert len(panel) == 4  # 2 dates × 2 tickers
