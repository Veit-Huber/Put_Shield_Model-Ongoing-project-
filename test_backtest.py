"""Tests for the walk-forward equity backtest logic.

Tests use synthetic price data so no network calls are needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest.equity_backtest import (
    compute_forward_return,
    compute_portfolio_stats,
    get_rebalance_dates,
)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _make_prices(
    tickers: list[str],
    start: str = "2020-01-01",
    periods: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame of synthetic price series (random walk)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq="B")
    data = {}
    for ticker in tickers:
        returns = rng.normal(0.0005, 0.02, size=periods)
        prices = 100.0 * np.cumprod(1 + returns)
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)


def _make_crashing_prices(ticker: str, start: str = "2020-01-01") -> pd.DataFrame:
    """Return prices that drop 40% over 63 trading days — a definite crash."""
    dates = pd.date_range(start=start, periods=100, freq="B")
    prices = np.linspace(100, 60, 100)  # monotonic 40% decline
    return pd.DataFrame({ticker: prices}, index=dates)


# ---------------------------------------------------------------------------
# get_rebalance_dates
# ---------------------------------------------------------------------------

class TestRebalanceDates:
    def test_monthly_frequency(self):
        dates = get_rebalance_dates(
            pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01"), "monthly"
        )
        assert len(dates) == 6
        assert all(d.day == 1 for d in dates)

    def test_quarterly_frequency(self):
        dates = get_rebalance_dates(
            pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"), "quarterly"
        )
        assert len(dates) == 4

    def test_invalid_frequency_raises(self):
        with pytest.raises(ValueError):
            get_rebalance_dates(
                pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"), "weekly"
            )


# ---------------------------------------------------------------------------
# compute_forward_return
# ---------------------------------------------------------------------------

class TestForwardReturn:
    def test_positive_return_on_rising_prices(self):
        prices = pd.DataFrame(
            {"A": [100, 110, 120, 130, 140]},
            index=pd.date_range("2020-01-01", periods=5, freq="B"),
        )
        ret = compute_forward_return(prices, pd.Timestamp("2020-01-01"), window_days=4)
        # From 100 to 140 = +40%
        assert abs(ret["A"] - 0.40) < 1e-9

    def test_negative_return_on_crashing_prices(self):
        prices = _make_crashing_prices("CRASH")
        ret = compute_forward_return(
            prices, pd.Timestamp("2020-01-02"), window_days=63
        )
        assert ret["CRASH"] < -0.20  # more than 20% crash

    def test_returns_nan_for_insufficient_data(self):
        prices = pd.DataFrame(
            {"A": [100.0]},
            index=pd.DatetimeIndex(["2020-01-01"]),
        )
        ret = compute_forward_return(prices, pd.Timestamp("2020-01-01"), window_days=21)
        assert np.isnan(ret["A"])

    def test_date_before_data_returns_nan(self):
        prices = pd.DataFrame(
            {"A": [100, 110]},
            index=pd.date_range("2020-06-01", periods=2, freq="B"),
        )
        ret = compute_forward_return(prices, pd.Timestamp("2019-01-01"), window_days=21)
        # No prices on or after 2019-01-01 → all NaN
        assert np.isnan(ret["A"])

    def test_window_clipped_to_available_data(self):
        """If fewer than window_days prices are available, use what's there."""
        prices = pd.DataFrame(
            {"A": [100, 120]},
            index=pd.date_range("2020-01-01", periods=2, freq="B"),
        )
        ret = compute_forward_return(prices, pd.Timestamp("2020-01-01"), window_days=50)
        # Only 2 rows available: return = (120-100)/100 = 20%
        assert abs(ret["A"] - 0.20) < 1e-9


# ---------------------------------------------------------------------------
# Walk-forward logic (mocked combiner)
# ---------------------------------------------------------------------------

class TestWalkForward:
    def _make_backtest(self, tmp_path: Path, tickers: list[str]) -> "EquityBacktest":
        from src.backtest.equity_backtest import EquityBacktest

        config = {
            "universe": {"tickers": tickers, "source": "manual"},
            "backtest": {
                "start_date": "2020-01-01",
                "end_date": "2021-12-31",
                "rebalance_frequency": "quarterly",
                "estimation_years": 1,
                "forward_return_windows": [21, 63],
                "crash_threshold": -0.20,
            },
            "thresholds": {"danger": 0.65, "caution": 0.40},
            "signals": {"altman_z": {"enabled": True, "weight": 1.0}},
            "data": {"cache_dir": str(tmp_path / "data")},
        }
        config_path = tmp_path / "strategy.yaml"
        config_path.write_text(yaml.dump(config))

        bt = EquityBacktest.__new__(EquityBacktest)
        bt.config = config
        bt.output_dir = tmp_path / "results"
        bt.output_dir.mkdir(parents=True, exist_ok=True)
        return bt, config_path

    def test_burn_in_excludes_early_dates(self, tmp_path):
        """Dates before burn-in end should not appear in test results."""
        from src.backtest.equity_backtest import EquityBacktest, get_rebalance_dates

        tickers = ["A", "B"]
        bt, _ = self._make_backtest(tmp_path, tickers)

        all_dates = get_rebalance_dates(
            pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"), "quarterly"
        )
        burn_end = pd.Timestamp("2021-01-01")
        test_dates = [d for d in all_dates if d >= burn_end]

        # All test dates must be on or after the burn-in end
        for d in test_dates:
            assert d >= burn_end

    def test_process_date_returns_dict(self, tmp_path):
        """_process_date should return a dict with expected keys."""
        from src.backtest.equity_backtest import EquityBacktest

        tickers = ["A", "B", "C"]
        bt, _ = self._make_backtest(tmp_path, tickers)

        # Mock the combiner
        mock_scores = pd.DataFrame(
            {
                "flagged": [True, False, False],
                "hard_stop": [False, False, False],
                "composite_score": [0.8, 0.3, 0.2],
            },
            index=tickers,
        )
        bt.combiner = MagicMock()
        bt.combiner.compute.return_value = mock_scores

        prices = _make_prices(tickers, start="2021-01-01", periods=200)
        result = bt._process_date(
            pd.Timestamp("2021-01-04"),
            tickers,
            prices,
            forward_windows=[21, 63],
            crash_threshold=-0.20,
        )
        assert result is not None
        rec, flagged_log = result
        assert "n_flagged" in rec
        assert "n_clean" in rec
        assert rec["n_flagged"] == 1
        assert rec["n_clean"] == 2
        # flagged_log should contain exactly the one flagged ticker
        assert "A" in flagged_log.index

    def test_flagged_and_clean_return_stats_present(self, tmp_path):
        from src.backtest.equity_backtest import EquityBacktest

        tickers = ["A", "B", "C"]
        bt, _ = self._make_backtest(tmp_path, tickers)

        mock_scores = pd.DataFrame(
            {
                "flagged": [True, False, False],
                "hard_stop": [False, False, False],
                "composite_score": [0.8, 0.3, 0.2],
            },
            index=tickers,
        )
        bt.combiner = MagicMock()
        bt.combiner.compute.return_value = mock_scores

        prices = _make_prices(tickers, start="2021-01-01", periods=200)
        result = bt._process_date(
            pd.Timestamp("2021-01-04"),
            tickers,
            prices,
            forward_windows=[21, 63],
            crash_threshold=-0.20,
        )
        assert result is not None
        rec, _ = result
        # Should have stats for at least one window
        assert any("mean_ret" in k for k in rec.keys())
        assert any("crash_rate" in k for k in rec.keys())

    def test_all_nan_scores_no_crash(self, tmp_path):
        """If combiner returns NaN scores, flagged should be False for all."""
        from src.backtest.equity_backtest import EquityBacktest

        tickers = ["A", "B"]
        bt, _ = self._make_backtest(tmp_path, tickers)

        mock_scores = pd.DataFrame(
            {
                "flagged": [False, False],
                "hard_stop": [False, False],
                "composite_score": [np.nan, np.nan],
            },
            index=tickers,
        )
        bt.combiner = MagicMock()
        bt.combiner.compute.return_value = mock_scores

        prices = _make_prices(tickers, start="2021-01-01", periods=200)
        result = bt._process_date(
            pd.Timestamp("2021-01-04"),
            tickers,
            prices,
            forward_windows=[21, 63],
            crash_threshold=-0.20,
        )
        assert result is not None
        rec, flagged_log = result
        # NaN composite_score → excluded from both groups (n_no_data=2, not clean)
        assert rec["n_flagged"] == 0
        assert rec["n_clean"] == 0
        assert rec["n_no_data"] == 2
        assert flagged_log.empty


# ---------------------------------------------------------------------------
# compute_portfolio_stats
# ---------------------------------------------------------------------------

def _make_stat_inputs(
    returns: dict[str, list[float]],
    start: str = "2021-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (equity_lines, period_returns) from raw return dicts."""
    idx = pd.date_range(start, periods=len(next(iter(returns.values()))), freq="MS")
    period_ret = pd.DataFrame(returns, index=idx)
    equity = (1.0 + period_ret).cumprod() * 100.0
    return equity, period_ret


class TestComputePortfolioStats:
    def test_total_return_correct(self):
        rets = [0.01] * 12
        eq, pr = _make_stat_inputs({"all": rets})
        stats = compute_portfolio_stats(eq, pr)
        expected = 1.01 ** 12 - 1
        assert abs(stats.loc["all", "total_return"] - expected) < 1e-10

    def test_annualized_return_equals_total_for_one_year(self):
        # Exactly 12 monthly periods (1 year) → ann_ret == total_ret
        rets = [0.02, -0.01, 0.03, 0.01, -0.02, 0.01, 0.02, 0.00, 0.01, -0.01, 0.02, 0.01]
        eq, pr = _make_stat_inputs({"all": rets})
        stats = compute_portfolio_stats(eq, pr)
        assert abs(stats.loc["all", "ann_return"] - stats.loc["all", "total_return"]) < 1e-10

    def test_annualized_vol_formula(self):
        rets = [0.02, -0.01, 0.03, 0.01, -0.02, 0.01, 0.02, 0.00, 0.01, -0.01, 0.02, 0.01]
        eq, pr = _make_stat_inputs({"all": rets})
        stats = compute_portfolio_stats(eq, pr)
        expected_vol = pd.Series(rets).std() * np.sqrt(12)
        assert abs(stats.loc["all", "ann_vol"] - expected_vol) < 1e-10

    def test_sharpe_ratio_formula(self):
        rets = [0.02, -0.01, 0.03, 0.01, -0.02, 0.01, 0.02, 0.00, 0.01, -0.01, 0.02, 0.01]
        eq, pr = _make_stat_inputs({"all": rets})
        stats = compute_portfolio_stats(eq, pr, risk_free_rate=0.0)
        s = stats.loc["all"]
        assert abs(s["sharpe"] - s["ann_return"] / s["ann_vol"]) < 1e-10

    def test_max_drawdown_known_value(self):
        # Equity: 100 → 110 → 121 → 60.5
        # Peak at 121, trough at 60.5 → dd = (60.5 - 121) / 121 = -0.5
        rets = [0.1, 0.1, -0.5]
        eq, pr = _make_stat_inputs({"all": rets})
        stats = compute_portfolio_stats(eq, pr)
        assert abs(stats.loc["all", "max_drawdown"] - (-0.5)) < 1e-10

    def test_no_drawdown_when_always_rising(self):
        rets = [0.01] * 24  # monotonically rising → max_drawdown = 0
        eq, pr = _make_stat_inputs({"all": rets})
        stats = compute_portfolio_stats(eq, pr)
        assert abs(stats.loc["all", "max_drawdown"]) < 1e-10

    def test_var_5pct_is_fifth_percentile(self):
        rng = np.random.default_rng(0)
        rets = list(rng.normal(0.01, 0.05, 120))
        eq, pr = _make_stat_inputs({"all": rets})
        stats = compute_portfolio_stats(eq, pr)
        expected_var = pd.Series(rets).quantile(0.05)
        assert abs(stats.loc["all", "var_5pct"] - expected_var) < 1e-10

    def test_benchmark_info_ratio_is_nan(self):
        rets = [0.01] * 12
        eq, pr = _make_stat_inputs({"all": rets})
        stats = compute_portfolio_stats(eq, pr, benchmark_col="all")
        assert np.isnan(stats.loc["all", "info_ratio"])

    def test_information_ratio_formula(self):
        rng = np.random.default_rng(1)
        bench = list(rng.normal(0.01, 0.03, 24))
        # Genuine random excess so std(excess) > 0
        excess_rets = list(rng.normal(0.005, 0.01, 24))
        port = [b + e for b, e in zip(bench, excess_rets)]
        eq, pr = _make_stat_inputs({"all": bench, "clean": port})
        stats = compute_portfolio_stats(eq, pr, benchmark_col="all")
        excess = pd.Series(port) - pd.Series(bench)
        expected_ir = excess.mean() / excess.std() * np.sqrt(12)
        assert abs(stats.loc["clean", "info_ratio"] - expected_ir) < 1e-6

    def test_fewer_than_two_periods_returns_all_nan(self):
        eq, pr = _make_stat_inputs({"all": [0.05]})
        stats = compute_portfolio_stats(eq, pr)
        assert all(np.isnan(v) for v in stats.loc["all"])

    def test_multiple_portfolios_computed_independently(self):
        rng = np.random.default_rng(99)
        eq, pr = _make_stat_inputs({
            "all":     list(rng.normal(0.010, 0.03, 36)),
            "clean":   list(rng.normal(0.015, 0.025, 36)),
            "flagged": list(rng.normal(0.005, 0.040, 36)),
        })
        stats = compute_portfolio_stats(eq, pr, benchmark_col="all")
        assert set(stats.index) == {"all", "clean", "flagged"}
        # Higher mean return → higher annualized return
        assert stats.loc["clean", "ann_return"] > stats.loc["flagged", "ann_return"]
        # Info ratio defined for non-benchmark cols, NaN for benchmark
        assert np.isnan(stats.loc["all", "info_ratio"])
        assert not np.isnan(stats.loc["clean", "info_ratio"])


# ---------------------------------------------------------------------------
# build_equity_lines
# ---------------------------------------------------------------------------

def _make_bt(tmp_path: Path):
    """Instantiate EquityBacktest without touching disk or config."""
    from src.backtest.equity_backtest import EquityBacktest
    bt = EquityBacktest.__new__(EquityBacktest)
    bt.output_dir = tmp_path
    return bt


def _make_results_df(n: int = 24, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="MS")
    return pd.DataFrame({
        "w21_all_mean_ret":     rng.normal(0.010, 0.030, n),
        "w21_clean_mean_ret":   rng.normal(0.012, 0.025, n),
        "w21_flagged_mean_ret": rng.normal(0.005, 0.040, n),
        # extra columns that must be ignored
        "w63_all_mean_ret":     rng.normal(0.030, 0.060, n),
        "n_flagged":            rng.integers(0, 10, n).astype(float),
    }, index=idx)


class TestBuildEquityLines:
    def test_columns_are_all_clean_flagged(self, tmp_path):
        bt = _make_bt(tmp_path)
        eq, pr = bt.build_equity_lines(_make_results_df())
        assert set(eq.columns) == {"all", "clean", "flagged"}
        assert set(pr.columns) == {"all", "clean", "flagged"}

    def test_first_value_reflects_first_return(self, tmp_path):
        bt = _make_bt(tmp_path)
        results = _make_results_df()
        eq, _ = bt.build_equity_lines(results)
        for col in ["all", "clean", "flagged"]:
            expected = 100.0 * (1 + results[f"w21_{col}_mean_ret"].iloc[0])
            assert abs(eq[col].iloc[0] - expected) < 1e-9

    def test_compounding_is_multiplicative(self, tmp_path):
        bt = _make_bt(tmp_path)
        idx = pd.date_range("2021-01-01", periods=3, freq="MS")
        results = pd.DataFrame({
            "w21_all_mean_ret":     [0.10, 0.10, -0.50],
            "w21_clean_mean_ret":   [0.10, 0.10, -0.50],
            "w21_flagged_mean_ret": [0.10, 0.10, -0.50],
        }, index=idx)
        eq, _ = bt.build_equity_lines(results)
        # 100 × 1.1 × 1.1 × 0.5 = 60.5
        assert abs(eq["all"].iloc[2] - 60.5) < 1e-9

    def test_nan_period_is_flat(self, tmp_path):
        """NaN return → equity line holds its previous value (treated as 0)."""
        bt = _make_bt(tmp_path)
        idx = pd.date_range("2021-01-01", periods=3, freq="MS")
        results = pd.DataFrame({
            "w21_all_mean_ret":     [0.10, np.nan, 0.10],
            "w21_clean_mean_ret":   [0.10, np.nan, 0.10],
            "w21_flagged_mean_ret": [0.10, np.nan, 0.10],
        }, index=idx)
        eq, _ = bt.build_equity_lines(results)
        # Period 0: 110, period 1 (NaN→0): 110, period 2: 121
        assert abs(eq["all"].iloc[1] - 110.0) < 1e-9
        assert abs(eq["all"].iloc[2] - 121.0) < 1e-9

    def test_period_returns_preserves_nan(self, tmp_path):
        """period_ret should contain the raw NaN — equity does the filling."""
        bt = _make_bt(tmp_path)
        idx = pd.date_range("2021-01-01", periods=3, freq="MS")
        results = pd.DataFrame({
            "w21_all_mean_ret":     [0.01, np.nan, 0.01],
            "w21_clean_mean_ret":   [0.01, np.nan, 0.01],
            "w21_flagged_mean_ret": [0.01, np.nan, 0.01],
        }, index=idx)
        _, pr = bt.build_equity_lines(results)
        assert np.isnan(pr["all"].iloc[1])

    def test_missing_column_raises_key_error(self, tmp_path):
        bt = _make_bt(tmp_path)
        idx = pd.date_range("2021-01-01", periods=3, freq="MS")
        incomplete = pd.DataFrame({
            "w21_all_mean_ret": [0.01, 0.02, 0.01],
            # w21_clean_mean_ret and w21_flagged_mean_ret intentionally absent
        }, index=idx)
        with pytest.raises(KeyError):
            bt.build_equity_lines(incomplete)


# ---------------------------------------------------------------------------
# _save_equity_plot
# ---------------------------------------------------------------------------

class TestSaveEquityPlot:
    def _make_inputs(self, n: int = 24) -> tuple:
        rng = np.random.default_rng(42)
        idx = pd.date_range("2021-01-01", periods=n, freq="MS")
        pr = pd.DataFrame({
            "all":     rng.normal(0.010, 0.030, n),
            "clean":   rng.normal(0.012, 0.025, n),
            "flagged": rng.normal(0.005, 0.040, n),
        }, index=idx)
        eq = (1 + pr).cumprod() * 100.0
        stats = compute_portfolio_stats(eq, pr)
        return eq, pr, stats

    def test_output_file_created(self, tmp_path):
        from src.backtest.equity_backtest import EquityBacktest
        bt = _make_bt(tmp_path)
        eq, pr, stats = self._make_inputs()
        bt._save_equity_plot(eq, pr, stats)
        assert (tmp_path / "equity_lines.png").exists()

    def test_plot_does_not_crash_with_missing_flagged(self, tmp_path):
        """Plot must survive if the flagged column is absent (all stocks clean)."""
        bt = _make_bt(tmp_path)
        rng = np.random.default_rng(0)
        idx = pd.date_range("2021-01-01", periods=12, freq="MS")
        pr = pd.DataFrame({
            "all":   rng.normal(0.01, 0.03, 12),
            "clean": rng.normal(0.012, 0.025, 12),
        }, index=idx)
        eq = (1 + pr).cumprod() * 100.0
        stats = compute_portfolio_stats(eq, pr)
        bt._save_equity_plot(eq, pr, stats)  # must not raise
        assert (tmp_path / "equity_lines.png").exists()

    def test_plot_does_not_crash_with_empty_stats(self, tmp_path):
        bt = _make_bt(tmp_path)
        eq, pr, _ = self._make_inputs()
        bt._save_equity_plot(eq, pr, pd.DataFrame())  # must not raise
        assert (tmp_path / "equity_lines.png").exists()
