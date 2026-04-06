"""Walk-forward equity backtest for put-shield signals.

For each rebalance date:
  1. Compute composite danger scores for all tickers.
  2. Split universe into "flagged" vs "clean" groups.
  3. Measure forward 1-month and 3-month returns.

Walk-forward protocol:
  - First ``estimation_years`` of data = burn-in (signal calibration).
  - Then test one year at a time, rolling forward.

Reports:
  - Crash rate per group (% stocks with > 20% drawdown in forward window)
  - Mean / median / 5th-percentile return per group
  - Signal precision and recall for crash prediction
  - Matplotlib summary plot saved to results/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.aggregator.combiner import SignalCombiner
from src.data.universe import get_full_universe, get_sp500
from src.data.yahoo import YahooDataSource

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "configs/strategy.yaml") -> dict[str, Any]:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def get_rebalance_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str = "monthly",
) -> list[pd.Timestamp]:
    """Return a list of rebalance dates between *start* and *end*."""
    if frequency == "monthly":
        freq = "MS"
    elif frequency == "quarterly":
        freq = "QS"
    else:
        raise ValueError(f"Unknown rebalance_frequency: {frequency!r}")
    return list(pd.date_range(start=start, end=end, freq=freq))


def fetch_price_panel(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices indexed by date, columns = tickers."""
    ds = YahooDataSource(cache_dir=cache_dir)
    frames: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            prices = ds.get_price_history(ticker, start=start, end=end)
            if prices is not None and not prices.empty:
                # Newer yfinance returns MultiIndex columns even for single tickers.
                # prices["Close"] may be a DataFrame instead of a Series — squeeze it.
                if "Close" in prices.columns:
                    close = prices["Close"]
                    if isinstance(close, pd.DataFrame):
                        close = close.squeeze()
                    frames[ticker] = close
                else:
                    # MultiIndex: find ("Close", ticker) or ("Close", ticker.upper())
                    for key in [("Close", ticker), ("Close", ticker.upper())]:
                        if key in prices.columns:
                            frames[ticker] = prices[key]
                            break
        except Exception as exc:
            logger.warning("Failed to fetch prices for %s: %s", ticker, exc)
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames)


def compute_forward_return(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    window_days: int,
    max_entry_lag_days: int = 5,
) -> pd.Series:
    """Return the forward return for each ticker from *date* over *window_days* trading days.

    Returns NaN for all tickers if the first available price is more than
    *max_entry_lag_days* business days after *date* — the rebalance date has
    no price nearby, so we cannot enter the trade.
    """
    future_prices = prices[prices.index >= date]
    if len(future_prices) < 2:
        return pd.Series(np.nan, index=prices.columns)

    # Guard: if the nearest available price is far in the future, we have no
    # entry price for this rebalance date.
    first_available = future_prices.index[0]
    lag_bdays = len(pd.bdate_range(date, first_available)) - 1
    if lag_bdays > max_entry_lag_days:
        return pd.Series(np.nan, index=prices.columns)

    start_price = future_prices.iloc[0]
    end_idx = min(window_days, len(future_prices) - 1)
    end_price = future_prices.iloc[end_idx]

    return (end_price - start_price) / start_price


# ---------------------------------------------------------------------------
# Portfolio analytics (module-level — pure functions)
# ---------------------------------------------------------------------------


def compute_portfolio_stats(
    equity_lines: pd.DataFrame,
    period_returns: pd.DataFrame,
    benchmark_col: str = "all",
    periods_per_year: int = 12,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Compute annualized portfolio statistics for each equity line.

    Parameters
    ----------
    equity_lines:
        Compounded portfolio value starting at 100, columns = portfolio names.
    period_returns:
        Raw (unfilled) period returns — same columns and index.
    benchmark_col:
        Column used as the benchmark when computing the information ratio.
    periods_per_year:
        Rebalance periods per year (12 for monthly).
    risk_free_rate:
        Annualized risk-free rate for the Sharpe ratio.

    Returns
    -------
    pd.DataFrame indexed by portfolio name with columns:
        total_return, ann_return, ann_vol, sharpe,
        max_drawdown, var_5pct, info_ratio.
    """
    sqrt_ppy = np.sqrt(periods_per_year)
    _keys = ["total_return", "ann_return", "ann_vol", "sharpe",
             "max_drawdown", "var_5pct", "info_ratio"]
    records: dict[str, dict[str, float]] = {}

    for col in equity_lines.columns:
        eq = equity_lines[col].dropna()
        ret = period_returns[col].dropna()

        if eq.empty or len(ret) < 2:
            records[col] = {k: np.nan for k in _keys}
            continue

        n = len(ret)
        total_ret  = float(eq.iloc[-1] / 100.0 - 1.0)
        ann_ret    = float((1.0 + total_ret) ** (periods_per_year / n) - 1.0)
        ann_vol    = float(ret.std() * sqrt_ppy)
        sharpe     = float((ann_ret - risk_free_rate) / ann_vol) if ann_vol > 0 else np.nan

        rolling_max = eq.cummax()
        max_dd      = float(((eq - rolling_max) / rolling_max).min())

        var_5 = float(ret.quantile(0.05))

        if col != benchmark_col and benchmark_col in period_returns.columns:
            bench  = period_returns[benchmark_col].reindex(ret.index).fillna(0.0)
            excess = ret - bench
            ir     = float(excess.mean() / excess.std() * sqrt_ppy) if excess.std() > 0 else np.nan
        else:
            ir = np.nan  # benchmark vs itself is undefined

        records[col] = {
            "total_return": total_ret,
            "ann_return":   ann_ret,
            "ann_vol":      ann_vol,
            "sharpe":       sharpe,
            "max_drawdown": max_dd,
            "var_5pct":     var_5,
            "info_ratio":   ir,
        }

    return pd.DataFrame(records).T


class EquityBacktest:
    """Walk-forward equity backtest engine.

    Parameters
    ----------
    config_path:
        Path to strategy.yaml.
    output_dir:
        Where to save the summary plot.
    """

    def __init__(
        self,
        config_path: str | Path = "configs/strategy.yaml",
        output_dir: str | Path = "results",
    ) -> None:
        self.config = load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.combiner = SignalCombiner(config_path=config_path)

    # ------------------------------------------------------------------
    # Universe resolution
    # ------------------------------------------------------------------

    def _resolve_universe(self, cfg: dict) -> list[str]:
        """Return the ticker list based on the 'source' field in strategy.yaml."""
        source = cfg.get("universe", {}).get("source", "manual")
        cache_dir = cfg.get("data", {}).get("cache_dir", "data/raw")

        if source == "manual":
            return cfg["universe"]["tickers"]
        elif source == "sp500":
            return get_sp500(cache_path=f"{cache_dir}/sp500_tickers.csv")
        elif source == "full":
            return get_full_universe(cache_path=f"{cache_dir}/sp500_tickers.csv")
        else:
            logger.warning("Unknown universe source '%s', falling back to manual list.", source)
            return cfg["universe"]["tickers"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Execute the full walk-forward backtest.

        Returns a DataFrame of per-rebalance-date performance statistics.
        """
        cfg = self.config
        tickers = self._resolve_universe(cfg)
        start = cfg["backtest"]["start_date"]
        end = cfg["backtest"]["end_date"]
        freq = cfg["backtest"].get("rebalance_frequency", "monthly")
        estimation_years = cfg["backtest"].get("estimation_years", 3)
        forward_windows = cfg["backtest"].get("forward_return_windows", [21, 63])
        crash_threshold = cfg["backtest"].get("crash_threshold", -0.20)
        cache_dir = cfg.get("data", {}).get("cache_dir", "data/raw")

        logger.info("Fetching price data for %d tickers (%s → %s)", len(tickers), start, end)
        prices = fetch_price_panel(tickers, start=start, end=end, cache_dir=cache_dir)
        if prices.empty:
            raise RuntimeError("No price data retrieved. Check ticker list and network.")

        all_dates = get_rebalance_dates(
            pd.Timestamp(start), pd.Timestamp(end), frequency=freq
        )
        burn_in_end = pd.Timestamp(start) + pd.DateOffset(years=estimation_years)
        test_dates = [d for d in all_dates if d >= burn_in_end]

        logger.info(
            "Walk-forward: %d total dates, %d test dates (burn-in ends %s)",
            len(all_dates), len(test_dates), burn_in_end.date(),
        )

        records = []
        flagged_logs: list[pd.DataFrame] = []
        for date in test_dates:
            logger.info("Processing rebalance date %s", date.date())
            result = self._process_date(date, tickers, prices, forward_windows, crash_threshold)
            if result:
                rec, flagged_log = result
                records.append(rec)
                if not flagged_log.empty:
                    flagged_logs.append(flagged_log)

        results_df = pd.DataFrame(records).set_index("date") if records else pd.DataFrame()

        # Save per-date results CSV (n_flagged, n_clean, n_no_data, returns, etc.)
        if not results_df.empty:
            results_path = self.output_dir / "backtest_results.csv"
            results_df.to_csv(results_path)
            logger.info("Saved per-date results to %s", results_path)

        # Save flagged-stock log (one row per ticker per date)
        if flagged_logs:
            flagged_df = pd.concat(flagged_logs).reset_index()
            flagged_path = self.output_dir / "flagged_stocks.csv"
            flagged_df.to_csv(flagged_path, index=False)
            logger.info("Saved flagged stock log to %s", flagged_path)

        if not results_df.empty:
            self._save_plot(results_df, forward_windows)
            self._print_summary(results_df)

            # Equity-line analysis (21-day / monthly)
            try:
                equity_lines, period_returns = self.build_equity_lines(results_df)
                stats = compute_portfolio_stats(equity_lines, period_returns)
                self._save_equity_plot(equity_lines, period_returns, stats)
                self._print_portfolio_stats(stats)
                stats_path = self.output_dir / "portfolio_stats.csv"
                stats.to_csv(stats_path)
                logger.info("Saved portfolio stats to %s", stats_path)
            except Exception as exc:
                logger.error("Equity line analysis failed: %s", exc, exc_info=True)

        return results_df

    # ------------------------------------------------------------------
    # Per-date logic
    # ------------------------------------------------------------------

    def _process_date(
        self,
        date: pd.Timestamp,
        tickers: list[str],
        prices: pd.DataFrame,
        forward_windows: list[int],
        crash_threshold: float,
    ) -> tuple[dict, pd.DataFrame] | None:
        # Compute signals
        try:
            scores_df = self.combiner.compute(tickers, date)
        except Exception as exc:
            logger.error("Combiner failed for %s: %s", date, exc)
            return None

        # Tickers with no signal data at all (composite_score NaN) are not
        # genuinely "clean" — exclude them from both groups for transparency.
        has_data_mask = scores_df["composite_score"].notna()
        n_no_data = (~has_data_mask).sum()
        if n_no_data > 0:
            logger.debug("%s: %d tickers had no signal data (excluded from counts)", date.date(), n_no_data)

        flagged_mask = scores_df["flagged"].fillna(False) & has_data_mask
        clean_mask   = ~scores_df["flagged"].fillna(False) & has_data_mask
        flagged = set(scores_df.index[flagged_mask])
        clean   = set(scores_df.index[clean_mask])

        rec: dict[str, Any] = {
            "date": date,
            "n_flagged": len(flagged),
            "n_clean": len(clean),
        }

        for window in forward_windows:
            fwd_returns = compute_forward_return(prices, date, window)
            for group_name, group_tickers in [("flagged", flagged), ("clean", clean), ("all", set(tickers))]:
                group_returns = fwd_returns[
                    [t for t in group_tickers if t in fwd_returns.index]
                ].dropna()
                if group_returns.empty:
                    continue
                prefix = f"w{window}_{group_name}"
                rec[f"{prefix}_mean_ret"] = group_returns.mean()
                rec[f"{prefix}_median_ret"] = group_returns.median()
                rec[f"{prefix}_p5_ret"] = group_returns.quantile(0.05)
                rec[f"{prefix}_crash_rate"] = (group_returns < crash_threshold).mean()

        # Precision / recall for crash prediction (using the longer window)
        main_window = max(forward_windows)
        fwd_returns_main = compute_forward_return(prices, date, main_window)
        actual_crash = fwd_returns_main < crash_threshold
        for ticker in tickers:
            if ticker not in fwd_returns_main.index:
                actual_crash[ticker] = False

        valid = set(fwd_returns_main.dropna().index)
        tp = len(flagged & {t for t in valid if actual_crash.get(t, False)})
        fp = len(flagged & {t for t in valid if not actual_crash.get(t, False)})
        fn = len(clean & {t for t in valid if actual_crash.get(t, False)})

        rec["precision"] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec["recall"] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        rec["n_no_data"] = int((~has_data_mask).sum())

        logger.info(
            "  %s — flagged: %d  clean: %d  no_data: %d  (flagged: %s)",
            date.date(),
            len(flagged),
            len(clean),
            rec["n_no_data"],
            ", ".join(sorted(flagged)) if flagged else "none",
        )

        # Build per-ticker flagged log for this date
        flagged_log = scores_df.loc[flagged_mask].copy()
        flagged_log.index.name = "ticker"
        flagged_log.insert(0, "date", date)

        return rec, flagged_log

    # ------------------------------------------------------------------
    # Equity line construction
    # ------------------------------------------------------------------

    def build_equity_lines(
        self, results: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build compounded equity lines from monthly (21-day) group returns.

        Parameters
        ----------
        results:
            Output of ``run()`` — must contain ``w21_{all,clean,flagged}_mean_ret``.

        Returns
        -------
        equity_lines : pd.DataFrame
            Compounded value of $100 invested at inception.
            Columns: ['all', 'clean', 'flagged'].
        period_returns : pd.DataFrame
            Raw (unfilled) period returns — same columns and index.
            NaN where a group had no stocks or no price data that period.
        """
        col_map = {
            "all":     "w21_all_mean_ret",
            "clean":   "w21_clean_mean_ret",
            "flagged": "w21_flagged_mean_ret",
        }
        missing = [c for c in col_map.values() if c not in results.columns]
        if missing:
            raise KeyError(f"build_equity_lines: missing columns in results: {missing}")

        period_ret = pd.DataFrame(
            {name: results[col] for name, col in col_map.items()}
        )
        # NaN → 0 (flat): when a group has no eligible stocks that period,
        # treat it as holding cash so the equity line stays flat rather than
        # interpolating a stale return.
        equity = (1.0 + period_ret.fillna(0.0)).cumprod() * 100.0
        return equity, period_ret

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _save_plot(self, results: pd.DataFrame, forward_windows: list[int]) -> None:
        main_window = max(forward_windows)
        prefix_f = f"w{main_window}_flagged"
        prefix_c = f"w{main_window}_clean"

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Put-Shield Backtest — {main_window}-day forward window", fontsize=14)

        # Mean return
        ax = axes[0, 0]
        if f"{prefix_f}_mean_ret" in results.columns:
            ax.plot(results.index, results[f"{prefix_f}_mean_ret"] * 100, label="Flagged", color="red")
        if f"{prefix_c}_mean_ret" in results.columns:
            ax.plot(results.index, results[f"{prefix_c}_mean_ret"] * 100, label="Clean", color="green")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title("Mean Forward Return (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5th percentile return (left tail)
        ax = axes[0, 1]
        if f"{prefix_f}_p5_ret" in results.columns:
            ax.plot(results.index, results[f"{prefix_f}_p5_ret"] * 100, label="Flagged", color="red")
        if f"{prefix_c}_p5_ret" in results.columns:
            ax.plot(results.index, results[f"{prefix_c}_p5_ret"] * 100, label="Clean", color="green")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title("5th Percentile Return — Left Tail (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Crash rate
        ax = axes[1, 0]
        if f"{prefix_f}_crash_rate" in results.columns:
            ax.plot(results.index, results[f"{prefix_f}_crash_rate"] * 100, label="Flagged", color="red")
        if f"{prefix_c}_crash_rate" in results.columns:
            ax.plot(results.index, results[f"{prefix_c}_crash_rate"] * 100, label="Clean", color="green")
        ax.set_title("Crash Rate >20% (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Precision / recall
        ax = axes[1, 1]
        if "precision" in results.columns:
            ax.plot(results.index, results["precision"], label="Precision", color="blue")
        if "recall" in results.columns:
            ax.plot(results.index, results["recall"], label="Recall", color="orange")
        ax.set_ylim(0, 1)
        ax.set_title("Signal Precision & Recall")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / "backtest_summary.png"
        plt.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved backtest plot to %s", path)

    def _save_equity_plot(
        self,
        equity_lines: pd.DataFrame,
        period_returns: pd.DataFrame,
        stats: pd.DataFrame,
    ) -> None:
        """Save the equity-line comparison chart to ``results/equity_lines.png``.

        Layout
        ------
        Top (full width):  compounded equity lines, log scale.
        Bottom-left:       drawdown curves with fill.
        Bottom-right:      rolling 12-month compounded return.
        Footer text:       annualized stats table.
        """
        COLORS = {"all": "#1f77b4", "clean": "#2ca02c", "flagged": "#d62728"}
        LABELS = {"all": "S&P 500 (all)", "clean": "Clean basket", "flagged": "Flagged basket"}
        ORDER  = ["all", "clean", "flagged"]

        fig = plt.figure(figsize=(14, 10))
        gs  = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.45, wspace=0.3)
        ax_eq   = fig.add_subplot(gs[0, :])
        ax_dd   = fig.add_subplot(gs[1, 0], sharex=ax_eq)
        ax_roll = fig.add_subplot(gs[1, 1], sharex=ax_eq)

        # --- Equity lines (log scale) ---
        for col in ORDER:
            if col not in equity_lines.columns:
                continue
            ax_eq.plot(
                equity_lines.index, equity_lines[col],
                color=COLORS[col], label=LABELS[col], linewidth=1.8,
            )
            # Annotate final value
            final = equity_lines[col].dropna()
            if not final.empty:
                ax_eq.annotate(
                    f"${final.iloc[-1]:.0f}",
                    xy=(final.index[-1], final.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=COLORS[col], fontsize=8.5, va="center",
                )
        ax_eq.set_yscale("log")
        ax_eq.set_title("Equal-Weighted Equity Lines — $100 Invested (log scale)", fontsize=12)
        ax_eq.set_ylabel("Portfolio Value ($)")
        ax_eq.legend(loc="upper left")
        ax_eq.grid(True, alpha=0.3, which="both")

        # --- Drawdown ---
        for col in ORDER:
            if col not in equity_lines.columns:
                continue
            eq = equity_lines[col].dropna()
            dd = (eq - eq.cummax()) / eq.cummax() * 100.0
            ax_dd.fill_between(dd.index, dd, 0, alpha=0.25, color=COLORS[col])
            ax_dd.plot(dd.index, dd, color=COLORS[col], linewidth=0.9, label=LABELS[col])
        ax_dd.set_title("Drawdown (%)")
        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.legend(fontsize=7)
        ax_dd.grid(True, alpha=0.3)

        # --- Rolling 12-month compounded return ---
        for col in ORDER:
            if col not in period_returns.columns:
                continue
            ret  = period_returns[col].fillna(0.0)
            roll = (1 + ret).rolling(12).apply(lambda x: x.prod() - 1, raw=True) * 100.0
            ax_roll.plot(roll.index, roll, color=COLORS[col], label=LABELS[col], linewidth=1.1)
        ax_roll.axhline(0, color="black", linewidth=0.5)
        ax_roll.set_title("Rolling 12-Month Return (%)")
        ax_roll.set_ylabel("Return (%)")
        ax_roll.legend(fontsize=7)
        ax_roll.grid(True, alpha=0.3)

        # --- Footer: stats table ---
        if not stats.empty:
            col_hdrs = "  {:<22} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
                "Portfolio", "AnnRet", "AnnVol", "Sharpe", "MaxDD", "VaR5%", "IR"
            )
            lines = ["  Annualized stats (21-day / monthly, rf=0%):  " + col_hdrs]
            for col in ORDER:
                if col not in stats.index:
                    continue
                s  = stats.loc[col]
                ir = f"{s['info_ratio']:.2f}" if not np.isnan(s["info_ratio"]) else " N/A"
                lines.append(
                    "  {:<22} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
                        LABELS[col],
                        f"{s['ann_return']:.1%}",
                        f"{s['ann_vol']:.1%}",
                        f"{s['sharpe']:.2f}",
                        f"{s['max_drawdown']:.1%}",
                        f"{s['var_5pct']:.1%}",
                        ir,
                    )
                )
            fig.text(
                0.01, 0.002, "\n".join(lines),
                fontsize=7.5, family="monospace", verticalalignment="bottom",
            )

        fig.suptitle("Put-Shield — Equity Line Analysis (21-day / monthly)", fontsize=13)
        path = self.output_dir / "equity_lines.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved equity line chart to %s", path)

    def _print_summary(self, results: pd.DataFrame) -> None:
        logger.info("=" * 60)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 60)
        for col in results.columns:
            if results[col].dtype in [float, np.float64]:
                logger.info("  %-35s: mean=%.3f  std=%.3f", col, results[col].mean(), results[col].std())
        logger.info("=" * 60)

    def _print_portfolio_stats(self, stats: pd.DataFrame) -> None:
        LABELS = {"all": "S&P 500 (all)", "clean": "Clean basket", "flagged": "Flagged basket"}
        logger.info("=" * 72)
        logger.info("PORTFOLIO STATS  (21-day / monthly, annualized, rf = 0%%)")
        logger.info("=" * 72)
        hdr = "  {:<20} {:>8} {:>8} {:>7} {:>8} {:>7} {:>9} {:>6}".format(
            "Portfolio", "TotRet", "AnnRet", "AnnVol", "Sharpe", "MaxDD", "VaR(5%)", "IR"
        )
        logger.info(hdr)
        logger.info("-" * 72)
        for col in ["all", "clean", "flagged"]:
            if col not in stats.index:
                continue
            s  = stats.loc[col]
            ir = f"{s['info_ratio']:.2f}" if not np.isnan(s["info_ratio"]) else "  N/A"
            logger.info(
                "  {:<20} {:>8} {:>8} {:>7} {:>8} {:>7} {:>9} {:>6}".format(
                    LABELS.get(col, col),
                    f"{s['total_return']:.1%}",
                    f"{s['ann_return']:.1%}",
                    f"{s['ann_vol']:.1%}",
                    f"{s['sharpe']:.2f}",
                    f"{s['max_drawdown']:.1%}",
                    f"{s['var_5pct']:.1%}",
                    ir,
                )
            )
        logger.info("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    bt = EquityBacktest()
    results = bt.run()
    print(results.to_string())
