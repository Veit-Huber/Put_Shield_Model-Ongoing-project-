# Put-Shield — Crash Detection for Put-Selling Strategies

A quantitative risk-screening system that identifies stocks at elevated risk of catastrophic drawdown before selling put options on them. Built to protect a put-selling portfolio from Enron- and Lehman-style blow-ups.

## What it does

Options traders who sell puts on volatile stocks collect premium — but occasionally one of those stocks collapses 60–80%, wiping out months of gains in a single position. Put-Shield screens every candidate stock before entry and flags the ones most likely to crash.

It is a **bad-apple detector**, not a return predictor. The goal is asymmetric: miss some safe stocks rather than ever sell a put on the next Bed Bath & Beyond.

## Architecture

```
Yahoo Finance ──► Data Layer    ──► Signal Layer  ──► Aggregator ──► Screener / Backtest
(yfinance)        (caching,          (Altman Z,        (weighted       (CSV output,
                  filing lags)       Beneish M, ...)    composite)      walk-forward)
```

The four layers communicate through clean interfaces — adding a new signal requires no changes to the aggregator or backtest.

## Signals implemented

| Signal | What it detects | Hard-stop threshold |
|---|---|---|
| **Altman Z-Score** | Bankruptcy probability from 5 financial ratios | Z < 1.81 (distress zone) |
| **Beneish M-Score** | Reported earnings manipulation probability from 8 financial ratios | M > -1.78 (distress zone) |

Planned: Piotroski F-Score (financial health), IV skew, short interest, macro regime, etc.

## Backtest results (8-ticker universe, 2021–2023)

| Group | Mean 3-month return | Crash rate (>20% drawdown) |
|---|---|---|
| **Flagged by system** | −16.9% | 38.9% |
| **Cleared by system** | +9.1% | 12.3% |

Flagged stocks crashed **3× more often** than cleared stocks. The two stocks flagged from April 2023 onward (post-meme-stock collapse) returned −46% in a single 63-day window.

## Screener usage

```bash
# 1. Add your candidates to watchlist.csv (one ticker per row)
# 2. Run the screener
python -m src.dashboard.screener

# Output — screener_output.csv:
# ticker  verdict     danger_score  z_score  formula
# AMC     HARD STOP   0.981         0.82     Z''
# TSLA    CAUTION     0.523         1.94     Z''
# AAPL    SAFE        0.118         4.31     Z''
```

## Backtest usage

```bash
# Small test universe (fast)
# Set source: manual in configs/strategy.yaml, then:
python -m src.backtest.equity_backtest

# Full S&P 500 + historical failures including Enron, Lehman, BBBY (slower, more accurate)
# Set source: full in configs/strategy.yaml, then:
python -m src.backtest.equity_backtest
```

## Key design decisions

**Filing-lag adjustment** — financial data is shifted forward 45 days (quarterly) or 75 days (annual) before use, matching real SEC filing deadlines. This prevents look-ahead bias in backtesting.

**Two Z-Score variants** — original Z for manufacturing firms (uses market cap), Z'' for tech/finance/services (uses book equity). Sector is detected automatically from Yahoo Finance metadata.

**Survivorship bias correction** — the backtest universe includes delisted and bankrupt companies (Enron, Lehman, Sears, Hertz, etc.), not just current S&P 500 constituents. Testing only on survivors overstates signal performance.

**NaN-safe aggregation** — missing data returns NaN scores, not zero. The aggregator excludes NaN signals from the weighted average rather than treating absence of data as safety.

## Stack

```
Python 3.11+  |  pandas / numpy  |  yfinance  |  pydantic  |  pyarrow  |  matplotlib  |  pytest
```

## Project structure

```
src/
  data/         DataSource ABC, Yahoo Finance fetcher, universe builder
  signals/      BaseSignal ABC, Altman Z-Score
  aggregator/   Config-driven weighted signal combiner
  backtest/     Walk-forward equity backtest engine
  dashboard/    CSV screener
configs/        strategy.yaml — all thresholds and weights
tests/          38 unit tests, no network required
```

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v      # 38 tests, ~1 second, no internet required
```
