"""Put-Shield screener — CSV in, CSV out.

Usage:
    python -m src.dashboard.screener                        # uses watchlist.csv
    python -m src.dashboard.screener my_candidates.csv     # custom input file
    python -m src.dashboard.screener candidates.csv --out results.csv

Input CSV must have a column named 'ticker'.
Output CSV contains one row per ticker with danger scores and a plain-English verdict.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from src.aggregator.combiner import SignalCombiner

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Verdict labels ────────────────────────────────────────────────────────────

def _verdict(score: float, hard_stop: bool) -> str:
    """Convert a numeric danger score into a plain-English verdict."""
    if hard_stop:
        return "HARD STOP"
    if pd.isna(score):
        return "NO DATA"
    if score >= 0.65:
        return "DANGER"
    if score >= 0.40:
        return "CAUTION"
    return "SAFE"


def _verdict_sort_key(verdict: str) -> int:
    """Lower number = worse / more urgent."""
    return {"HARD STOP": 0, "DANGER": 1, "CAUTION": 2, "NO DATA": 3, "SAFE": 4}.get(verdict, 5)


# ── Main screener logic ───────────────────────────────────────────────────────

def run_screener(
    input_path: str | Path,
    output_path: str | Path | None = None,
    config_path: str | Path = "configs/strategy.yaml",
    cache_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Run the screener on tickers listed in *input_path*.

    Returns the results DataFrame and writes it to *output_path* (defaults to
    screener_output.csv next to the input file).
    """
    input_path = Path(input_path)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # ── Read tickers ──────────────────────────────────────────────────────────
    try:
        watchlist = pd.read_csv(input_path)
    except Exception as exc:
        logger.error("Could not read %s: %s", input_path, exc)
        sys.exit(1)

    if "ticker" not in watchlist.columns:
        logger.error("CSV must have a column named 'ticker'. Found: %s", list(watchlist.columns))
        sys.exit(1)

    tickers = watchlist["ticker"].dropna().str.upper().str.strip().unique().tolist()
    if not tickers:
        logger.error("No tickers found in %s", input_path)
        sys.exit(1)

    logger.info("Screening %d tickers as of today (%s)…", len(tickers), date.today())

    # ── Run signals ───────────────────────────────────────────────────────────
    combiner = SignalCombiner(config_path=config_path, cache_dir=cache_dir)
    as_of = pd.Timestamp(date.today())

    try:
        scores = combiner.compute(tickers, as_of)
    except Exception as exc:
        logger.error("Screener failed: %s", exc)
        sys.exit(1)

    # ── Build output table ────────────────────────────────────────────────────
    out = pd.DataFrame(index=scores.index)
    out.index.name = "ticker"

    out["verdict"] = [
        _verdict(scores.loc[t, "composite_score"], bool(scores.loc[t, "hard_stop"]))
        for t in out.index
    ]
    out["danger_score"] = scores["composite_score"].round(3)
    out["hard_stop"] = scores["hard_stop"].astype(bool)

    # Pull through individual signal scores
    for col in scores.columns:
        if col.endswith("_score") and col != "composite_score":
            out[col] = scores[col].round(3)

    # Attach raw values (z_score, formula, m_score, imputed_count, etc.)
    # by re-running each signal once and pulling their raw_values DataFrames.
    out = _attach_raw_signal_values(out, tickers, combiner, as_of)

    # Sort by urgency: HARD STOP first, then DANGER, CAUTION, SAFE, NO DATA
    out["_sort"] = out["verdict"].map(_verdict_sort_key)
    out = out.sort_values("_sort").drop(columns="_sort")
    out["as_of_date"] = date.today()

    # ── Write output ──────────────────────────────────────────────────────────
    if output_path is None:
        output_path = input_path.parent / "screener_output.csv"
    output_path = Path(output_path)

    out.reset_index().to_csv(output_path, index=False)
    logger.info("Results written to %s", output_path)

    # ── Print summary to terminal ─────────────────────────────────────────────
    _print_table(out)

    return out


def _attach_raw_signal_values(
    out: pd.DataFrame,
    tickers: list[str],
    combiner: SignalCombiner,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Attach raw metric values from every enabled signal to the output table."""
    # Columns to surface per signal — add entries here as new signals are built
    WANTED = {
        "altman_z":  ["z_score", "formula"],
        "beneish_m": ["m_score", "imputed_count"],
    }
    for signal_name, cols in WANTED.items():
        entry = combiner.signals.get(signal_name)
        if entry is None:
            continue
        try:
            signal_obj, _ = entry
            result = signal_obj.compute(tickers, as_of)
            for col in cols:
                if col in result.raw_values.columns:
                    vals = result.raw_values[col]
                    out[col] = vals.round(2) if vals.dtype == float else vals
        except Exception as exc:
            logger.debug("Could not attach raw values for %s: %s", signal_name, exc)
    return out


def _print_table(out: pd.DataFrame) -> None:
    """Print a readable summary table to the terminal."""
    divider = "─" * 82
    print(f"\n{divider}")
    print(f"  PUT-SHIELD SCREENER — {date.today()}")
    print(divider)
    print(f"  {'TICKER':<8}  {'VERDICT':<12}  {'SCORE':>6}  {'Z-SCORE':>8}  {'M-SCORE':>8}  {'FORMULA'}")
    print(divider)

    for ticker, row in out.iterrows():
        verdict = row["verdict"]
        score_str = f"{row['danger_score']:.3f}" if pd.notna(row.get("danger_score")) else "  N/A"
        z_str = f"{row['z_score']:.2f}" if "z_score" in row and pd.notna(row.get("z_score")) else "    N/A"
        m_str = f"{row['m_score']:.2f}" if "m_score" in row and pd.notna(row.get("m_score")) else "    N/A"
        formula = row.get("formula", "")

        flag = ""
        if verdict == "HARD STOP":
            flag = "  ◄◄ AVOID"
        elif verdict == "DANGER":
            flag = "  ◄ WARNING"

        print(f"  {ticker:<8}  {verdict:<12}  {score_str:>6}  {z_str:>8}  {m_str:>8}  {formula}{flag}")

    print(divider)
    counts = out["verdict"].value_counts()
    print(f"  Hard stops: {counts.get('HARD STOP', 0)}  |  "
          f"Danger: {counts.get('DANGER', 0)}  |  "
          f"Caution: {counts.get('CAUTION', 0)}  |  "
          f"Safe: {counts.get('SAFE', 0)}  |  "
          f"No data: {counts.get('NO DATA', 0)}")
    print(f"{divider}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Put-Shield stock screener")
    parser.add_argument(
        "input", nargs="?", default="watchlist.csv",
        help="CSV file with a 'ticker' column (default: watchlist.csv)"
    )
    parser.add_argument(
        "--out", default=None,
        help="Output CSV path (default: screener_output.csv next to input)"
    )
    parser.add_argument(
        "--config", default="configs/strategy.yaml",
        help="Strategy config file (default: configs/strategy.yaml)"
    )
    args = parser.parse_args()

    run_screener(
        input_path=args.input,
        output_path=args.out,
        config_path=args.config,
    )
