from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from analyze_candidate_families import group_report, write_csv
from analyze_histdata_macro_robustness import build_macro_feature_frame, json_safe
from analyze_histdata_regime_sweep import add_regime_sweep_buckets
from build_usd_high_impact_events import generated_events
from run_execution_aware_sell_experiment import SPREAD_BY_SESSION, add_session_column
from train_xgboost import FEATURE_COLUMNS, build_training_features, load_csv
from xauusd_signal.macro_features import add_real_macro_features, load_us10y_csv
from xauusd_signal.research.candidate_families import generate_overlap_macro_trend_candidates
from xauusd_signal.research.candidates import CandidateConfig
from xauusd_signal.research.execution import ExecutionConfig, execution_aware_trade_labels, summarize_r
from xauusd_signal.research.labels import TripleBarrierConfig


def focused_buy_v2_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["side"].eq("BUY")
        & frame["source_candidate_family"].eq("trend_continuation")
        & frame["usd_return_80_bucket"].eq("falling_fast")
    )


def focused_rule_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    base = frame["side"].eq("BUY") & frame["source_candidate_family"].eq("trend_continuation")
    usd80 = frame["usd_return_80_bucket"].eq("falling_fast")
    usd20_80 = usd80 & frame["usd_return_20_bucket"].eq("falling_fast")
    yields_falling = frame["us10y_change_10d_bucket"].eq("falling") & frame["us10y_change_20d_bucket"].eq("falling")
    return {
        "usd80_falling_fast": base & usd80,
        "usd20_and_usd80_falling_fast": base & usd20_80,
        "yields_10d_20d_falling": base & yields_falling,
        "usd80_or_yields_falling": base & (usd80 | yields_falling),
    }


def add_time_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["year"] = pd.to_datetime(output["timestamp"], utc=True).dt.year
    output["period"] = pd.cut(
        output["year"],
        bins=[2012, 2016, 2020, 2024, 2027],
        labels=["2013-2016", "2017-2020", "2021-2024", "2025-2026"],
    ).astype(str)
    return output


def build_production_feature_frame(m15: Path, h1: Path, h4: Path, dxy: Path, us10y: Path) -> pd.DataFrame:
    dxy_frame = load_csv(dxy)
    frame = build_training_features(load_csv(m15), load_csv(h1), load_csv(h4), dxy_frame, 0.0).dropna(subset=FEATURE_COLUMNS)
    frame = add_real_macro_features(frame, dxy_frame, load_us10y_csv(us10y))
    macro_columns = [
        "real_dxy_return_20",
        "real_dxy_return_80",
        "us10y_yield",
        "us10y_change_10d",
        "us10y_change_20d",
        "sell_regime_block",
    ]
    return frame.dropna(subset=macro_columns).reset_index(drop=True)


def label_dataset(
    frame: pd.DataFrame,
    label_config: TripleBarrierConfig,
    execution_config: ExecutionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = add_session_column(frame.reset_index(drop=True))
    frame["source_index"] = frame.index
    news_events = generated_events(
        frame["timestamp"].min().to_pydatetime().astimezone(UTC),
        frame["timestamp"].max().to_pydatetime().astimezone(UTC),
    )
    news_events["timestamp"] = pd.to_datetime(news_events["timestamp"], utc=True)
    candidates = generate_overlap_macro_trend_candidates(frame, CandidateConfig())
    labeled = execution_aware_trade_labels(
        candidates,
        frame,
        label_config,
        news_events,
        SPREAD_BY_SESSION,
        execution_config,
    )
    usable = labeled.loc[~labeled["execution_blocked"].astype(bool)].copy()
    return labeled, add_time_buckets(add_regime_sweep_buckets(usable))


def period_rows(frame: pd.DataFrame, group_column: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows = []
    for value, group in frame.groupby(group_column, dropna=False):
        rows.append({group_column: str(value), **summarize_r(group["event_r"].to_numpy())})
    return rows


def evaluate_dataset(
    name: str,
    frame: pd.DataFrame,
    label_config: TripleBarrierConfig,
    execution_config: ExecutionConfig,
    report_dir: Path,
) -> dict[str, Any]:
    labeled, usable = label_dataset(frame, label_config, execution_config)
    broad_buy = usable.loc[usable["side"].eq("BUY")].copy()
    broad_trend_buy = usable.loc[usable["side"].eq("BUY") & usable["source_candidate_family"].eq("trend_continuation")].copy()

    dataset_dir = report_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    variant_rows = []
    variants = {}
    for variant, mask in focused_rule_masks(usable).items():
        focused = usable.loc[mask].copy()
        variant_summary = summarize_r(focused["event_r"].to_numpy())
        bad_periods = [
            row
            for row in period_rows(focused, "period")
            if row["trades"] >= 25
            and (
                row["expected_r"] is None
                or row["expected_r"] < 0
                or row["profit_factor"] is None
                or row["profit_factor"] < 1
            )
        ]
        variants[variant] = {
            **variant_summary,
            "bad_periods": len(bad_periods),
            "periods": period_rows(focused, "period"),
        }
        variant_rows.append({"variant": variant, **variant_summary, "bad_periods": len(bad_periods)})
        write_csv(dataset_dir / f"{variant}_by_period.csv", period_rows(focused, "period"))
        write_csv(dataset_dir / f"{variant}_by_year.csv", period_rows(focused, "year"))
    focused = usable.loc[focused_buy_v2_mask(usable)].copy()
    write_csv(dataset_dir / "variant_summary.csv", variant_rows)
    write_csv(dataset_dir / "broad_buy_by_period.csv", period_rows(broad_buy, "period"))
    write_csv(dataset_dir / "by_source_family_side.csv", group_report(usable, ["source_candidate_family", "side"]))

    focused_summary = summarize_r(focused["event_r"].to_numpy())
    broad_buy_summary = summarize_r(broad_buy["event_r"].to_numpy())
    broad_trend_buy_summary = summarize_r(broad_trend_buy["event_r"].to_numpy())
    blocked_counts = labeled["execution_block_reason"].value_counts(dropna=False).to_dict() if not labeled.empty else {}
    summary = {
        "dataset": name,
        "feature_rows": int(len(frame)),
        "feature_start": frame["timestamp"].iloc[0].isoformat() if not frame.empty else None,
        "feature_end": frame["timestamp"].iloc[-1].isoformat() if not frame.empty else None,
        "candidate_count": int(len(labeled)),
        "usable_count": int(len(usable)),
        "blocked_counts": {str(key): int(value) for key, value in blocked_counts.items()},
        "rule": {
            "side": "BUY",
            "source_candidate_family": "trend_continuation",
            "usd_return_80_bucket": "falling_fast",
        },
        "focused_v2": focused_summary,
        "variants": variants,
        "broad_buy_baseline": broad_buy_summary,
        "broad_trend_buy_baseline": broad_trend_buy_summary,
        "focused_edge_minus_broad_buy": (
            focused_summary["expected_r"] - broad_buy_summary["expected_r"]
            if focused_summary["expected_r"] is not None and broad_buy_summary["expected_r"] is not None
            else None
        ),
        "focused_edge_minus_broad_trend_buy": (
            focused_summary["expected_r"] - broad_trend_buy_summary["expected_r"]
            if focused_summary["expected_r"] is not None and broad_trend_buy_summary["expected_r"] is not None
            else None
        ),
        "focused_bad_periods": variants["usd80_falling_fast"]["bad_periods"],
        "focused_periods": period_rows(focused, "period"),
    }
    (dataset_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    return summary


def decision_for(results: list[dict[str, Any]], min_trades: int, min_expected_r: float, min_profit_factor: float, max_bad_periods: int) -> str:
    if not results:
        return "missing_datasets"
    for result in results:
        focused = result["focused_v2"]
        if focused["trades"] < min_trades:
            return "not_enough_focused_trades"
        if focused["expected_r"] is None or focused["expected_r"] < min_expected_r:
            return "redesign_focused_rule"
        if focused["profit_factor"] is None or focused["profit_factor"] < min_profit_factor:
            return "redesign_focused_rule"
        if result["focused_bad_periods"] > max_bad_periods:
            return "add_stronger_regime_or_kill_switch"
    return "candidate_ready_for_paper_gate"


def best_variant_across_datasets(
    results: list[dict[str, Any]],
    min_trades: int,
    min_expected_r: float,
    min_profit_factor: float,
    max_bad_periods: int,
) -> dict[str, Any] | None:
    if not results:
        return None
    names = list(results[0]["variants"].keys())
    candidates = []
    for name in names:
        summaries = [result["variants"][name] for result in results]
        if any(item["trades"] < min_trades for item in summaries):
            continue
        if any(item["expected_r"] is None or item["expected_r"] < min_expected_r for item in summaries):
            continue
        if any(item["profit_factor"] is None or item["profit_factor"] < min_profit_factor for item in summaries):
            continue
        if any(item["bad_periods"] > max_bad_periods for item in summaries):
            continue
        candidates.append(
            {
                "variant": name,
                "min_expected_r": min(item["expected_r"] for item in summaries),
                "min_profit_factor": min(item["profit_factor"] for item in summaries if item["profit_factor"] is not None),
                "total_trades": sum(item["trades"] for item in summaries),
            }
        )
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item["min_expected_r"], item["min_profit_factor"], item["total_trades"]), reverse=True)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hist-m15", type=Path, default=Path("data/research/histdata/xauusd_15min.csv"))
    parser.add_argument("--hist-h1", type=Path, default=Path("data/research/histdata/xauusd_1h.csv"))
    parser.add_argument("--hist-h4", type=Path, default=Path("data/research/histdata/xauusd_4h.csv"))
    parser.add_argument("--hist-dxy", type=Path, default=Path("data/research/macro/dxy_daily.csv"))
    parser.add_argument("--hist-us10y", type=Path, default=Path("data/research/macro/us10y_daily.csv"))
    parser.add_argument("--prod-m15", type=Path, default=Path("data/training/xauusd_m15.csv"))
    parser.add_argument("--prod-h1", type=Path, default=Path("data/training/xauusd_h1.csv"))
    parser.add_argument("--prod-h4", type=Path, default=Path("data/training/xauusd_h4.csv"))
    parser.add_argument("--prod-dxy", type=Path, default=Path("data/training/dxy_m15.csv"))
    parser.add_argument("--prod-us10y", type=Path, default=Path("data/training/us10y_daily.csv"))
    parser.add_argument("--tp-atr", type=float, default=2.0)
    parser.add_argument("--sl-atr", type=float, default=1.0)
    parser.add_argument("--vertical", type=int, default=8)
    parser.add_argument("--entry-delay", type=int, default=1)
    parser.add_argument("--min-trades", type=int, default=100)
    parser.add_argument("--min-expected-r", type=float, default=0.05)
    parser.add_argument("--min-profit-factor", type=float, default=1.10)
    parser.add_argument("--max-bad-periods", type=int, default=1)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/research/focused_buy_v2"))
    args = parser.parse_args()

    label_config = TripleBarrierConfig(args.tp_atr, args.sl_atr, args.vertical)
    execution_config = ExecutionConfig(entry_delay_candles=args.entry_delay)
    datasets = [
        (
            "histdata_long",
            build_macro_feature_frame(args.hist_m15, args.hist_h1, args.hist_h4, args.hist_dxy, args.hist_us10y),
        ),
        (
            "production_current",
            build_production_feature_frame(args.prod_m15, args.prod_h1, args.prod_h4, args.prod_dxy, args.prod_us10y),
        ),
    ]
    args.report_dir.mkdir(parents=True, exist_ok=True)
    results = [evaluate_dataset(name, frame, label_config, execution_config, args.report_dir) for name, frame in datasets]
    summary = {
        "hypothesis": "Focused BUY v2: overlap macro trend continuation with broad USD falling fast",
        "decision_rules": {
            "min_trades": args.min_trades,
            "min_expected_r": args.min_expected_r,
            "min_profit_factor": args.min_profit_factor,
            "max_bad_periods": args.max_bad_periods,
        },
        "datasets": results,
    }
    best_variant = best_variant_across_datasets(results, args.min_trades, args.min_expected_r, args.min_profit_factor, args.max_bad_periods)
    summary["best_variant_across_datasets"] = best_variant
    summary["decision"] = "variant_ready_for_paper_gate" if best_variant else decision_for(
        results,
        args.min_trades,
        args.min_expected_r,
        args.min_profit_factor,
        args.max_bad_periods,
    )
    (args.report_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    for result in results:
        focused = result["focused_v2"]
        broad = result["broad_buy_baseline"]
        print(
            f"dataset={result['dataset']} focused_trades={focused['trades']} "
            f"focused_expected_r={focused['expected_r']} focused_pf={focused['profit_factor']} "
            f"broad_buy_expected_r={broad['expected_r']} bad_periods={result['focused_bad_periods']}"
        )
        for variant, variant_summary in result["variants"].items():
            print(
                f"dataset={result['dataset']} variant={variant} trades={variant_summary['trades']} "
                f"expected_r={variant_summary['expected_r']} pf={variant_summary['profit_factor']} "
                f"bad_periods={variant_summary['bad_periods']}"
            )
    print(f"decision={summary['decision']} report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
