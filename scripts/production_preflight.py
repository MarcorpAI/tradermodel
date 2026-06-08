from __future__ import annotations

import csv
import os
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

from xauusd_signal.config import load_settings


REQUIRED_ENV = ["GROQ_API_KEY", "DISCORD_WEBHOOK_URL"]


def check(condition: bool, label: str, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    suffix = f" {detail}" if detail else ""
    print(f"{status} {label}{suffix}")
    return condition


def check_csv(path: Path, required_columns: set[str], label: str) -> bool:
    if not check(path.exists(), label, str(path)):
        return False
    frame = pd.read_csv(path, nrows=5)
    missing = required_columns - set(frame.columns)
    return check(not missing, f"{label}_schema", f"missing={sorted(missing)}" if missing else "")


def latest_timestamp(path: Path) -> datetime | None:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    if frame.empty:
        return None
    return pd.to_datetime(frame["timestamp"], utc=True).max().to_pydatetime()


def check_calendar(path: Path) -> bool:
    if not check(path.exists(), "calendar_file", str(path)):
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"timestamp", "title", "impact", "currency"}
        missing = expected - set(reader.fieldnames or [])
        if not check(not missing, "calendar_schema", f"missing={sorted(missing)}" if missing else ""):
            return False
        rows = list(reader)
    invalid_rows = 0
    for row in rows:
        try:
            datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        except ValueError:
            invalid_rows += 1
    return check(invalid_rows == 0, "calendar_rows", f"events={len(rows)} invalid={invalid_rows}")


def check_model(path: Path) -> bool:
    if not check(path.exists(), "model_file", str(path)):
        return False
    artifact = joblib.load(path)
    ok = True
    ok &= check(isinstance(artifact, dict), "model_artifact_dict")
    artifact_type = artifact.get("artifact_type") if isinstance(artifact, dict) else None
    ok &= check(artifact_type in {"side_meta_xgboost", "overlap_macro_trend_xgboost"}, "model_artifact_type", str(artifact_type))
    if not ok:
        return False

    feature_columns = artifact.get("feature_columns", [])
    if artifact_type == "side_meta_xgboost":
        ok &= check(artifact.get("side") == "SELL", "model_side", str(artifact.get("side")))
        ok &= check(float(artifact.get("threshold", 0)) == 0.55, "model_threshold", str(artifact.get("threshold")))
        ok &= check("sell_regime_block" in feature_columns, "model_macro_gate_feature")
        ok &= check(len(feature_columns) >= 40, "model_feature_count", str(len(feature_columns)))
    elif artifact_type == "overlap_macro_trend_xgboost":
        enabled_sides = list(artifact.get("enabled_sides", []))
        ok &= check(enabled_sides == ["BUY"], "model_enabled_sides", str(enabled_sides))
        ok &= check(float(artifact.get("threshold", 0)) == 0.5, "model_threshold", str(artifact.get("threshold")))
        required_features = {
            "side_buy",
            "side_sell",
            "source_family_breakout",
            "source_family_ema_pullback",
            "source_family_trend_continuation",
        }
        missing = required_features - set(feature_columns)
        ok &= check(not missing, "model_focused_features", f"missing={sorted(missing)}" if missing else "")
        ok &= check(len(feature_columns) >= 45, "model_feature_count", str(len(feature_columns)))
    return bool(ok)


def check_paper_signal_gate(config: dict) -> bool:
    gate = config.get("paper_signal_gate", {})
    if not gate:
        return check(False, "paper_signal_gate_config", "missing")
    ok = True
    ok &= check(bool(gate.get("enabled", False)), "paper_signal_gate_enabled")
    ok &= check(
        gate.get("variant") == "usd80_or_yields_falling",
        "paper_signal_gate_variant",
        str(gate.get("variant")),
    )
    ok &= check(int(gate.get("min_paper_trades_before_live", 0)) >= 50, "paper_signal_gate_min_trades", str(gate.get("min_paper_trades_before_live")))
    return bool(ok)


def main() -> None:
    load_dotenv()
    settings = load_settings()
    root = settings.root
    ok = True
    runtime_mode = str(settings.raw.get("runtime", {}).get("mode", "")).lower()
    ok &= check(runtime_mode == "research_only", "runtime_research_only", runtime_mode)
    ok &= check_paper_signal_gate(settings.raw)

    for name in REQUIRED_ENV:
        ok &= check(bool(os.getenv(name)), f"env_{name}")
    if str(settings.raw["calendar"].get("provider", "manual")).lower() in {"fmp", "financial_modeling_prep"}:
        ok &= check(bool(os.getenv("FMP_API_KEY") or os.getenv("FINANCIAL_MODELING_PREP_API_KEY")), "env_FMP_API_KEY")

    ok &= check_model(settings.model_path)

    macro = settings.raw["macro_data"]
    dxy_path = root / macro["dxy_path"]
    us10y_path = root / macro["us10y_path"]
    ok &= check_csv(dxy_path, {"timestamp", "open", "high", "low", "close"}, "dxy_macro_file")
    ok &= check_csv(us10y_path, {"timestamp", "close"}, "us10y_macro_file")

    now = datetime.now(UTC)
    for label, path, max_age_days in [("dxy_macro_freshness", dxy_path, 7), ("us10y_macro_freshness", us10y_path, 10)]:
        latest = latest_timestamp(path)
        age_days = (now - latest).total_seconds() / 86400 if latest else float("inf")
        ok &= check(latest is not None and age_days <= max_age_days, label, f"latest={latest} age_days={age_days:.1f}")

    calendar_path = root / settings.raw["calendar"]["manual_events_csv"]
    ok &= check_calendar(calendar_path)

    db_path = settings.db_path
    ok &= check(db_path.parent.exists(), "database_dir", str(db_path.parent))
    ok &= check(settings.log_path.parent.exists(), "log_dir", str(settings.log_path.parent))

    if not ok:
        raise SystemExit("production_preflight_failed")
    print("production_preflight_ready=true")


if __name__ == "__main__":
    main()
