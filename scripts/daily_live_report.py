from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat()


def build_report(hours: int = 24) -> dict:
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)

    activity_path = Path("data/live_activity.csv")
    paper_path = Path("data/paper_trades.csv")
    out_path = Path("models/daily_live_report.json")

    activity = _load_csv(activity_path)
    paper = _load_csv(paper_path)

    if not activity.empty and "time_utc" in activity.columns:
        activity["time_utc"] = pd.to_datetime(activity["time_utc"], utc=True, errors="coerce")
        activity = activity.dropna(subset=["time_utc"])
        activity_recent = activity[activity["time_utc"] >= pd.Timestamp(since)]
    else:
        activity_recent = pd.DataFrame()

    if not paper.empty and "time_utc" in paper.columns:
        paper["time_utc"] = pd.to_datetime(paper["time_utc"], utc=True, errors="coerce")
        paper = paper.dropna(subset=["time_utc"])
        paper_recent = paper[paper["time_utc"] >= pd.Timestamp(since)]
    else:
        paper_recent = pd.DataFrame()

    actions = {}
    if not activity_recent.empty and "action" in activity_recent.columns:
        actions = activity_recent["action"].value_counts().to_dict()

    paper_summary = {
        "signals": 0,
        "buy_signals": 0,
        "sell_signals": 0,
        "avg_confidence": 0.0,
    }
    if not paper_recent.empty:
        side = paper_recent.get("side", pd.Series(dtype=str)).astype(str).str.upper()
        conf = pd.to_numeric(paper_recent.get("confidence", pd.Series(dtype=float)), errors="coerce").dropna()
        paper_summary = {
            "signals": int(len(paper_recent)),
            "buy_signals": int((side == "BUY").sum()),
            "sell_signals": int((side == "SELL").sum()),
            "avg_confidence": float(conf.mean()) if not conf.empty else 0.0,
        }

    report = {
        "generated_at_utc": _safe_iso(now),
        "window_hours": int(hours),
        "since_utc": _safe_iso(since),
        "activity": {
            "rows": int(len(activity_recent)),
            "actions": actions,
        },
        "paper": paper_summary,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = build_report(hours=24)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
