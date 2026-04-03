from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

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


def _parse_agent_fields(detail: object) -> dict[str, object]:
    txt = str(detail or "")
    if "agent=" not in txt:
        return {}

    tail = txt.split("agent=", 1)[1]
    chunk = "agent=" + tail
    out: dict[str, object] = {}

    for key in ["agent", "strategy_class", "calls", "decisions", "last_side"]:
        m = re.search(rf"{key}=([^;|]+)", chunk)
        if not m:
            continue
        val = m.group(1).strip()
        if key in {"calls", "decisions"}:
            try:
                out[key] = int(val)
            except ValueError:
                continue
        else:
            out[key] = val
    return out


def _build_agents_summary(activity_recent: pd.DataFrame) -> dict:
    if activity_recent.empty or "detail" not in activity_recent.columns:
        return {"rows_with_agent": 0, "by_agent": {}}

    parsed = activity_recent["detail"].map(_parse_agent_fields)
    aux = activity_recent.copy()
    aux["agent"] = parsed.map(lambda d: str(d.get("agent", "")))
    aux["strategy_class"] = parsed.map(lambda d: str(d.get("strategy_class", "")))
    aux["agent_calls"] = parsed.map(lambda d: int(d.get("calls", 0)) if d.get("calls", None) is not None else 0)
    aux["agent_decisions"] = parsed.map(lambda d: int(d.get("decisions", 0)) if d.get("decisions", None) is not None else 0)
    aux["last_side"] = parsed.map(lambda d: str(d.get("last_side", "")))

    aux = aux[aux["agent"].str.len() > 0]
    if aux.empty:
        return {"rows_with_agent": 0, "by_agent": {}}

    decision_actions = {
        "paper_signal",
        "paper_signal_eventless",
        "order_sent",
        "order_sent_eventless",
    }

    by_agent: dict[str, object] = {}
    for agent, g in aux.groupby("agent"):
        rows = int(len(g))
        signal_rows = int(g["action"].astype(str).isin(decision_actions).sum()) if "action" in g.columns else 0
        signal_rate = float(signal_rows / rows) if rows > 0 else 0.0
        by_agent[str(agent)] = {
            "strategy_class": str(g["strategy_class"].iloc[-1]) if "strategy_class" in g.columns else "",
            "rows": rows,
            "signal_rows": signal_rows,
            "signal_rate": signal_rate,
            "max_calls": int(pd.to_numeric(g["agent_calls"], errors="coerce").fillna(0).max()),
            "max_decisions": int(pd.to_numeric(g["agent_decisions"], errors="coerce").fillna(0).max()),
            "last_side": str(g["last_side"].iloc[-1]) if "last_side" in g.columns else "",
            "last_seen_utc": str(g["time_utc"].max().isoformat()) if "time_utc" in g.columns else "",
        }

    return {
        "rows_with_agent": int(len(aux)),
        "by_agent": by_agent,
    }


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

    agents_summary = _build_agents_summary(activity_recent)

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
        "agents": agents_summary,
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
