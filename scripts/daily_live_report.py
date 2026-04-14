from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

import pandas as pd

from src.config import settings
from src.mt5_executor import MT5Executor


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


def _build_mt5_performance_summary(hours: int) -> dict:
    summary = {
        "available": False,
        "symbol": str(settings.symbol),
        "closed_deals": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "net_profit": 0.0,
        "profit_factor": 0.0,
        "avg_deal_profit": 0.0,
        "max_drawdown_profit": 0.0,
        "error": "",
    }

    execu = MT5Executor()
    try:
        execu.initialize()
        days = max(1, int((int(hours) + 23) // 24))
        deals = execu.get_recent_deals(symbol=str(settings.symbol), days=days)
        if deals.empty:
            summary["available"] = True
            return summary

        if "entry_label" in deals.columns:
            deals = deals[deals["entry_label"].astype(str).str.upper() == "CLOSE"].copy()
        if deals.empty:
            summary["available"] = True
            return summary

        deals["profit"] = pd.to_numeric(deals.get("profit", 0.0), errors="coerce").fillna(0.0)
        deals = deals.sort_values("time_utc") if "time_utc" in deals.columns else deals

        pnl = deals["profit"].astype(float)
        wins = int((pnl > 0).sum())
        losses = int((pnl < 0).sum())
        gross_profit = float(pnl[pnl > 0].sum()) if wins > 0 else 0.0
        gross_loss_abs = float((-pnl[pnl < 0]).sum()) if losses > 0 else 0.0
        net_profit = float(pnl.sum())
        pf = float(gross_profit / gross_loss_abs) if gross_loss_abs > 0 else (999.0 if gross_profit > 0 else 0.0)

        eq = pnl.cumsum()
        running_max = eq.cummax() if not eq.empty else eq
        dd = (running_max - eq) if not eq.empty else eq
        max_dd = float(dd.max()) if not dd.empty else 0.0

        n = int(len(deals))
        summary.update(
            {
                "available": True,
                "closed_deals": n,
                "wins": wins,
                "losses": losses,
                "win_rate": float(wins / n) if n > 0 else 0.0,
                "gross_profit": gross_profit,
                "gross_loss": float(-gross_loss_abs),
                "net_profit": net_profit,
                "profit_factor": pf,
                "avg_deal_profit": float(net_profit / n) if n > 0 else 0.0,
                "max_drawdown_profit": max_dd,
            }
        )
    except Exception as ex:
        summary["error"] = str(ex)[:220]
    finally:
        try:
            execu.shutdown()
        except Exception:
            pass

    return summary


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
    mt5_summary = _build_mt5_performance_summary(hours=hours)

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
        "mt5_performance": mt5_summary,
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
