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
        "total_deals": 0,
        "open_deals": 0,
        "close_deals": 0,
        "reverse_deals": 0,
        "close_by_deals": 0,
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

        deals_all = deals.copy()
        if "entry_label" in deals_all.columns:
            entry_upper = deals_all["entry_label"].astype(str).str.upper()
            summary["total_deals"] = int(len(deals_all))
            summary["open_deals"] = int((entry_upper == "OPEN").sum())
            summary["close_deals"] = int((entry_upper == "CLOSE").sum())
            summary["reverse_deals"] = int((entry_upper == "REVERSE").sum())
            summary["close_by_deals"] = int((entry_upper == "CLOSE_BY").sum())
            deals = deals_all[entry_upper == "CLOSE"].copy()
        else:
            summary["total_deals"] = int(len(deals_all))
            summary["close_deals"] = int(len(deals_all))
            deals = deals_all.copy()

        summary["closed_deals"] = int(summary["close_deals"])

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


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _build_performance_semaphore(mt5_summary: dict, state_path: Path) -> dict:
    out = {
        "state": "AMARILLO",
        "message": "Sin suficientes datos para confirmar estabilidad.",
        "reasons": [],
        "degraded_vs_best": False,
        "thresholds": {
            "min_deals": 6,
            "green_pf": 1.20,
            "yellow_pf": 1.00,
            "green_win_rate": 0.50,
            "yellow_win_rate": 0.45,
            "green_dd": 6.00,
            "yellow_dd": 9.00,
        },
        "best_reference": None,
    }

    if not isinstance(mt5_summary, dict) or not bool(mt5_summary.get("available", False)):
        out["state"] = "ROJO"
        out["message"] = "Sin datos MT5 disponibles para evaluar rendimiento."
        out["reasons"] = ["mt5_unavailable"]
        return out

    deals = _to_int(mt5_summary.get("closed_deals", 0))
    pf = _to_float(mt5_summary.get("profit_factor", 0.0))
    net = _to_float(mt5_summary.get("net_profit", 0.0))
    wr = _to_float(mt5_summary.get("win_rate", 0.0))
    dd = _to_float(mt5_summary.get("max_drawdown_profit", 0.0))

    if deals < int(out["thresholds"]["min_deals"]):
        out["state"] = "AMARILLO"
        out["message"] = "Muestra pequeña: esperar más operaciones para una lectura confiable."
        out["reasons"].append("low_sample")
    else:
        red_flags = []
        yellow_flags = []

        if pf < 1.0:
            red_flags.append("pf_below_1")
        elif pf < float(out["thresholds"]["green_pf"]):
            yellow_flags.append("pf_soft")

        if net < 0.0:
            red_flags.append("net_negative")
        elif net < 1.0:
            yellow_flags.append("net_low")

        if dd > float(out["thresholds"]["yellow_dd"]):
            red_flags.append("dd_high")
        elif dd > float(out["thresholds"]["green_dd"]):
            yellow_flags.append("dd_elevated")

        if wr < float(out["thresholds"]["yellow_win_rate"]):
            red_flags.append("win_rate_low")
        elif wr < float(out["thresholds"]["green_win_rate"]):
            yellow_flags.append("win_rate_soft")

        if red_flags:
            out["state"] = "ROJO"
            out["message"] = "Rendimiento degradado: conviene mantener rollback activo y no subir riesgo."
            out["reasons"] = red_flags + yellow_flags
        elif yellow_flags:
            out["state"] = "AMARILLO"
            out["message"] = "Rendimiento intermedio: mantener monitoreo, sin cambios agresivos."
            out["reasons"] = yellow_flags
        else:
            out["state"] = "VERDE"
            out["message"] = "Rendimiento saludable en la ventana actual."

    if state_path.exists():
        try:
            state_obj = json.loads(state_path.read_text(encoding="utf-8"))
            best_metrics = ((state_obj.get("best") or {}).get("metrics") or {}) if isinstance(state_obj, dict) else {}
            if isinstance(best_metrics, dict) and best_metrics:
                best_pf = _to_float(best_metrics.get("profit_factor", 0.0))
                best_net = _to_float(best_metrics.get("net_profit", 0.0))
                best_dd = _to_float(best_metrics.get("max_drawdown_profit", 0.0))
                best_deals = _to_int(best_metrics.get("closed_deals", 0))

                out["best_reference"] = {
                    "closed_deals": best_deals,
                    "profit_factor": best_pf,
                    "net_profit": best_net,
                    "max_drawdown_profit": best_dd,
                }

                if deals >= 6 and best_deals >= 6:
                    degraded = (
                        (pf < best_pf * 0.80)
                        or (net < (best_net - max(2.0, abs(best_net) * 0.20)))
                        or (dd > max(3.0, best_dd * 1.35))
                    )
                    out["degraded_vs_best"] = bool(degraded)
                    if degraded:
                        if out["state"] == "VERDE":
                            out["state"] = "AMARILLO"
                        out["reasons"].append("degraded_vs_best")
        except Exception:
            pass

    return out


def build_report(hours: int = 24) -> dict:
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)

    activity_path = Path("data/live_activity.csv")
    paper_path = Path("data/paper_trades.csv")
    state_path = Path("models/auto_opt_state.json")
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
    performance_semaphore = _build_performance_semaphore(mt5_summary=mt5_summary, state_path=state_path)

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
        "performance_semaphore": performance_semaphore,
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
