from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.calendar_sources import fetch_and_store_events
from src.config import settings
from src.data_collection import collect_mt5_history_bars, init_mt5, shutdown_mt5


def seconds_until_next_midnight_local(now_utc: datetime | None = None) -> int:
    now = now_utc or datetime.now(timezone.utc)
    now_local = now.astimezone(settings.local_tz)
    next_midnight_local = datetime.combine(
        now_local.date() + timedelta(days=1),
        datetime.min.time(),
        tzinfo=settings.local_tz,
    )
    return int((next_midnight_local - now_local).total_seconds())


def run_daily_snapshot() -> None:
    events = fetch_and_store_events(days_ahead=1)
    print(f"[daily_jobs] Eventos guardados para hoy: {len(events)}")

    init_mt5()
    try:
        bars = collect_mt5_history_bars(days_back=365, out_csv="data/mt5_history_m1.csv")
        print(f"[daily_jobs] Barras historicas MT5 actualizadas: {len(bars)}")
    finally:
        shutdown_mt5()

    report = build_daily_live_report(hours=24)
    print(f"[daily_jobs] Reporte diario generado: {json.dumps(report, ensure_ascii=True)}")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_daily_live_report(hours: int = 24) -> dict:
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)

    activity_path = Path(settings.live_activity_csv)
    paper_path = Path(settings.data_dir) / "paper_trades.csv"
    out_path = Path(settings.model_dir) / "daily_live_report.json"

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

    report = {
        "generated_at_utc": now.isoformat(),
        "window_hours": int(hours),
        "since_utc": since.isoformat(),
        "activity": {
            "rows": int(len(activity_recent)),
            "actions": actions,
        },
        "paper": {
            "signals": int(len(paper_recent)),
            "buy_signals": int((paper_recent.get("side", pd.Series(dtype=str)).astype(str).str.upper() == "BUY").sum()) if not paper_recent.empty else 0,
            "sell_signals": int((paper_recent.get("side", pd.Series(dtype=str)).astype(str).str.upper() == "SELL").sum()) if not paper_recent.empty else 0,
            "avg_confidence": float(pd.to_numeric(paper_recent.get("confidence", pd.Series(dtype=float)), errors="coerce").mean()) if not paper_recent.empty else 0.0,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_scheduler_forever() -> None:
    print(f"[daily_jobs] Scheduler iniciado. Ejecuta diariamente a las 00:00 (UTC{settings.utc_offset_hours:+d}).")
    while True:
        wait_s = seconds_until_next_midnight_local()
        print(f"[daily_jobs] Esperando {wait_s} segundos hasta el proximo corte diario...")
        time.sleep(max(wait_s, 1))

        try:
            run_daily_snapshot()
        except Exception as ex:
            print(f"[daily_jobs] Error en ejecucion diaria: {ex}")

        # Delay corto para evitar doble ejecución por borde de reloj.
        time.sleep(2)


def main() -> None:
    run_scheduler_forever()


if __name__ == "__main__":
    main()
