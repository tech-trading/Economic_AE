from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

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
