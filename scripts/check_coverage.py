"""Comprueba cobertura temporal entre events.csv y market_ticks.csv.

Imprime rangos min/max, conteos y porcentaje de eventos con al menos
un tick en +/-60s (configurable).
"""
from __future__ import annotations

import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
EVENTS = os.path.join(DATA, "events.csv")
TICKS = os.path.join(DATA, "market_ticks.csv")


def main(window_seconds: int = 60):
    if not os.path.exists(EVENTS):
        print("No existe", EVENTS)
        return
    if not os.path.exists(TICKS):
        print("No existe", TICKS)
        return

    events = pd.read_csv(EVENTS)
    ticks = pd.read_csv(TICKS, parse_dates=["time_utc"]) if os.path.exists(TICKS) else pd.DataFrame()

    # normalize event times if present
    time_cols = [c for c in events.columns if "time" in c.lower()]
    if not time_cols:
        print("No hay columna de tiempo en events.csv")
        return
    et_col = time_cols[0]
    events[et_col] = pd.to_datetime(events[et_col], utc=True)

    ticks["time_utc"] = pd.to_datetime(ticks["time_utc"], utc=True)

    print(f"events: {len(events)} rows, time col: {et_col}")
    print(f"ticks: {len(ticks)} rows")
    print("events time range:", events[et_col].min(), "->", events[et_col].max())
    print("ticks time range:", ticks["time_utc"].min(), "->", ticks["time_utc"].max())

    # fraction of events with at least one tick in +/-window
    ticks = ticks.sort_values("time_utc").reset_index(drop=True)
    times = ticks["time_utc"]

    has_tick = 0
    missing_events = []
    for _, r in events.iterrows():
        et = r[et_col]
        start = et - pd.Timedelta(seconds=window_seconds)
        end = et + pd.Timedelta(seconds=window_seconds)
        # searchsorted
        i0 = int(times.searchsorted(start, side="left"))
        i1 = int(times.searchsorted(end, side="right"))
        if i1 - i0 > 0:
            has_tick += 1
        else:
            missing_events.append(et)

    pct = 100.0 * has_tick / len(events) if len(events) else 0.0
    print(f"Events with >=1 tick within +/-{window_seconds}s: {has_tick}/{len(events)} ({pct:.2f}%)")
    if missing_events:
        print("First 5 events without ticks:")
        for t in missing_events[:5]:
            print(" ", t)


if __name__ == "__main__":
    main(window_seconds=60)
