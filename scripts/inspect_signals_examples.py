"""Imprime 5 señales (zscore) y ticks +/-60s alrededor del evento para inspección.
"""
from __future__ import annotations

import os
import sys
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
COMPARE_DIR = os.path.join(DATA_DIR, "comparison")


def load_signals():
    candidates = [
        os.path.join(COMPARE_DIR, "first_move_signals_zscore.csv"),
        os.path.join(DATA_DIR, "backtest_signals_zscore.csv"),
        os.path.join(COMPARE_DIR, "backtest_signals_zscore.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return pd.read_csv(c)
    return pd.DataFrame()


def load_ticks():
    tpath = os.path.join(DATA_DIR, "market_ticks.csv")
    if not os.path.exists(tpath):
        return pd.DataFrame()
    df = pd.read_csv(tpath, parse_dates=["time_utc"]) 
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    return df.sort_values("time_utc").reset_index(drop=True)


def find_event_time(row):
    # possible columns: event_time, time_utc, event_timestamp
    for k in ("event_time", "time_utc", "event_timestamp", "time"):
        if k in row and pd.notna(row[k]):
            try:
                return pd.to_datetime(row[k], utc=True)
            except Exception:
                pass
    # fallback: try event_id mapping not available here
    return None


def main():
    sigs = load_signals()
    ticks = load_ticks()

    if sigs.empty:
        print("No se encontraron señales zscore en data/comparison o data/")
        sys.exit(0)

    if ticks.empty:
        print("No se encontraron ticks en data/market_ticks.csv")

    n = min(5, len(sigs))
    print(f"Mostrando {n} señales zscore (primeras filas) y ticks +/-60s alrededor\n")

    for i in range(n):
        row = sigs.iloc[i]
        print("--- Signal", i+1, "---")
        print(row.to_dict())
        et = find_event_time(row)
        if et is None:
            print("  (No pude determinar event_time para esta señal)\n")
            continue

        if ticks.empty:
            print("  (No hay ticks para mostrar)\n")
            continue

        start = et - pd.Timedelta(seconds=60)
        end = et + pd.Timedelta(seconds=60)
        seg = ticks[(ticks["time_utc"] >= start) & (ticks["time_utc"] <= end)].copy()
        seg["mid"] = (seg["bid"] + seg["ask"]) / 2.0
        print(f"  event_time: {et}  ticks in window: {len(seg)}")
        if len(seg) > 0:
            print(seg.head(20).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
