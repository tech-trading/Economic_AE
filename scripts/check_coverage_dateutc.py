"""Check tick coverage around events using `date_utc` as event time.

Prints number and fraction of events with >=1 tick within +/- window seconds.
"""
from __future__ import annotations

import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
EVENTS = os.path.join(DATA, "events.csv")
TICKS = os.path.join(DATA, "market_ticks.csv")


def coverage(window_seconds: int):
    events = pd.read_csv(EVENTS, parse_dates=["date_utc"]) if os.path.exists(EVENTS) else pd.DataFrame()
    ticks = pd.read_csv(TICKS, parse_dates=["time_utc"]) if os.path.exists(TICKS) else pd.DataFrame()
    if events.empty or ticks.empty:
        print("Missing events or ticks file")
        return
    ticks = ticks.sort_values("time_utc").reset_index(drop=True)
    times = ticks["time_utc"]
    has = 0
    for et in events["date_utc"]:
        i0 = int(times.searchsorted(et - pd.Timedelta(seconds=window_seconds), side="left"))
        i1 = int(times.searchsorted(et + pd.Timedelta(seconds=window_seconds), side="right"))
        if i1 - i0 > 0:
            has += 1
    n = len(events)
    print(f"window={window_seconds}s: {has}/{n} events have >=1 tick ({100*has/n:.2f}%)")


def main():
    for w in (60, 3600, 86400):
        coverage(w)


if __name__ == "__main__":
    main()
