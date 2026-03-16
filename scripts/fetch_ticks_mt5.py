"""Descarga ticks desde MetaTrader5 para los eventos en `data/events.csv`.

Requisitos: MetaTrader Terminal abierto y cuenta con histórico; paquete `MetaTrader5`.

Uso ejemplo:
  python scripts/fetch_ticks_mt5.py --symbols EURUSD,GBPUSD

Si no se pasan fechas, usa el rango de `data/events.csv` (`date_utc`).
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone, timedelta
import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
EVENTS_CSV = os.path.join(DATA, "events.csv")
OUT_CSV = os.path.join(DATA, "market_ticks.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="EURUSD", help="Comma-separated symbols to fetch (e.g. EURUSD,GBPUSD)")
    p.add_argument("--start", help="UTC start datetime YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")
    p.add_argument("--end", help="UTC end datetime")
    p.add_argument("--pad-days", type=int, default=1, help="Pad event range by N days")
    p.add_argument("--append", action="store_true", help="Append to existing market_ticks.csv instead of overwrite")
    return p.parse_args()


def infer_range_from_events(pad_days: int = 1):
    if not os.path.exists(EVENTS_CSV):
        raise FileNotFoundError("events.csv not found in data/")
    ev = pd.read_csv(EVENTS_CSV, parse_dates=["date_utc"]) if "date_utc" in pd.read_csv(EVENTS_CSV, nrows=0).columns else pd.read_csv(EVENTS_CSV)
    if "date_utc" not in ev.columns:
        raise ValueError("events.csv has no 'date_utc' column")
    start = ev["date_utc"].min()
    end = ev["date_utc"].max()
    # ensure timezone-aware UTC
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    if start.tzinfo is None:
        start = start.tz_localize('UTC')
    else:
        start = start.tz_convert('UTC')
    if end.tzinfo is None:
        end = end.tz_localize('UTC')
    else:
        end = end.tz_convert('UTC')
    start = (start - pd.Timedelta(days=pad_days)).to_pydatetime()
    end = (end + pd.Timedelta(days=pad_days)).to_pydatetime()
    return start, end


def fetch_ticks_for_symbol(symbol: str, dt_from: datetime, dt_to: datetime):
    # dt_from/dt_to should be timezone-aware UTC datetimes
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not available. Install 'MetaTrader5' and ensure terminal is running.")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    # MetaTrader5 copy_ticks_range expects naive datetimes in local timezone; convert to UTC naive
    frm = dt_from.replace(tzinfo=timezone.utc).astimezone(tz=timezone.utc).replace(tzinfo=None)
    to = dt_to.replace(tzinfo=timezone.utc).astimezone(tz=timezone.utc).replace(tzinfo=None)

    ticks = mt5.copy_ticks_range(symbol, frm, to, mt5.COPY_TICKS_ALL)
    if ticks is None:
        return pd.DataFrame()
    df = pd.DataFrame(ticks)
    if df.empty:
        return df
    # mt5 returns 'time' as seconds since epoch
    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    # ensure bid/ask present
    for col in ("bid", "ask"):
        if col not in df.columns:
            df[col] = pd.NA
    df = df[["time_utc", "bid", "ask"]].copy()
    df["symbol"] = symbol
    return df


def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if args.start and args.end:
        start = pd.to_datetime(args.start)
        end = pd.to_datetime(args.end)
        if start.tzinfo is None:
            start = start.tz_localize('UTC')
        else:
            start = start.tz_convert('UTC')
        if end.tzinfo is None:
            end = end.tz_localize('UTC')
        else:
            end = end.tz_convert('UTC')
        start = start.to_pydatetime()
        end = end.to_pydatetime()
    else:
        start, end = infer_range_from_events(args.pad_days)

    print(f"Fetching ticks for {symbols} from {start} to {end}")
    all_dfs = []
    for sym in symbols:
        print("Fetching", sym)
        df = fetch_ticks_for_symbol(sym, start, end)
        print(" -> ticks fetched:", len(df))
        if df is not None and not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("No ticks fetched for any symbol")
        return
    out = pd.concat(all_dfs, ignore_index=True)
    out = out.sort_values(["time_utc", "symbol"]).reset_index(drop=True)

    if args.append and os.path.exists(OUT_CSV):
        existing = pd.read_csv(OUT_CSV, parse_dates=["time_utc"]) if os.path.exists(OUT_CSV) else pd.DataFrame()
        if not existing.empty:
            out = pd.concat([existing, out], ignore_index=True).drop_duplicates(subset=["time_utc", "symbol"]).sort_values("time_utc")

    out.to_csv(OUT_CSV, index=False)
    print("Saved ticks to", OUT_CSV)


if __name__ == "__main__":
    main()
