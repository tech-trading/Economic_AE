"""Construye ventanas de ticks alrededor de cada evento y calcula z-score y estadísticas.
Salida:
 - data/event_windows_summary.csv (resumen por evento)
 - data/event_windows/<event_id>.csv (ticks de la ventana)

Uso: python scripts/build_event_windows.py --before 60 --after 60 --lookback 30
"""
from __future__ import annotations
import os
import argparse
from datetime import datetime, timezone
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, 'data')
EVENTS_MONTHLY = os.path.join(DATA, 'events_monthly.csv')
TICKS_CSV = os.path.join(DATA, 'market_ticks.csv')
OUT_DIR = os.path.join(DATA, 'event_windows')
SUMMARY_CSV = os.path.join(DATA, 'event_windows_summary.csv')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--before', type=int, default=60, help='Seconds before event')
    p.add_argument('--after', type=int, default=60, help='Seconds after event')
    p.add_argument('--lookback', type=int, default=30, help='Seconds used to compute z-score (from before window)')
    p.add_argument('--resample', choices=['none','1s','1min'], default='none', help='Resample ticks to bars (optional)')
    return p.parse_args()


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def compute_zscore(mid_series: pd.Series, lookback_seconds: int) -> float:
    if mid_series is None or mid_series.empty:
        return 0.0
    # use last lookback_seconds from the series (assumes index is time_utc)
    try:
        end = mid_series.index.max()
        start = end - pd.Timedelta(seconds=lookback_seconds)
        window = mid_series.loc[start:end]
        if window.empty or window.std() == 0:
            return 0.0
        return float((mid_series.iloc[-1] - window.mean()) / window.std())
    except Exception:
        return 0.0


def summarize(events_df: pd.DataFrame, ticks_df: pd.DataFrame, before: int, after: int, lookback: int, resample: str):
    rows = []
    total = len(events_df)
    for i, r in events_df.iterrows():
        event_id = r['event_id']
        event_time = pd.to_datetime(r['date_utc'])
        if event_time.tzinfo is None:
            event_time = event_time.tz_localize('UTC')
        else:
            event_time = event_time.tz_convert('UTC')
        start = event_time - pd.Timedelta(seconds=before)
        end = event_time + pd.Timedelta(seconds=after)
        window = ticks_df[(ticks_df['time_utc'] >= start) & (ticks_df['time_utc'] <= end)].copy()
        n_ticks = len(window)
        has_ticks = n_ticks > 0
        z = 0.0
        max_move = 0.0
        min_move = 0.0
        abs_move = 0.0
        first_tick_time = None
        last_tick_time = None
        if has_ticks:
            window = window.sort_values('time_utc')
            window = window.set_index('time_utc')
            # compute mid
            window['mid'] = (window['bid'] + window['ask']) / 2.0
            # optional resample
            if resample != 'none':
                window = window['mid'].resample(resample).last().dropna().to_frame()
                window.columns = ['mid']
            mid_series = window['mid']
            z = compute_zscore(mid_series.loc[:event_time], lookback)
            first_tick_time = mid_series.index.min()
            last_tick_time = mid_series.index.max()
            # compute moves relative to price at event (or nearest)
            try:
                price_at_event = float(mid_series.iloc[-1])
                max_move = float((mid_series.max() - price_at_event) / price_at_event)
                min_move = float((mid_series.min() - price_at_event) / price_at_event)
                abs_move = float(max(abs(max_move), abs(min_move)))
            except Exception:
                max_move = min_move = abs_move = 0.0
            # save per-event ticks
            out_file = os.path.join(OUT_DIR, f"{event_id}.csv")
            # reset index to include time_utc as column
            window_reset = window.reset_index()
            window_reset.to_csv(out_file, index=False)

        rows.append({
            'event_id': event_id,
            'event_time': event_time,
            'n_ticks': int(n_ticks),
            'has_ticks': bool(has_ticks),
            'zscore': float(z),
            'max_move': float(max_move),
            'min_move': float(min_move),
            'abs_move': float(abs_move),
            'first_tick_time': pd.to_datetime(first_tick_time) if first_tick_time is not None else pd.NaT,
            'last_tick_time': pd.to_datetime(last_tick_time) if last_tick_time is not None else pd.NaT,
        })
    out_df = pd.DataFrame(rows)
    # ensure proper types and save
    out_df['event_time'] = pd.to_datetime(out_df['event_time'], utc=True)
    out_df.to_csv(SUMMARY_CSV, index=False)
    # basic summary
    total_with = int(out_df['has_ticks'].sum())
    pct = total_with / max(1, total)
    print(f"Events: {total}, with_ticks: {total_with} ({pct:.2%})")
    print(f"Saved summary to {SUMMARY_CSV} and per-event files to {OUT_DIR}")
    return out_df


def main():
    args = parse_args()
    ensure_dirs()
    if not os.path.exists(EVENTS_MONTHLY):
        raise FileNotFoundError(f"{EVENTS_MONTHLY} not found")
    if not os.path.exists(TICKS_CSV):
        raise FileNotFoundError(f"{TICKS_CSV} not found. Run fetch_ticks_mt5.py first.")

    events = pd.read_csv(EVENTS_MONTHLY)
    ticks = pd.read_csv(TICKS_CSV, parse_dates=['time_utc'])
    if 'time_utc' not in ticks.columns:
        raise ValueError('market_ticks.csv must contain time_utc column')
    ticks['time_utc'] = pd.to_datetime(ticks['time_utc'], utc=True)
    # run summarize
    summarize(events, ticks, before=args.before, after=args.after, lookback=args.lookback, resample=args.resample)


if __name__ == '__main__':
    main()
