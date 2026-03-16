"""Evaluate already-generated backtest signal CSVs against bundle ret_post.

Looks for `data/backtest_signals_default.csv` and `data/backtest_signals_zscore.csv`
or in `data/comparison/`. Computes accuracy, avg post-return and simple PnL.
"""
from __future__ import annotations

import os
import pandas as pd
import numpy as np
from src.config import settings
from src.feature_engineering import build_event_dataset


def find_signal_file(name: str):
    candidates = [
        os.path.join(settings.data_dir, f'backtest_signals_{name}.csv'),
        os.path.join(settings.data_dir, 'backtest_signals.csv'),
        os.path.join(settings.data_dir, 'comparison', f'backtest_signals_{name}.csv'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def metrics(df_signals: pd.DataFrame, bundle, post_window_seconds=60):
    if df_signals is None or df_signals.empty:
        return {'n_signals': 0}
    # Recompute post returns robustly from ticks if available, otherwise fall back to bundle.ret_post
    mapping = {}
    # try to use ticks if present in module-scope (we'll attempt to import quickly)
    try:
        import pandas as _pd
        from src.config import settings as _settings
        ticks = _pd.read_csv(_settings.market_csv, parse_dates=['time_utc']) if _pd.io.common.file_exists(_settings.market_csv) else _pd.DataFrame()
        if not ticks.empty:
            ticks['time_utc'] = _pd.to_datetime(ticks['time_utc'], utc=True)
            ticks = ticks.sort_values('time_utc')
            times = ticks['time_utc']
            mids = (ticks['bid'] + ticks['ask']) / 2.0
            for i, eid in enumerate(bundle.event_ids):
                et = bundle.event_times.iloc[i]
                # first post tick at +5s, late tick at +post_window_seconds
                t1 = et + _pd.Timedelta(seconds=5)
                t2 = et + _pd.Timedelta(seconds=post_window_seconds)
                idx1 = int(times.searchsorted(t1, side='left'))
                idx2 = int(times.searchsorted(t2, side='right')) - 1
                if 0 <= idx1 < len(mids) and 0 <= idx2 < len(mids) and idx2 >= idx1:
                    p1 = float(mids.iat[idx1])
                    p2 = float(mids.iat[idx2])
                    mapping[eid] = (p2 - p1) / p1 if p1 != 0 else 0.0
                else:
                    mapping[eid] = float(bundle.ret_post[i]) if hasattr(bundle, 'ret_post') else 0.0
        else:
            for i, eid in enumerate(bundle.event_ids):
                mapping[eid] = float(bundle.ret_post[i]) if hasattr(bundle, 'ret_post') else 0.0
    except Exception:
        for i, eid in enumerate(bundle.event_ids):
            mapping[eid] = float(bundle.ret_post[i]) if hasattr(bundle, 'ret_post') else 0.0
    # Attach ret_post
    df_signals = df_signals.copy()
    df_signals['ret_post'] = df_signals['event_id'].map(mapping)
    df_signals = df_signals.dropna(subset=['ret_post'])
    if df_signals.empty:
        return {'n_signals': 0}
    df_signals['is_correct'] = ((df_signals['side'] == 'BUY') & (df_signals['ret_post'] > 0)) | ((df_signals['side'] == 'SELL') & (df_signals['ret_post'] < 0))
    n = len(df_signals)
    accuracy = float(df_signals['is_correct'].mean())
    avg_ret = float(df_signals['ret_post'].mean())
    df_signals['pnl'] = np.where(df_signals['side'] == 'BUY', df_signals['ret_post'], -df_signals['ret_post'])
    total_pnl = float(df_signals['pnl'].sum())
    return {'n_signals': n, 'accuracy': accuracy, 'avg_ret': avg_ret, 'total_pnl': total_pnl}


def main():
    events = pd.read_csv(settings.events_csv)
    ticks = pd.read_csv(settings.market_csv, parse_dates=['time_utc']) if os.path.exists(settings.market_csv) else pd.DataFrame()
    if not ticks.empty:
        ticks['time_utc'] = pd.to_datetime(ticks['time_utc'], utc=True)
        ticks = ticks.sort_values('time_utc')

    bundle = build_event_dataset(events, ticks, lookback_seconds=settings.lookback_seconds)
    print('Bundle samples:', getattr(bundle.X_tabular, 'shape', None))

    results = {}
    for name in ['default', 'zscore']:
        path = find_signal_file(name)
        if path is None:
            print('No signals file found for', name)
            results[name] = {'n_signals': 0}
            continue
        df = pd.read_csv(path)
        m = metrics(df, bundle)
        results[name] = m
        print(name, '->', m)


if __name__ == '__main__':
    main()
