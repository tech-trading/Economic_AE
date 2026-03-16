"""Compare default and zscore strategies and compute basic metrics.

Outputs a small CSV per strategy and prints summary metrics.
"""
from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np

from src.config import settings
from src.models import load_artifacts
from src.feature_engineering import build_event_dataset
from src.strategies import get_strategy


def run_for_strategy(name: str, bundle, ticks, tabular, lstm, feat_cols):
    strat = get_strategy(name, settings, {})
    signals = []
    times = ticks['time_utc'] if not ticks.empty else None
    for i in range(bundle.X_tabular.shape[0]):
        ev_id = bundle.event_ids.iloc[i]
        event_time = bundle.event_times.iloc[i]
        if times is not None:
            idx = int(times.searchsorted(event_time, side='right'))
            ticks_up_to = ticks.iloc[:idx].copy()
        else:
            ticks_up_to = pd.DataFrame()
        dec = strat.decide(pd.Series(dtype=object), ticks_up_to, bundle, tabular, lstm, feat_cols, {}, settings)
        if dec is not None:
            signals.append({'event_idx': i, 'event_id': ev_id, 'side': dec.side, 'confidence': dec.confidence, 'proba_buy': getattr(dec, 'proba_buy', 0.5), 'ret_post': float(bundle.ret_post[i])})
    df = pd.DataFrame(signals)
    return df


def metrics_from_signals(df: pd.DataFrame):
    if df.empty:
        return {'n_signals': 0}
    # Directional accuracy: BUY matches positive ret_post
    df['is_correct'] = ((df['side'] == 'BUY') & (df['ret_post'] > 0)) | ((df['side'] == 'SELL') & (df['ret_post'] < 0))
    n = len(df)
    accuracy = float(df['is_correct'].mean())
    avg_ret = float(df['ret_post'].mean())
    # simple PnL: BUY -> ret_post, SELL -> -ret_post
    df['pnl'] = np.where(df['side'] == 'BUY', df['ret_post'], -df['ret_post'])
    total_pnl = float(df['pnl'].sum())
    return {'n_signals': n, 'accuracy': accuracy, 'avg_ret': avg_ret, 'total_pnl': total_pnl}


def main():
    events = pd.read_csv(settings.events_csv)
    ticks = pd.read_csv(settings.market_csv, parse_dates=['time_utc']) if os.path.exists(settings.market_csv) else pd.DataFrame()
    if not ticks.empty:
        ticks['time_utc'] = pd.to_datetime(ticks['time_utc'], utc=True)
        ticks = ticks.sort_values('time_utc')

    # load only tabular artifacts to avoid importing tensorflow in constrained envs
    try:
        from src.models import load_tabular_artifacts

        tabular, feat_cols = load_tabular_artifacts(settings.model_dir)
        lstm = None
    except Exception:
        # fallback to full loader
        tabular, lstm, feat_cols = load_artifacts(settings.model_dir)
    bundle = build_event_dataset(events, ticks, lookback_seconds=settings.lookback_seconds)
    print('Built bundle samples:', getattr(bundle.X_tabular, 'shape', None))

    out_dir = os.path.join(settings.data_dir, 'comparison')
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for name in ['default', 'zscore', 'momentum']:
        print('Running', name)
        try:
            df = run_for_strategy(name, bundle, ticks, tabular, lstm, feat_cols)
        except Exception as e:
            print('Failed to run strategy', name, 'reason:', e)
            df = pd.DataFrame()
        out_path = os.path.join(out_dir, f'backtest_signals_{name}.csv')
        df.to_csv(out_path, index=False)
        m = metrics_from_signals(df) if not df.empty else {'n_signals': 0}
        results[name] = m
        print(name, m)

    # Save summary
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print('Comparison finished. Outputs in', out_dir)


if __name__ == '__main__':
    main()
