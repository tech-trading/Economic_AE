"""Compare default and zscore strategies and compute basic metrics.

Outputs a small CSV per strategy and prints summary metrics.
"""
from __future__ import annotations

import os
import json
import argparse
import sys
from types import SimpleNamespace
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import settings
from src.models import load_artifacts
from src.feature_engineering import build_event_dataset
from src.strategies import get_strategy


def _compute_eval_return(ticks: pd.DataFrame, event_time: pd.Timestamp, horizon_seconds: int, min_post_ticks: int) -> float | None:
    if ticks.empty or "time_utc" not in ticks.columns:
        return None

    times = pd.to_datetime(ticks["time_utc"], utc=True, errors="coerce")
    if times.isna().all():
        return None

    t0 = pd.to_datetime(event_time, utc=True, errors="coerce")
    if pd.isna(t0):
        return None

    start_time = t0 + pd.Timedelta(seconds=5)
    end_time = t0 + pd.Timedelta(seconds=max(10, int(horizon_seconds)))

    i0 = int(times.searchsorted(start_time, side="left"))
    i1 = int(times.searchsorted(end_time, side="right")) - 1
    if i0 >= len(ticks):
        return None
    if i1 <= i0:
        i1 = min(len(ticks) - 1, i0 + 1)
        if i1 <= i0:
            return None

    span = (i1 - i0 + 1)
    if span < 2:
        return None

    min_required = max(2, int(min_post_ticks))
    if span < min_required:
        # Relax requirement on sparse ranges; keep at least a 2-point return.
        pass

    post = ticks.iloc[i0 : i1 + 1].copy()
    mid = ((post["bid"].astype(float) + post["ask"].astype(float)) / 2.0).dropna()
    if len(mid) < 2:
        return None

    first_post = float(mid.iloc[0])
    last_post = float(mid.iloc[-1])
    if first_post == 0:
        return None
    return float((last_post - first_post) / first_post)


def _bundle_slice(bundle, idx: int):
    seq = bundle.X_seq[idx: idx + 1] if hasattr(bundle, "X_seq") and len(bundle.X_seq) > idx else np.zeros((1, 1, 1), dtype=np.float32)
    tab = bundle.X_tabular.iloc[[idx]].copy() if hasattr(bundle, "X_tabular") and len(bundle.X_tabular) > idx else pd.DataFrame()
    return SimpleNamespace(X_tabular=tab, X_seq=seq)


def run_for_strategy(name: str, bundle, ticks, tabular, lstm, feat_cols, horizon_seconds: int, min_post_ticks: int):
    strat = get_strategy(name, settings, {})
    signals = []
    times = ticks['time_utc'] if not ticks.empty else None
    for i in range(bundle.X_tabular.shape[0]):
        ev_id = bundle.event_ids.iloc[i]
        event_time = pd.to_datetime(bundle.event_times.iloc[i], utc=True, errors='coerce')
        if pd.isna(event_time):
            continue
        if times is not None:
            idx = int(times.searchsorted(event_time, side='right'))
            ticks_up_to = ticks.iloc[:idx].copy()
        else:
            ticks_up_to = pd.DataFrame()
        row_bundle = _bundle_slice(bundle, i)
        dec = strat.decide(pd.Series(dtype=object), ticks_up_to, row_bundle, tabular, lstm, feat_cols, {}, settings)
        if dec is not None:
            ret_eval = _compute_eval_return(ticks, event_time, horizon_seconds=horizon_seconds, min_post_ticks=min_post_ticks)
            if ret_eval is None:
                ret_eval = float(bundle.ret_post[i]) if i < len(bundle.ret_post) else 0.0
            signals.append({'event_idx': i, 'event_id': ev_id, 'side': dec.side, 'confidence': dec.confidence, 'proba_buy': getattr(dec, 'proba_buy', 0.5), 'ret_post': float(ret_eval)})
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
    parser = argparse.ArgumentParser(description='Compare multiple strategies on a chosen dataset.')
    parser.add_argument('--events-csv', default=settings.events_csv)
    parser.add_argument('--market-csv', default=settings.market_csv)
    parser.add_argument('--horizon-minutes', type=float, default=5.0)
    parser.add_argument('--horizon-seconds', type=int, default=None)
    parser.add_argument('--min-post-ticks', type=int, default=5)
    args = parser.parse_args()

    # Backward compatible: if horizon-seconds is provided it takes precedence.
    horizon_seconds = int(args.horizon_seconds) if args.horizon_seconds is not None else int(float(args.horizon_minutes) * 60)

    events = pd.read_csv(args.events_csv)
    ticks = pd.read_csv(args.market_csv, parse_dates=['time_utc']) if os.path.exists(args.market_csv) else pd.DataFrame()
    if not ticks.empty:
        ticks['time_utc'] = pd.to_datetime(ticks['time_utc'], utc=True)
        ticks = ticks.sort_values('time_utc')

    original_events = int(len(events))
    if not ticks.empty and 'date_utc' in events.columns:
        events = events.copy()
        events['date_utc'] = pd.to_datetime(events['date_utc'], utc=True, errors='coerce')
        tmin = pd.to_datetime(ticks['time_utc'].min(), utc=True, errors='coerce')
        tmax = pd.to_datetime(ticks['time_utc'].max(), utc=True, errors='coerce')
        max_eval_time = tmax - pd.Timedelta(seconds=max(10, horizon_seconds))
        events = events[(events['date_utc'].notna()) & (events['date_utc'] >= tmin) & (events['date_utc'] <= max_eval_time)]
        events = events.reset_index(drop=True)

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
    for name in ['default', 'zscore', 'momentum', 'donchian', 'donchian_nylondon', 'ema_rsi_trend', 'turtle_atr', 'agentic_hybrid']:
        print('Running', name)
        try:
            df = run_for_strategy(name, bundle, ticks, tabular, lstm, feat_cols, horizon_seconds=horizon_seconds, min_post_ticks=int(args.min_post_ticks))
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
    with open(os.path.join(out_dir, 'summary_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(
            {
                'events_csv': str(args.events_csv),
                'market_csv': str(args.market_csv),
                'horizon_seconds': int(horizon_seconds),
                'horizon_minutes': float(horizon_seconds) / 60.0,
                'min_post_ticks': int(args.min_post_ticks),
                'events_original': int(original_events),
                'events_after_tick_range_filter': int(len(events)),
                'samples': int(getattr(bundle.X_tabular, 'shape', [0])[0]) if not bundle.X_tabular.empty else 0,
            },
            f,
            indent=2,
        )
    print('Comparison finished. Outputs in', out_dir)


if __name__ == '__main__':
    main()
