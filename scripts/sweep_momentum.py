"""Barrido simple de hiperparámetros para `MomentumStrategy`.
Salida: guarda `data/comparison/momentum_sweep_results.csv`.
"""
from __future__ import annotations
import os
import json
import itertools
import pandas as pd
from src.config import settings
from types import SimpleNamespace
from src.models import load_artifacts
from src.feature_engineering import build_event_dataset
from src.strategies import get_strategy

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, 'data')
CMP = os.path.join(DATA, 'comparison')
os.makedirs(CMP, exist_ok=True)

# load events from monthly file (used for these backtests)
events_path = os.path.join(DATA, 'events_monthly.csv')
events = pd.read_csv(events_path, parse_dates=['date_utc'])

# load ticks if present
ticks_path = os.path.join(DATA, 'market_ticks.csv')
ticks = pd.read_csv(ticks_path, parse_dates=['time_utc']) if os.path.exists(ticks_path) else pd.DataFrame()
if not ticks.empty:
    ticks['time_utc'] = pd.to_datetime(ticks['time_utc'], utc=True)
    ticks = ticks.sort_values('time_utc')

# artifacts (tabular only)
try:
    tabular, feat_cols = load_artifacts(settings.model_dir)
    lstm = None
except Exception:
    # try tabular-only
    from src.models import load_tabular_artifacts
    tabular, feat_cols = load_tabular_artifacts(settings.model_dir)
    lstm = None

bundle = build_event_dataset(events, ticks, lookback_seconds=settings.lookback_seconds)
print('Bundle samples:', getattr(bundle.X_tabular, 'shape', None))

# grid
lookbacks = [60, 300, 1800]
thresholds = [1e-4, 1e-3, 1e-2]
weights = [0.5, 1.0, 2.0]

results = []
for lb, thr, w in itertools.product(lookbacks, thresholds, weights):
    # create a lightweight copy of settings with overrides
    base = {k: getattr(settings, k) for k in dir(settings) if not k.startswith('_') and k.isidentifier()}
    cfg = SimpleNamespace(**base)
    cfg.momentum_lookback_seconds = lb
    cfg.momentum_threshold = thr
    cfg.momentum_weight = w
    strat = get_strategy('momentum', cfg, {})
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
    import numpy as np
    if len(signals) == 0:
        metrics = {'n_signals': 0}
    else:
        df = pd.DataFrame(signals)
        df['is_correct'] = ((df['side'] == 'BUY') & (df['ret_post'] > 0)) | ((df['side'] == 'SELL') & (df['ret_post'] < 0))
        n = len(df)
        accuracy = float(df['is_correct'].mean())
        avg_ret = float(df['ret_post'].mean())
        df['pnl'] = np.where(df['side'] == 'BUY', df['ret_post'], -df['ret_post'])
        total_pnl = float(df['pnl'].sum())
        metrics = {'n_signals': n, 'accuracy': accuracy, 'avg_ret': avg_ret, 'total_pnl': total_pnl}
    results.append({'lookback': lb, 'threshold': thr, 'weight': w, **metrics})
    print('done', lb, thr, w, metrics)

out_df = pd.DataFrame(results)
out_df.to_csv(os.path.join(CMP, 'momentum_sweep_results.csv'), index=False)
print('Saved sweep to', os.path.join(CMP, 'momentum_sweep_results.csv'))
