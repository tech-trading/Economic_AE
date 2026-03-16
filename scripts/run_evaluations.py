"""Run evaluate_signals.metrics for multiple post windows and print results."""
from __future__ import annotations

from scripts.evaluate_signals import metrics
from src.config import settings
from src.feature_engineering import build_event_dataset
import pandas as pd
import os

windows = [60, 300, 1800]

events = pd.read_csv(settings.events_csv)
ticks = pd.read_csv(settings.market_csv, parse_dates=['time_utc']) if os.path.exists(settings.market_csv) else pd.DataFrame()
if not ticks.empty:
    ticks['time_utc'] = pd.to_datetime(ticks['time_utc'], utc=True)
    ticks = ticks.sort_values('time_utc')

bundle = build_event_dataset(events, ticks, lookback_seconds=settings.lookback_seconds)

path_def = os.path.join(settings.data_dir, 'backtest_signals_default.csv')
path_z = os.path.join(settings.data_dir, 'backtest_signals_zscore.csv')

df_def = pd.read_csv(path_def) if os.path.exists(path_def) else pd.DataFrame()
df_z = pd.read_csv(path_z) if os.path.exists(path_z) else pd.DataFrame()

results = {}
for w in windows:
    print('--- window (s)=', w)
    m_def = metrics(df_def, bundle, post_window_seconds=w)
    m_z = metrics(df_z, bundle, post_window_seconds=w)
    print('default ->', m_def)
    print('zscore  ->', m_z)
    results[w] = {'default': m_def, 'zscore': m_z}

print('All windows done')
