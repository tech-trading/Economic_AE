"""Diagnostic helper for backtest bundles and model outputs."""
from __future__ import annotations

import sys
sys.path.insert(0, '.')
from src.config import settings
import pandas as pd
from src.models import load_artifacts, ensemble_predict_proba
from src.feature_engineering import build_event_dataset

print('events csv:', settings.events_csv)
print('market csv:', settings.market_csv)

events = pd.read_csv(settings.events_csv)
try:
    ticks = pd.read_csv(settings.market_csv, parse_dates=['time_utc'])
    ticks['time_utc'] = pd.to_datetime(ticks['time_utc'], utc=True)
except Exception as e:
    print('failed to load ticks:', e)
    ticks = pd.DataFrame()

print('events:', len(events), 'ticks:', len(ticks))

tabular, lstm, feat = load_artifacts(settings.model_dir)
print('loaded models, features count:', len(feat))

bundle = build_event_dataset(events, ticks, lookback_seconds=settings.lookback_seconds)
print('bundle X_tabular shape:', getattr(bundle.X_tabular, 'shape', None))

if not bundle.X_tabular.empty:
    for i in range(min(10, bundle.X_tabular.shape[0])):
        x_row = bundle.X_tabular.iloc[i].reindex(feat, fill_value=0.0)
        proba = ensemble_predict_proba(tabular, lstm, x_row.to_numpy(dtype='float32'), bundle.X_seq[i])
        conf = max(proba, 1.0-proba)
        print(f'sample {i}: proba_buy={proba:.4f}, confidence={conf:.4f}, pre_zscore={x_row.get("pre_zscore")}')
else:
    print('bundle empty or no samples')
