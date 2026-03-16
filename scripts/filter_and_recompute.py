"""Filtrar señales a eventos que tienen `first_move_time` y recomputar métricas.
Guarda CSVs filtrados en `data/comparison/filtered_backtest_{name}.csv`.
"""
from __future__ import annotations
import os
import glob
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
CMP = os.path.join(ROOT, 'data', 'comparison')

for fm in glob.glob(os.path.join(CMP, 'first_move_signals_*.csv')):
    name = os.path.basename(fm).replace('first_move_signals_', '').replace('.csv', '')
    df_fm = pd.read_csv(fm, parse_dates=['first_move_time'])
    valid_ids = set(df_fm.loc[df_fm['first_move_time'].notnull(), 'event_id'].tolist())
    backtest_path = os.path.join(CMP, f'backtest_signals_{name}.csv')
    if not os.path.exists(backtest_path):
        continue
    df_bt = pd.read_csv(backtest_path)
    df_filt = df_bt[df_bt['event_id'].isin(valid_ids)].copy()
    out_path = os.path.join(CMP, f'filtered_backtest_{name}.csv')
    df_filt.to_csv(out_path, index=False)
    # compute simple metrics
    if df_filt.empty:
        print(name, 'filtered -> 0 signals')
        continue
    df_filt['is_correct'] = ((df_filt['side'] == 'BUY') & (df_filt['ret_post'] > 0)) | ((df_filt['side'] == 'SELL') & (df_filt['ret_post'] < 0))
    n = len(df_filt)
    acc = float(df_filt['is_correct'].mean())
    avg = float(df_filt['ret_post'].mean())
    df_filt['pnl'] = np.where(df_filt['side'] == 'BUY', df_filt['ret_post'], -df_filt['ret_post'])
    total = float(df_filt['pnl'].sum())
    print(name, 'filtered ->', {'n_signals': n, 'accuracy': acc, 'avg_ret': avg, 'total_pnl': total})

print('Filtered outputs written to', CMP)
