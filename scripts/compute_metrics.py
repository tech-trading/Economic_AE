"""Resumen rápido de métricas para archivos de backtest y first-move.
Uso: python scripts/compute_metrics.py
"""
from __future__ import annotations
import os
import glob
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
CMP_DIR = os.path.join(ROOT, 'data', 'comparison')


def summarize_backtest(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return os.path.splitext(os.path.basename(path))[0], {'n_signals': 0}
    name = os.path.splitext(os.path.basename(path))[0]
    if df.empty:
        return name, {'n_signals': 0}
    # compute metrics similar to scripts/compare_strategies.py
    if 'side' in df.columns and 'ret_post' in df.columns:
        df['is_correct'] = ((df['side'] == 'BUY') & (df['ret_post'] > 0)) | ((df['side'] == 'SELL') & (df['ret_post'] < 0))
        n = len(df)
        accuracy = float(df['is_correct'].mean())
        avg_ret = float(df['ret_post'].mean())
        df['pnl'] = np.where(df['side'] == 'BUY', df['ret_post'], -df['ret_post'])
        total_pnl = float(df['pnl'].sum())
        return name, {'n_signals': n, 'accuracy': accuracy, 'avg_ret': avg_ret, 'total_pnl': total_pnl}
    else:
        return name, {'n_signals': len(df)}


def summarize_first_move(path):
    df = pd.read_csv(path, parse_dates=['first_move_time']) if os.path.exists(path) else pd.DataFrame()
    name = os.path.splitext(os.path.basename(path))[0]
    if df.empty:
        return name, {'n_signals': 0}
    n = len(df)
    has_first_move = df['first_move_time'].notnull().sum()
    prop_first_move = float(has_first_move / n)
    mean_ret = float(df['ret_first_move'].mean()) if 'ret_first_move' in df.columns else 0.0
    total_pnl = float(df['pnl'].sum()) if 'pnl' in df.columns else float((df['ret_first_move']).sum()) if 'ret_first_move' in df.columns else 0.0
    return name, {'n_signals': n, 'has_first_move': int(has_first_move), 'prop_first_move': prop_first_move, 'mean_ret_first_move': mean_ret, 'total_pnl_first_move': total_pnl}


def main():
    out = {}
    for f in glob.glob(os.path.join(CMP_DIR, 'backtest_signals_*.csv')):
        k, v = summarize_backtest(f)
        out[k] = v
    for f in glob.glob(os.path.join(CMP_DIR, 'first_move_signals_*.csv')):
        k, v = summarize_first_move(f)
        out[k] = v
    print('Summary:')
    for k, v in out.items():
        print('-', k, ':', v)

if __name__ == '__main__':
    main()
