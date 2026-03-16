"""Reconstruir dataset LSTM a 182 pasos (sin imputación). Si no hay muestras,
hacer fallback a 121 pasos con imputación y reentrenar.

Luego valida predicciones y, si todo OK, elimina entornos viejos `.venv-tf` y
`.venv311` tras verificar que `.venv` funciona.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import shutil
from train_event_lstm import build_and_train

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, 'data')
EVENT_WINDOWS = os.path.join(DATA, 'event_windows')
SUMMARY = os.path.join(DATA, 'event_windows_summary.csv')
MODEL_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def build_dataset_no_impute(before:int, after:int, resample:str, threshold:float):
    if not os.path.exists(SUMMARY):
        raise FileNotFoundError(f"{SUMMARY} missing. Run build_event_windows.py first.")
    summary = pd.read_csv(SUMMARY, parse_dates=['event_time'])
    X = []
    y = []
    ids = []
    seq_len = int(pd.Timedelta(seconds=before+after).total_seconds()) + 1
    for _, r in summary.iterrows():
        if not r['has_ticks']:
            continue
        event_id = r['event_id']
        event_time = pd.to_datetime(r['event_time'], utc=True)
        ev_file = os.path.join(EVENT_WINDOWS, f"{event_id}.csv")
        if not os.path.exists(ev_file):
            continue
        ev = pd.read_csv(ev_file, parse_dates=['time_utc'])
        if 'mid' not in ev.columns:
            continue
        s = ev.set_index('time_utc')['mid'].sort_index()
        if s.index.duplicated().any():
            s = s[~s.index.duplicated(keep='last')]
        start = event_time - pd.Timedelta(seconds=before)
        end = event_time + pd.Timedelta(seconds=after)
        idx = pd.date_range(start=start, end=end, freq=resample, tz='UTC')
        # reindex without method -> keep NaNs (no imputation)
        s_rs = s.reindex(idx)
        # skip sequences with any NaN
        if s_rs.isna().any():
            continue
        if len(s_rs) != seq_len:
            continue
        event_price = s_rs.loc[event_time] if event_time in s_rs.index else float(s_rs.iloc[before])
        post = s_rs.loc[event_time:]
        max_move = (post.max() - event_price) / event_price
        min_move = (post.min() - event_price) / event_price
        if max(abs(max_move), abs(min_move)) < threshold:
            label = 0
        else:
            label = 1 if abs(max_move) >= abs(min_move) and max_move>0 else 2
        seq = s_rs.values.astype(np.float32)
        mean = seq.mean()
        std = seq.std() if seq.std() != 0 else 1.0
        seq = (seq - mean) / std
        X.append(seq.reshape(-1,1))
        y.append(label)
        ids.append(event_id)
    if len(X) == 0:
        return None, None, None
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int32)
    return X, y, np.array(ids)


def build_dataset_with_impute(before:int, after:int, resample:str, threshold:float):
    # reuse logic similar to train_event_lstm (allows fillna)
    from train_event_lstm import build_dataset
    return build_dataset(before, after, resample, threshold)


def save_dataset(X, y, ids, name='event_dataset.npz'):
    path = os.path.join(MODEL_DIR, name)
    np.savez(path, X=X, y=y, ids=ids)
    print('Saved dataset to', path)
    return path


def validate_predictions():
    # run the existing checker which uses current .venv python
    ok = os.system(f"{os.path.join('.venv','Scripts','python.exe')} scripts/check_lstm_predictions.py")
    return ok == 0


def remove_old_envs(envs:list[str]):
    removed = []
    for e in envs:
        p = os.path.join(ROOT, e)
        if os.path.exists(p):
            try:
                print('Removing', p)
                shutil.rmtree(p)
                removed.append(p)
            except Exception as ex:
                print('Failed to remove', p, ex)
    return removed


def main():
    # try 182 steps: before=90 after=90
    print('Attempting build at 182 steps (before=90 after=90) without imputation')
    X, y, ids = build_dataset_no_impute(90, 90, '1s', 1e-4)
    if X is None:
        print('No sequences found without imputation. Falling back to 121 steps with imputation and retrain.')
        X, y, ids = build_dataset_with_impute(60, 60, '1s', 1e-4)
        if X is None:
            raise RuntimeError('Fallback dataset build produced no samples. Aborting.')
        save_dataset(X, y, ids, name='event_dataset_fallback.npz')
        print('Training model on fallback dataset (121 steps)')
        build_and_train(X, y, epochs=10, batch=32)
    else:
        print('Built dataset with', X.shape)
        save_dataset(X, y, ids, name='event_dataset_182.npz')
        print('Training model on 182-step dataset')
        build_and_train(X, y, epochs=10, batch=32)

    print('Validating predictions using check_lstm_predictions.py')
    if validate_predictions():
        print('Validation OK. Removing old virtualenvs .venv-tf and .venv311 (if exist)')
        removed = remove_old_envs(['.venv-tf', '.venv311'])
        print('Removed:', removed)
    else:
        print('Validation failed — not removing old envs.')


if __name__ == '__main__':
    main()
