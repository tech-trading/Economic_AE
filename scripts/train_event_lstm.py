"""Preparar dataset de ventanas por evento y entrenar un LSTM simple.
Salida:
 - models/event_lstm.h5
 - models/event_dataset.npz (X, y, ids)

Uso ejemplo:
  python scripts/train_event_lstm.py --before 60 --after 60 --resample 1s --threshold 1e-4 --epochs 10
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, 'data')
EVENT_WINDOWS = os.path.join(DATA, 'event_windows')
SUMMARY = os.path.join(DATA, 'event_windows_summary.csv')
MODEL_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--before', type=int, default=60)
    p.add_argument('--after', type=int, default=60)
    p.add_argument('--resample', default='1s', help='Pandas resample frequency (e.g., 1s, 1min)')
    p.add_argument('--threshold', type=float, default=1e-4, help='Label threshold for movement')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=32)
    return p.parse_args()


def build_dataset(before:int, after:int, resample:str, threshold:float):
    if not os.path.exists(SUMMARY):
        raise FileNotFoundError(f"{SUMMARY} missing. Run build_event_windows.py first.")
    summary = pd.read_csv(SUMMARY, parse_dates=['event_time'])
    rows = []
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
        # remove duplicate timestamps by keeping last observation
        if s.index.duplicated().any():
            s = s[~s.index.duplicated(keep='last')]
        # create uniform index
        start = event_time - pd.Timedelta(seconds=before)
        end = event_time + pd.Timedelta(seconds=after)
        idx = pd.date_range(start=start, end=end, freq=resample, tz='UTC')
        s_rs = s.reindex(idx, method='ffill')
        # if still NaNs, fill with nearest
        s_rs = s_rs.fillna(method='bfill').fillna(method='ffill')
        if s_rs.isna().any():
            continue
        if len(s_rs) != seq_len:
            # skip inconsistent length
            continue
        # label: compare post-window (event_time..end)
        event_price = s_rs.loc[event_time] if event_time in s_rs.index else float(s_rs.iloc[before])
        post = s_rs.loc[event_time:]
        max_move = (post.max() - event_price) / event_price
        min_move = (post.min() - event_price) / event_price
        if max(abs(max_move), abs(min_move)) < threshold:
            label = 0  # NO_MOVE
        else:
            label = 1 if abs(max_move) >= abs(min_move) and max_move>0 else 2
        seq = s_rs.values.astype(np.float32)
        # normalize per-sequence (z-score)
        mean = seq.mean()
        std = seq.std() if seq.std() != 0 else 1.0
        seq = (seq - mean) / std
        X.append(seq.reshape(-1,1))
        y.append(label)
        ids.append(event_id)
    if len(X) == 0:
        raise RuntimeError('No sequences built (check coverage/resample).')
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int32)
    return X, y, np.array(ids)


def build_and_train(X, y, epochs:int, batch:int):
    try:
        # Optional dependency: installed in the project .venv.
        import tensorflow as tf  # pyright: ignore[reportMissingModuleSource]
    except Exception:
        raise RuntimeError('TensorFlow not available in this environment.')

    n_classes = len(np.unique(y))
    input_shape = X.shape[1:]  # seq_len, features
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch)
    # eval
    preds = model.predict(X_val)
    yhat = np.argmax(preds, axis=1)
    print(classification_report(y_val, yhat))
    # save
    model_path = os.path.join(MODEL_DIR, 'event_lstm.h5')
    model.save(model_path)
    print('Saved model to', model_path)
    return model, history


def main():
    args = parse_args()
    print('Building dataset...')
    X, y, ids = build_dataset(args.before, args.after, args.resample, args.threshold)
    print('Built', X.shape, 'labels distribution:', dict(zip(*np.unique(y, return_counts=True))))
    # save dataset
    np.savez(os.path.join(MODEL_DIR, 'event_dataset.npz'), X=X, y=y, ids=ids)
    print('Saved dataset to models/event_dataset.npz')
    print('Training LSTM...')
    model, hist = build_and_train(X, y, epochs=args.epochs, batch=args.batch)


if __name__ == '__main__':
    main()
