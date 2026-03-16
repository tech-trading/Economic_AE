"""Evaluate signals using return until the first mid movement after event.

For each signal in CSVs, finds the last mid before the event and the first tick
after event+5s where mid differs from that reference. Computes return and
aggregates metrics per strategy. Saves per-strategy CSVs and a JSON summary.

Usage: run from repo root with PYTHONPATH='.' and virtualenv active.
"""
from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np

from src.config import settings
from src.feature_engineering import build_event_dataset


def load_signals(name: str):
    candidates = [
        os.path.join(settings.data_dir, f"backtest_signals_{name}.csv"),
        os.path.join(settings.data_dir, "backtest_signals.csv"),
        os.path.join(settings.data_dir, "comparison", f"backtest_signals_{name}.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return pd.read_csv(c)
    return pd.DataFrame()


def compute_first_move_returns(signals: pd.DataFrame, events_df: pd.DataFrame, ticks: pd.DataFrame, max_window_seconds: int = 7200):
    if signals.empty:
        return pd.DataFrame(), {}

    # build mapping from event_id to event_time
    bundle = build_event_dataset(events_df, ticks, lookback_seconds=settings.lookback_seconds)
    event_time_map = {eid: t for eid, t in zip(bundle.event_ids.tolist(), bundle.event_times.tolist())}

    # prepare ticks
    if ticks.empty:
        return pd.DataFrame(), {}
    ticks = ticks.sort_values("time_utc").reset_index(drop=True)
    times = ticks["time_utc"]
    mids = ((ticks["bid"] + ticks["ask"]) / 2.0).to_numpy()

    out_rows = []
    for _, row in signals.iterrows():
        eid = row.get("event_id")
        if eid not in event_time_map:
            continue
        et = event_time_map[eid]
        # index of last tick before event
        idx_event = int(times.searchsorted(et, side="left"))
        pre_idx = idx_event - 1
        if pre_idx < 0 or pre_idx >= len(mids):
            ref_mid = mids[idx_event] if idx_event < len(mids) else None
        else:
            ref_mid = mids[pre_idx]

        if ref_mid is None:
            ret = 0.0
            first_move_time = None
        else:
            # find first tick at or after et+5s where mid != ref_mid
            t_start = et + pd.Timedelta(seconds=5)
            idx_start = int(times.searchsorted(t_start, side="left"))
            idx_end = int(times.searchsorted(et + pd.Timedelta(seconds=max_window_seconds), side="right")) - 1
            if idx_start < 0:
                idx_start = 0
            if idx_end >= len(mids):
                idx_end = len(mids) - 1

            segment = mids[idx_start: idx_end + 1]
            if segment.size == 0:
                ret = 0.0
                first_move_time = None
            else:
                # vectorized check for first different mid
                neq = (segment != ref_mid)
                if not neq.any():
                    ret = 0.0
                    first_move_time = None
                else:
                    rel_idx = int(np.argmax(neq))
                    idx_found = idx_start + rel_idx
                    p_new = mids[idx_found]
                    ret = (p_new - ref_mid) / ref_mid if ref_mid != 0 else 0.0
                    first_move_time = times.iat[idx_found]

        out = dict(row)
        out["first_move_time"] = str(first_move_time) if first_move_time is not None else None
        out["ret_first_move"] = float(ret)
        out_rows.append(out)

    df_out = pd.DataFrame(out_rows)
    # metrics
    if df_out.empty:
        metrics = {"n_signals": 0}
    else:
        df_out["is_correct"] = ((df_out["side"] == "BUY") & (df_out["ret_first_move"] > 0)) | ((df_out["side"] == "SELL") & (df_out["ret_first_move"] < 0))
        df_out["pnl"] = np.where(df_out["side"] == "BUY", df_out["ret_first_move"], -df_out["ret_first_move"])
        metrics = {
            "n_signals": int(len(df_out)),
            "accuracy": float(df_out["is_correct"].mean()),
            "avg_ret": float(df_out["ret_first_move"].mean()),
            "total_pnl": float(df_out["pnl"].sum()),
        }

    return df_out, metrics


def main():
    events = pd.read_csv(settings.events_csv)
    ticks = pd.read_csv(settings.market_csv, parse_dates=["time_utc"]) if os.path.exists(settings.market_csv) else pd.DataFrame()
    if not ticks.empty:
        ticks["time_utc"] = pd.to_datetime(ticks["time_utc"], utc=True)

    out_dir = os.path.join(settings.data_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for name in ["default", "zscore"]:
        sigs = load_signals(name)
        df_out, metrics = compute_first_move_returns(sigs, events, ticks, max_window_seconds=7200)
        out_path = os.path.join(out_dir, f"first_move_signals_{name}.csv")
        df_out.to_csv(out_path, index=False)
        results[name] = metrics
        print(name, metrics)

    with open(os.path.join(out_dir, "first_move_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Done. Outputs in", out_dir)


if __name__ == "__main__":
    main()
