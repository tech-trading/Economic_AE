from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import settings


@dataclass
class DatasetBundle:
    X_tabular: pd.DataFrame
    X_seq: np.ndarray
    y_direction: np.ndarray
    y_volatility: np.ndarray
    event_times: pd.Series
    event_ids: pd.Series
    ret_post: np.ndarray


def build_event_dataset(events: pd.DataFrame, market_ticks: pd.DataFrame, lookback_seconds: int = 300) -> DatasetBundle:
    if events.empty or market_ticks.empty:
        return DatasetBundle(pd.DataFrame(), np.empty((0, 0, 0)), np.array([]), np.array([]), pd.Series(dtype="datetime64[ns, UTC]"))

    ticks = market_ticks.copy()
    ticks["time_utc"] = pd.to_datetime(ticks["time_utc"], utc=True)
    ticks = ticks.sort_values("time_utc")

    feature_rows = []
    seq_rows = []
    y_volatility = []
    ret_post_values = []
    event_times = []
    event_ids = []

    for _, event in events.iterrows():
        event_time = pd.to_datetime(event["date_utc"], utc=True)

        pre = ticks[(ticks["time_utc"] < event_time) & (ticks["time_utc"] >= event_time - pd.Timedelta(seconds=lookback_seconds))]
        post = ticks[(ticks["time_utc"] >= event_time + pd.Timedelta(seconds=5)) & (ticks["time_utc"] <= event_time + pd.Timedelta(seconds=60))]

        if len(pre) < max(30, lookback_seconds // 4) or len(post) < 20:
            continue

        pre = pre.copy()
        pre["mid"] = (pre["bid"] + pre["ask"]) / 2.0
        pre["ret"] = pre["mid"].pct_change().fillna(0.0)
        pre["spread"] = pre["ask"] - pre["bid"]

        # Z-score of last pre midpoint relative to the pre window
        try:
            mid_std = float(pre["mid"].std())
            mid_mean = float(pre["mid"].mean())
            last_mid = float(pre["mid"].iloc[-1])
            pre_zscore = 0.0 if mid_std == 0 else float((last_mid - mid_mean) / mid_std)
        except Exception:
            pre_zscore = 0.0

        post = post.copy()
        post["mid"] = (post["bid"] + post["ask"]) / 2.0

        first_post = float(post["mid"].iloc[0])
        last_post = float(post["mid"].iloc[-1])
        ret_post = (last_post - first_post) / first_post
        vol_post = float(post["mid"].pct_change().fillna(0.0).std())

        y_volatility.append(1 if vol_post > pre["ret"].std() * 1.5 else 0)
        ret_post_values.append(ret_post)
        event_times.append(pd.Timestamp(event_time))
        event_id = event.get("event_id")
        event_ids.append(event_id)

        # Tabular summary features
        row = {
            "event_importance": float(event.get("importance", 0)),
            "pre_ret_mean": float(pre["ret"].mean()),
            "pre_ret_std": float(pre["ret"].std()),
            "pre_ret_abs_sum": float(pre["ret"].abs().sum()),
            "pre_spread_mean": float(pre["spread"].mean()),
            "pre_spread_std": float(pre["spread"].std()),
            "pre_zscore": float(pre_zscore),
            "pre_price_slope": _price_slope(pre["mid"].to_numpy()),
            "lookback_seconds": float(lookback_seconds),
        }
        feature_rows.append(row)

        # Sequence features [returns, spread] for LSTM.
        seq = np.stack(
            [
                pre["ret"].to_numpy(dtype=np.float32),
                pre["spread"].to_numpy(dtype=np.float32),
            ],
            axis=1,
        )
        seq_rows.append(seq)

    if not feature_rows:
        return DatasetBundle(pd.DataFrame(), np.empty((0, 0, 0)), np.array([]), np.array([]), pd.Series(dtype="datetime64[ns, UTC]"), pd.Series(dtype=object), np.array([]))

    min_len = min(seq.shape[0] for seq in seq_rows)
    X_seq = np.stack([seq[-min_len:] for seq in seq_rows], axis=0)
    X_tabular = pd.DataFrame(feature_rows)

    ret_arr = np.array(ret_post_values, dtype=np.float64)
    times_arr = pd.Series(event_times, dtype="datetime64[ns, UTC]")
    mode = (settings.direction_label_mode or "sign").strip().lower()
    if mode == "quantile_monthly":
        y_direction = np.zeros(ret_arr.shape[0], dtype=np.int32)
        month_key = times_arr.dt.tz_localize(None).dt.to_period("M").astype(str)
        for month in month_key.unique():
            idx = np.where(month_key.to_numpy() == month)[0]
            if idx.size < 2:
                y_direction[idx] = 0
                continue
            y_direction[idx] = _balanced_binary(ret_arr[idx])
    elif mode == "quantile":
        median = float(np.median(ret_arr))
        y_direction = (ret_arr >= median).astype(np.int32)
    else:
        y_direction = (ret_arr > 0.0).astype(np.int32)
        # If directional labels collapse to one class, force robust quantile labeling.
        if len(np.unique(y_direction)) < 2:
            median = float(np.median(ret_arr))
            y_direction = (ret_arr >= median).astype(np.int32)

    return DatasetBundle(
        X_tabular=X_tabular,
        X_seq=X_seq,
        y_direction=np.array(y_direction, dtype=np.int32),
        y_volatility=np.array(y_volatility, dtype=np.int32),
        event_times=pd.Series(event_times, dtype="datetime64[ns, UTC]"),
        event_ids=pd.Series(event_ids, dtype=object),
        ret_post=np.array(ret_post_values, dtype=np.float64),
    )


def _balanced_binary(values: np.ndarray) -> np.ndarray:
    n = values.size
    if n == 0:
        return np.array([], dtype=np.int32)
    if n == 1:
        return np.array([0], dtype=np.int32)

    order = np.argsort(values)
    labels = np.zeros(n, dtype=np.int32)
    labels[order[n // 2 :]] = 1
    return labels


def _price_slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    coeff = np.polyfit(x, values.astype(np.float64), deg=1)
    return float(coeff[0])
