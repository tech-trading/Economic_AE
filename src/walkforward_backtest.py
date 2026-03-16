from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src.config import settings
from src.feature_engineering import build_event_dataset
from src.models import ensemble_predict_proba, train_lstm, train_tabular_models
from src.policy import optimize_policy


def _simulate(y_true: np.ndarray, probs: np.ndarray, threshold: float, no_trade_band: float) -> dict[str, float]:
    trades = []
    for y, p in zip(y_true, probs):
        if abs(p - 0.5) < no_trade_band:
            continue
        if p < threshold and p > 1.0 - threshold:
            continue
        pred = 1 if p >= 0.5 else 0
        trades.append(1 if pred == y else -1)

    if not trades:
        return {
            "num_trades": 0.0,
            "hit_rate": 0.0,
            "avg_r": 0.0,
            "max_drawdown_r": 0.0,
        }

    arr = np.array(trades, dtype=np.float64)
    equity = np.cumsum(arr)
    running_max = np.maximum.accumulate(equity)
    drawdown = running_max - equity

    return {
        "num_trades": float(len(arr)),
        "hit_rate": float(np.mean(arr > 0)),
        "avg_r": float(np.mean(arr)),
        "max_drawdown_r": float(np.max(drawdown)),
    }


def main() -> None:
    events = pd.read_csv(settings.events_csv)
    market = pd.read_csv(settings.market_csv)

    bundle = build_event_dataset(events, market, lookback_seconds=settings.lookback_seconds)
    if bundle.X_tabular.empty:
        raise RuntimeError("No samples found for walk-forward backtest")

    X = bundle.X_tabular.to_numpy(dtype=np.float32)
    X_seq = bundle.X_seq
    y = bundle.y_direction
    times = pd.to_datetime(bundle.event_times, utc=True)

    periods = times.dt.tz_localize(None).dt.to_period("M").astype(str)
    unique_periods = sorted(periods.unique().tolist())
    period_label = "month"

    if settings.strict_monthly_validation and len(unique_periods) < 3:
        raise RuntimeError(
            "STRICT_MONTHLY_VALIDATION=true requires at least 3 distinct months. "
            "Run: python -m src.prepare_monthly_dataset and then set EVENTS_CSV=data/events_monthly.csv "
            "and MARKET_CSV=data/market_ticks_monthly.csv in .env"
        )

    if len(unique_periods) < 3:
        periods = times.dt.tz_localize(None).dt.to_period("W").astype(str)
        unique_periods = sorted(periods.unique().tolist())
        period_label = "week"

    use_sequential_fallback = False
    if len(unique_periods) < 3:
        use_sequential_fallback = True

    rows = []
    all_trades = []

    if use_sequential_fallback:
        n = X.shape[0]
        block = max(10, n // 4)
        split_points = [block, block * 2, block * 3]
        for i, split in enumerate(split_points, start=1):
            train_end = split
            test_end = min(split + block, n)
            if train_end < 30 or (test_end - train_end) < 10:
                continue

            train_mask = np.zeros(n, dtype=bool)
            test_mask = np.zeros(n, dtype=bool)
            train_mask[:train_end] = True
            test_mask[train_end:test_end] = True

            _append_period_result(
                rows,
                all_trades,
                f"seq_split_{i}",
                "split",
                train_mask,
                test_mask,
                X,
                X_seq,
                y,
            )
    else:
        for period in unique_periods[1:]:
            test_mask = periods == period
            train_mask = periods < period

            _append_period_result(
                rows,
                all_trades,
                period,
                period_label,
                train_mask,
                test_mask,
                X,
                X_seq,
                y,
            )

    os.makedirs(settings.model_dir, exist_ok=True)
    report_df = pd.DataFrame(rows)
    report_path = os.path.join(settings.model_dir, "walkforward_monthly_report.csv")
    report_df.to_csv(report_path, index=False)

    if not rows:
        summary = {
            "months_tested": 0,
            "total_trades": 0,
            "overall_hit_rate": 0.0,
            "overall_avg_r": 0.0,
            "overall_max_drawdown_r": 0.0,
            "note": "No valid splits (likely class imbalance per month). Collect more diverse historical windows.",
        }
        summary_path = os.path.join(settings.model_dir, "walkforward_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("Walk-forward monthly report: no valid splits.")
        print(summary)
        return

    if all_trades:
        arr = np.array(all_trades, dtype=np.float64)
        equity = np.cumsum(arr)
        running_max = np.maximum.accumulate(equity)
        drawdown = running_max - equity
        summary = {
            "months_tested": int(report_df.shape[0]),
            "total_trades": int(arr.size),
            "overall_hit_rate": float(np.mean(arr > 0)),
            "overall_avg_r": float(np.mean(arr)),
            "overall_max_drawdown_r": float(np.max(drawdown)),
        }
    else:
        summary = {
            "months_tested": int(report_df.shape[0]),
            "total_trades": 0,
            "overall_hit_rate": 0.0,
            "overall_avg_r": 0.0,
            "overall_max_drawdown_r": 0.0,
        }

    summary_path = os.path.join(settings.model_dir, "walkforward_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Walk-forward monthly report:")
    print(report_df)
    print("Walk-forward summary:")
    print(summary)


def _append_period_result(
    rows: list[dict],
    all_trades: list[int],
    period_value: str,
    period_key: str,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    X: np.ndarray,
    X_seq: np.ndarray,
    y: np.ndarray,
) -> None:
    if train_mask.sum() < 30 or test_mask.sum() < 10:
        return

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    X_seq_train, X_seq_test = X_seq[train_mask], X_seq[test_mask]

    if len(np.unique(y_train)) < 2:
        return
    if len(np.unique(y_test)) < 2:
        return

    tabular_models = train_tabular_models(X_train, y_train)
    lstm_model = train_lstm(X_seq_train, y_train, epochs=6, batch_size=32)

    probs_train = np.array(
        [ensemble_predict_proba(tabular_models, lstm_model, X_train[i], X_seq_train[i]) for i in range(X_train.shape[0])],
        dtype=np.float64,
    )
    policy = optimize_policy(y_train, probs_train)

    probs_test = np.array(
        [ensemble_predict_proba(tabular_models, lstm_model, X_test[i], X_seq_test[i]) for i in range(X_test.shape[0])],
        dtype=np.float64,
    )
    metrics = _simulate(
        y_test,
        probs_test,
        threshold=policy["decision_threshold"],
        no_trade_band=policy["no_trade_band"],
    )

    rows.append(
        {
            period_key: period_value,
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "threshold": policy["decision_threshold"],
            "no_trade_band": policy["no_trade_band"],
            **metrics,
        }
    )

    for y_i, p_i in zip(y_test, probs_test):
        if abs(p_i - 0.5) < policy["no_trade_band"]:
            continue
        if p_i < policy["decision_threshold"] and p_i > 1.0 - policy["decision_threshold"]:
            continue
        pred_i = 1 if p_i >= 0.5 else 0
        all_trades.append(1 if pred_i == y_i else -1)


if __name__ == "__main__":
    main()
