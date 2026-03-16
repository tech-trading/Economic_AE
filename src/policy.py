from __future__ import annotations

import json
import os

import numpy as np


def optimize_policy(y_true: np.ndarray, proba_buy: np.ndarray) -> dict[str, float]:
    best = {
        "decision_threshold": 0.60,
        "no_trade_band": 0.05,
        "score": -1e9,
        "num_trades": 0.0,
    }

    thresholds = np.arange(0.52, 0.81, 0.02)
    bands = np.arange(0.00, 0.13, 0.01)

    for threshold in thresholds:
        for band in bands:
            metrics = _simulate(y_true, proba_buy, threshold=threshold, no_trade_band=band)
            # Optimize for expected R with drawdown penalty.
            score = metrics["avg_r"] - 0.03 * metrics["max_drawdown_r"]
            if metrics["num_trades"] < 10:
                continue
            if score > best["score"]:
                best = {
                    "decision_threshold": float(threshold),
                    "no_trade_band": float(band),
                    "score": float(score),
                    "num_trades": float(metrics["num_trades"]),
                }

    return best


def _simulate(y_true: np.ndarray, proba_buy: np.ndarray, threshold: float, no_trade_band: float) -> dict[str, float]:
    trades = []
    for y, p in zip(y_true, proba_buy):
        if abs(p - 0.5) < no_trade_band:
            continue
        if p < threshold and p > 1.0 - threshold:
            continue

        pred = 1 if p >= 0.5 else 0
        trades.append(1 if pred == y else -1)

    if not trades:
        return {
            "num_trades": 0.0,
            "avg_r": 0.0,
            "max_drawdown_r": 0.0,
        }

    trades_np = np.array(trades, dtype=np.float64)
    equity = np.cumsum(trades_np)
    running_max = np.maximum.accumulate(equity)
    drawdown = running_max - equity

    return {
        "num_trades": float(len(trades_np)),
        "avg_r": float(np.mean(trades_np)),
        "max_drawdown_r": float(np.max(drawdown)),
    }


def save_policy(model_dir: str, policy: dict[str, float]) -> None:
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "trading_policy.json"), "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)


def load_policy(model_dir: str, default_threshold: float, default_no_trade_band: float) -> dict[str, float]:
    path = os.path.join(model_dir, "trading_policy.json")
    if not os.path.exists(path):
        return {
            "decision_threshold": float(default_threshold),
            "no_trade_band": float(default_no_trade_band),
        }

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return {
        "decision_threshold": float(raw.get("decision_threshold", default_threshold)),
        "no_trade_band": float(raw.get("no_trade_band", default_no_trade_band)),
    }
