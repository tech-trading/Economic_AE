from __future__ import annotations

import json
import os

import numpy as np


def optimize_policy(
    y_true: np.ndarray,
    proba_buy: np.ndarray,
    *,
    cost_per_trade_r: float = 0.0,
    spread_sensitivity: float = 0.15,
    drawdown_penalty: float = 0.08,
    loss_streak_penalty: float = 0.03,
    min_trades: int = 10,
) -> dict[str, float]:
    best = {
        "decision_threshold": 0.60,
        "no_trade_band": 0.05,
        "score": -1e9,
        "num_trades": 0.0,
        "avg_r_net": 0.0,
        "max_drawdown_r_net": 0.0,
        "max_loss_streak": 0.0,
        "profit_factor_net": 0.0,
    }

    thresholds = np.arange(0.52, 0.81, 0.02)
    bands = np.arange(0.00, 0.13, 0.01)

    for threshold in thresholds:
        for band in bands:
            metrics = simulate_policy(
                y_true,
                proba_buy,
                threshold=float(threshold),
                no_trade_band=float(band),
                cost_per_trade_r=float(cost_per_trade_r),
                spread_sensitivity=float(spread_sensitivity),
            )
            score = (
                float(metrics["avg_r_net"])
                - float(drawdown_penalty) * float(metrics["max_drawdown_r_net"])
                - float(loss_streak_penalty) * float(metrics["max_loss_streak"])
            )
            if metrics["num_trades"] < max(1, int(min_trades)):
                continue
            if score > best["score"]:
                best = {
                    "decision_threshold": float(threshold),
                    "no_trade_band": float(band),
                    "score": float(score),
                    "num_trades": float(metrics["num_trades"]),
                    "avg_r_net": float(metrics["avg_r_net"]),
                    "max_drawdown_r_net": float(metrics["max_drawdown_r_net"]),
                    "max_loss_streak": float(metrics["max_loss_streak"]),
                    "profit_factor_net": float(metrics["profit_factor_net"]),
                }

    return best


def simulate_policy(
    y_true: np.ndarray,
    proba_buy: np.ndarray,
    *,
    threshold: float,
    no_trade_band: float,
    cost_per_trade_r: float = 0.0,
    spread_sensitivity: float = 0.15,
) -> dict[str, float]:
    gross_trades: list[float] = []
    net_trades: list[float] = []
    skipped_band = 0
    skipped_threshold = 0

    for y, p in zip(y_true, proba_buy):
        if abs(p - 0.5) < no_trade_band:
            skipped_band += 1
            continue
        if p < threshold and p > 1.0 - threshold:
            skipped_threshold += 1
            continue

        pred = 1 if p >= 0.5 else 0
        gross_r = 1.0 if pred == y else -1.0
        edge = float(abs(p - 0.5) * 2.0)
        spread_penalty = float(max(0.0, 1.0 - edge)) * float(spread_sensitivity)
        net_r = gross_r - float(cost_per_trade_r) - spread_penalty

        gross_trades.append(gross_r)
        net_trades.append(net_r)

    if not net_trades:
        return {
            "num_trades": 0.0,
            "hit_rate": 0.0,
            "avg_r": 0.0,
            "avg_r_net": 0.0,
            "max_drawdown_r": 0.0,
            "max_drawdown_r_net": 0.0,
            "max_loss_streak": 0.0,
            "profit_factor_net": 0.0,
            "skipped_band": float(skipped_band),
            "skipped_threshold": float(skipped_threshold),
            "trades_net": np.array([], dtype=np.float64),
        }

    gross_np = np.array(gross_trades, dtype=np.float64)
    net_np = np.array(net_trades, dtype=np.float64)

    equity_gross = np.cumsum(gross_np)
    running_max_gross = np.maximum.accumulate(equity_gross)
    drawdown_gross = running_max_gross - equity_gross

    equity_net = np.cumsum(net_np)
    running_max_net = np.maximum.accumulate(equity_net)
    drawdown_net = running_max_net - equity_net

    max_loss_streak = 0
    cur_loss_streak = 0
    for x in net_np:
        if x < 0:
            cur_loss_streak += 1
            max_loss_streak = max(max_loss_streak, cur_loss_streak)
        else:
            cur_loss_streak = 0

    wins = net_np[net_np > 0]
    losses = net_np[net_np < 0]
    gross_wins = float(np.sum(wins)) if wins.size else 0.0
    gross_losses_abs = float(abs(np.sum(losses))) if losses.size else 0.0
    profit_factor_net = (gross_wins / gross_losses_abs) if gross_losses_abs > 0 else (999.0 if gross_wins > 0 else 0.0)

    return {
        "num_trades": float(len(net_np)),
        "hit_rate": float(np.mean(gross_np > 0)),
        "avg_r": float(np.mean(gross_np)),
        "avg_r_net": float(np.mean(net_np)),
        "max_drawdown_r": float(np.max(drawdown_gross)),
        "max_drawdown_r_net": float(np.max(drawdown_net)),
        "max_loss_streak": float(max_loss_streak),
        "profit_factor_net": float(profit_factor_net),
        "skipped_band": float(skipped_band),
        "skipped_threshold": float(skipped_threshold),
        "trades_net": net_np,
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
