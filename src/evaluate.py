from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.config import settings
from src.feature_engineering import build_event_dataset
from src.models import ensemble_predict_proba, load_artifacts
from src.policy import load_policy


def simulate_trades(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.6, no_trade_band: float = 0.05) -> dict[str, float]:
    trades = []
    for y, p in zip(y_true, proba):
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

    equity = np.cumsum(trades)
    running_max = np.maximum.accumulate(equity)
    drawdown = running_max - equity

    return {
        "num_trades": float(len(trades)),
        "hit_rate": float(np.mean(np.array(trades) > 0)),
        "avg_r": float(np.mean(trades)),
        "max_drawdown_r": float(np.max(drawdown)),
    }


def main() -> None:
    events = pd.read_csv(settings.events_csv)
    market = pd.read_csv(settings.market_csv)

    bundle = build_event_dataset(events, market, lookback_seconds=settings.lookback_seconds)
    if bundle.X_tabular.empty:
        raise RuntimeError("No evaluation samples were built.")

    tabular_models, lstm_model, _ = load_artifacts(settings.model_dir)
    if not tabular_models and lstm_model is None:
        raise RuntimeError("No saved models found. Run training first.")

    probs = []
    X_tab = bundle.X_tabular.to_numpy()
    X_seq = bundle.X_seq

    for i in range(X_tab.shape[0]):
        probs.append(ensemble_predict_proba(tabular_models, lstm_model, X_tab[i], X_seq[i]))

    probs_np = np.array(probs)
    y_pred = (probs_np >= 0.5).astype(int)

    policy = load_policy(
        settings.model_dir,
        default_threshold=settings.decision_threshold,
        default_no_trade_band=settings.no_trade_band,
    )

    print("Classification report:")
    print(classification_report(bundle.y_direction, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(bundle.y_direction, y_pred))

    trade_metrics = simulate_trades(
        bundle.y_direction,
        probs_np,
        threshold=policy["decision_threshold"],
        no_trade_band=policy["no_trade_band"],
    )
    print("Decision policy:", policy)
    print("Trade simulation metrics:")
    print(trade_metrics)

    os.makedirs(settings.model_dir, exist_ok=True)
    out = pd.DataFrame(
        {
            "y_true": bundle.y_direction,
            "y_pred": y_pred,
            "proba_buy": probs_np,
        }
    )
    out.to_csv(os.path.join(settings.model_dir, "evaluation_predictions.csv"), index=False)


if __name__ == "__main__":
    main()
