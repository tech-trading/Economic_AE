from __future__ import annotations

import os
import numpy as np

import pandas as pd

from src.config import settings
from src.feature_engineering import build_event_dataset
from src.models import TrainingArtifacts, ensemble_predict_proba, evaluate_tabular_cv, save_artifacts, train_lstm, train_tabular_models
from src.policy import optimize_policy, save_policy


def main() -> None:
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.model_dir, exist_ok=True)

    events = pd.read_csv(settings.events_csv)
    market = pd.read_csv(settings.market_csv)

    bundle = build_event_dataset(events, market, lookback_seconds=settings.lookback_seconds)
    if bundle.X_tabular.empty:
        raise RuntimeError("No training samples were built. Check event coverage and market tick data.")

    X = bundle.X_tabular.to_numpy()
    y = bundle.y_direction

    cv_scores = {
        name: evaluate_tabular_cv(X, y, model_name=name)
        for name in ["logreg", "rf", "gb"]
    }
    print("Time-series CV scores:")
    for name, scores in cv_scores.items():
        print(name, scores)

    tabular_models = train_tabular_models(X, y)
    lstm_model = train_lstm(bundle.X_seq, y)

    probs = []
    for i in range(X.shape[0]):
        probs.append(ensemble_predict_proba(tabular_models, lstm_model, X[i], bundle.X_seq[i]))

    policy = optimize_policy(y, np.array(probs, dtype=np.float64))
    save_policy(settings.model_dir, policy)
    print("Optimized policy:", policy)

    artifacts = TrainingArtifacts(
        tabular_models=tabular_models,
        lstm_model=lstm_model,
        feature_columns=bundle.X_tabular.columns.tolist(),
    )
    save_artifacts(artifacts, settings.model_dir)
    print(f"Models saved to: {settings.model_dir}")


if __name__ == "__main__":
    main()
