from __future__ import annotations

import json
import os
import importlib
from dataclasses import dataclass
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


tf = None
Sequential = None
LSTM = None
Dense = None
Dropout = None


def _load_tensorflow():
    global tf, Sequential, LSTM, Dense, Dropout

    if tf is not None and Sequential is not None:
        return tf, Sequential, LSTM, Dense, Dropout

    try:
        tf_module = importlib.import_module("tensorflow")
        keras_module = importlib.import_module("tensorflow.keras")
        layers_module = importlib.import_module("tensorflow.keras.layers")

        tf = tf_module
        Sequential = getattr(keras_module, "Sequential", None)
        LSTM = getattr(layers_module, "LSTM", None)
        Dense = getattr(layers_module, "Dense", None)
        Dropout = getattr(layers_module, "Dropout", None)
        return tf, Sequential, LSTM, Dense, Dropout
    except Exception:  # pragma: no cover
        return None, None, None, None, None


@dataclass
class TrainingArtifacts:
    tabular_models: Dict[str, Pipeline]
    lstm_model: object | None
    feature_columns: list[str]


def train_tabular_models(X: np.ndarray, y: np.ndarray) -> Dict[str, Pipeline]:
    sample_weight = _balanced_sample_weight(y)

    models = {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "rf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
            ]
        ),
        "gb": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }

    for name, model in models.items():
        if name == "gb":
            model.fit(X, y, clf__sample_weight=sample_weight)
        else:
            model.fit(X, y)

    return models


def build_lstm(input_steps: int, n_features: int):
    tf_mod, seq_cls, lstm_cls, dense_cls, dropout_cls = _load_tensorflow()
    if tf_mod is None or seq_cls is None or lstm_cls is None or dense_cls is None or dropout_cls is None:
        return None

    model = seq_cls(
        [
            lstm_cls(64, return_sequences=True, input_shape=(input_steps, n_features)),
            dropout_cls(0.2),
            lstm_cls(32),
            dropout_cls(0.2),
            dense_cls(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf_mod.keras.metrics.AUC(name="auc")])
    return model


def train_lstm(X_seq: np.ndarray, y: np.ndarray, epochs: int = 12, batch_size: int = 32):
    model = build_lstm(X_seq.shape[1], X_seq.shape[2])
    if model is None:
        return None

    model.fit(X_seq, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
    return model


def evaluate_tabular_cv(X: np.ndarray, y: np.ndarray, model_name: str = "rf") -> Dict[str, float]:
    model_map = {
        "logreg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "rf": Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"))]),
        "gb": Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier(random_state=42))]),
    }

    model = model_map[model_name]
    splitter = TimeSeriesSplit(n_splits=5)

    f1_scores = []
    auc_scores = []
    acc_scores = []

    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        acc_scores.append(accuracy_score(y_test, pred))
        f1_scores.append(f1_score(y_test, pred, zero_division=0))
        if len(np.unique(y_test)) > 1:
            auc_scores.append(roc_auc_score(y_test, proba))

    return {
        "acc_mean": float(np.mean(acc_scores)),
        "f1_mean": float(np.mean(f1_scores)),
        "auc_mean": float(np.nan if len(auc_scores) == 0 else np.mean(auc_scores)),
    }


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    total = counts.sum()
    class_weight = {cls: total / (len(classes) * cnt) for cls, cnt in zip(classes, counts)}
    return np.array([class_weight[val] for val in y_arr], dtype=np.float64)


def save_artifacts(artifacts: TrainingArtifacts, model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)

    for name, model in artifacts.tabular_models.items():
        joblib.dump(model, os.path.join(model_dir, f"{name}.joblib"))

    if artifacts.lstm_model is not None and hasattr(artifacts.lstm_model, "save"):
        artifacts.lstm_model.save(os.path.join(model_dir, "lstm.keras"))

    meta = {"feature_columns": artifacts.feature_columns}
    with open(os.path.join(model_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_artifacts(model_dir: str) -> Tuple[Dict[str, Pipeline], object | None, list[str]]:
    tabular = {}
    for name in ["logreg", "rf", "gb"]:
        path = os.path.join(model_dir, f"{name}.joblib")
        if os.path.exists(path):
            tabular[name] = joblib.load(path)

    lstm_model = None
    # Prefer the actively trained event model when available.
    event_lstm_path = os.path.join(model_dir, "event_lstm.h5")
    lstm_path = os.path.join(model_dir, "lstm.keras")
    tf_mod, _, _, _, _ = _load_tensorflow()
    if tf_mod is not None:
        if os.path.exists(event_lstm_path):
            lstm_model = tf_mod.keras.models.load_model(event_lstm_path)
        elif os.path.exists(lstm_path):
            lstm_model = tf_mod.keras.models.load_model(lstm_path)

    meta_path = os.path.join(model_dir, "metadata.json")
    feature_columns = []
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            feature_columns = json.load(f).get("feature_columns", [])

    return tabular, lstm_model, feature_columns


def load_tabular_artifacts(model_dir: str) -> Tuple[Dict[str, Pipeline], list[str]]:
    """Load only tabular joblib models and feature columns without importing TensorFlow.

    Use this when the environment cannot safely import TensorFlow or when only tabular models
    are required for inference.
    """
    tabular = {}
    for name in ["logreg", "rf", "gb"]:
        path = os.path.join(model_dir, f"{name}.joblib")
        if os.path.exists(path):
            tabular[name] = joblib.load(path)

    meta_path = os.path.join(model_dir, "metadata.json")
    feature_columns = []
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            feature_columns = json.load(f).get("feature_columns", [])

    return tabular, feature_columns


def ensemble_predict_proba(tabular_models: Dict[str, Pipeline], lstm_model, X_tab_row: np.ndarray, X_seq_row: np.ndarray) -> float:
    probs = []

    for model in tabular_models.values():
        p = float(model.predict_proba(X_tab_row.reshape(1, -1))[0, 1])
        probs.append(p)

    if lstm_model is not None:
        x_seq = _adapt_sequence_for_model(X_seq_row, lstm_model)
        pred = np.asarray(lstm_model.predict(x_seq, verbose=0))
        p_lstm = _prediction_to_buy_probability(pred)
        probs.append(float(p_lstm))

    if not probs:
        return 0.5

    return float(np.mean(probs))


def _adapt_sequence_for_model(x_seq_row: np.ndarray, model) -> np.ndarray:
    """Adapt sequence length/features to the loaded model input shape."""
    x = np.asarray(x_seq_row, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    expected_steps = None
    expected_features = None
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        # (batch, steps, features)
        expected_steps = int(shape[1]) if shape[1] is not None else None
        expected_features = int(shape[2]) if shape[2] is not None else None
    except Exception:
        pass

    # feature adaptation
    if expected_features is not None and x.shape[1] != expected_features:
        if x.shape[1] > expected_features:
            x = x[:, :expected_features]
        elif x.shape[1] == 1 and expected_features > 1:
            x = np.repeat(x, expected_features, axis=1)
        else:
            pad = np.zeros((x.shape[0], expected_features - x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)

    # time-step adaptation
    if expected_steps is not None and x.shape[0] != expected_steps:
        if x.shape[0] > expected_steps:
            x = x[-expected_steps:, :]
        else:
            pad = np.zeros((expected_steps - x.shape[0], x.shape[1]), dtype=np.float32)
            x = np.concatenate([pad, x], axis=0)

    return x.reshape(1, x.shape[0], x.shape[1])


def _prediction_to_buy_probability(pred: np.ndarray) -> float:
    """Convert model prediction output to a BUY probability in [0, 1]."""
    arr = np.asarray(pred, dtype=np.float32)
    if arr.size == 0:
        return 0.5

    # Handle common output layouts:
    # - Binary sigmoid: shape (1, 1)
    # - Binary softmax/proba: shape (1, 2)
    # - Event multiclass [no_move, buy, sell]: shape (1, 3)
    flat = arr.reshape(arr.shape[0], -1)[0]

    if flat.shape[0] == 1:
        return float(np.clip(flat[0], 0.0, 1.0))
    if flat.shape[0] == 2:
        return float(np.clip(flat[1], 0.0, 1.0))
    if flat.shape[0] >= 3:
        p_no_move = float(flat[0])
        p_buy = float(flat[1])
        # Keep neutrality when model predicts no_move.
        return float(np.clip(p_buy + 0.5 * p_no_move, 0.0, 1.0))

    return 0.5
