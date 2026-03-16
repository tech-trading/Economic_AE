from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from src.models import ensemble_predict_proba
from src.mt5_executor import TradeDecision


class Strategy:
    def decide(self, event_row: pd.Series, ticks: pd.DataFrame, bundle: Any, tabular_models, lstm_model, feature_columns, policy: dict, settings) -> TradeDecision | None:  # pragma: no cover - simple interface
        raise NotImplementedError()


class DefaultStrategy(Strategy):
    def decide(self, event_row, ticks, bundle, tabular_models, lstm_model, feature_columns, policy, settings):
        if bundle.X_tabular.empty:
            return None

        x_row = bundle.X_tabular.iloc[0].reindex(feature_columns, fill_value=0.0)
        proba_buy = ensemble_predict_proba(tabular_models, lstm_model, x_row.to_numpy(dtype=np.float32), bundle.X_seq[0])

        side = "BUY" if proba_buy >= 0.5 else "SELL"
        confidence = float(max(proba_buy, 1.0 - proba_buy))

        if confidence < policy.get("decision_threshold", 0.5):
            return None
        if abs(proba_buy - 0.5) < policy.get("no_trade_band", 0.0):
            return None

        return TradeDecision(side=side, confidence=confidence, proba_buy=float(proba_buy))


class ZScoreStrategy(Strategy):
    def __init__(self, lookback_seconds: int = 300, z_threshold: float = 0.7, z_weight: float = 1.0, mode: str = "weighted"):
        self.lookback_seconds = lookback_seconds
        self.z_threshold = z_threshold
        self.z_weight = z_weight
        self.mode = mode

    def _compute_z(self, ticks: pd.DataFrame) -> float:
        if ticks is None or ticks.empty:
            return 0.0

        # avoid expensive boolean masks and full copies for large ticksets
        try:
            times = ticks["time_utc"]
            # ensure monotonic increasing
            if not times.is_monotonic_increasing:
                ticks = ticks.sort_values("time_utc")
                times = ticks["time_utc"]

            utc_to = times.iat[-1]
            start_time = utc_to - pd.Timedelta(seconds=self.lookback_seconds)
            start_idx = int(times.searchsorted(start_time, side="left"))
            window = ticks.iloc[start_idx:]
        except Exception:
            window = ticks.tail(100)

        if window.empty:
            return 0.0

        # compute mid and stats
        mid = (window["bid"] + window["ask"]) / 2.0
        std = float(mid.std())
        if std == 0:
            return 0.0
        last = float(mid.iat[-1])
        return float((last - float(mid.mean())) / std)

    def decide(self, event_row, ticks, bundle, tabular_models, lstm_model, feature_columns, policy, settings):
        if bundle.X_tabular.empty:
            return None

        x_row = bundle.X_tabular.iloc[0].reindex(feature_columns, fill_value=0.0)
        proba_buy = ensemble_predict_proba(tabular_models, lstm_model, x_row.to_numpy(dtype=np.float32), bundle.X_seq[0])

        z = self._compute_z(ticks)

        if self.mode == "conjunctive":
            dir_model = 1 if proba_buy >= 0.5 else -1
            dir_z = 1 if z > self.z_threshold else (-1 if z < -self.z_threshold else 0)
            if dir_z == 0 or dir_model != dir_z:
                return None
            side = "BUY" if dir_model == 1 else "SELL"
            confidence = float(max(proba_buy, 1.0 - proba_buy))
            if confidence < policy.get("decision_threshold", 0.5):
                return None
            return TradeDecision(side=side, confidence=confidence, proba_buy=float(proba_buy))

        # weighted combination
        model_score = proba_buy - 0.5
        z_norm = float(np.tanh(z))  # bound to [-1,1]
        combined = model_score + (self.z_weight * z_norm / 2.0)

        confidence = float(min(1.0, abs(combined)))
        if confidence < policy.get("decision_threshold", 0.5):
            return None
        if abs(proba_buy - 0.5) < policy.get("no_trade_band", 0.0):
            return None

        side = "BUY" if combined >= 0 else "SELL"
        return TradeDecision(side=side, confidence=confidence, proba_buy=float(proba_buy))


class MomentumStrategy(Strategy):
    def __init__(self, lookback_seconds: int = 300, momentum_threshold: float = 0.0005, momentum_weight: float = 1.0, mode: str = "weighted"):
        self.lookback_seconds = lookback_seconds
        self.momentum_threshold = momentum_threshold
        self.momentum_weight = momentum_weight
        self.mode = mode

    def _compute_momentum(self, ticks: pd.DataFrame) -> float:
        if ticks is None or ticks.empty:
            return 0.0
        times = ticks["time_utc"]
        if not times.is_monotonic_increasing:
            ticks = ticks.sort_values("time_utc")
            times = ticks["time_utc"]
        utc_to = times.iat[-1]
        start_time = utc_to - pd.Timedelta(seconds=self.lookback_seconds)
        start_idx = int(times.searchsorted(start_time, side="left"))
        window = ticks.iloc[start_idx:]
        if window.empty:
            return 0.0
        mid = (window["bid"] + window["ask"]) / 2.0
        if len(mid) < 2:
            return 0.0
        # simple percentage momentum from first to last
        first = float(mid.iat[0])
        last = float(mid.iat[-1])
        if first == 0:
            return 0.0
        return (last - first) / first

    def decide(self, event_row, ticks, bundle, tabular_models, lstm_model, feature_columns, policy, settings):
        if bundle.X_tabular.empty:
            return None

        x_row = bundle.X_tabular.iloc[0].reindex(feature_columns, fill_value=0.0)
        proba_buy = ensemble_predict_proba(tabular_models, lstm_model, x_row.to_numpy(dtype=np.float32), bundle.X_seq[0])

        mom = self._compute_momentum(ticks)

        # conjunctive: require momentum direction and model agree
        if self.mode == "conjunctive":
            dir_model = 1 if proba_buy >= 0.5 else -1
            dir_mom = 1 if mom > self.momentum_threshold else (-1 if mom < -self.momentum_threshold else 0)
            if dir_mom == 0 or dir_model != dir_mom:
                return None
            side = "BUY" if dir_model == 1 else "SELL"
            confidence = float(max(proba_buy, 1.0 - proba_buy))
            if confidence < policy.get("decision_threshold", 0.5):
                return None
            return TradeDecision(side=side, confidence=confidence, proba_buy=float(proba_buy))

        # weighted: combine model score and momentum
        model_score = proba_buy - 0.5
        # normalize momentum (scale to reasonable range then tanh)
        mom_norm = float(np.tanh(mom * 200.0))  # scale multiplier tuned for percent returns
        combined = model_score + (self.momentum_weight * mom_norm / 2.0)

        confidence = float(min(1.0, abs(combined)))
        if confidence < policy.get("decision_threshold", 0.5):
            return None
        if abs(proba_buy - 0.5) < policy.get("no_trade_band", 0.0):
            return None

        side = "BUY" if combined >= 0 else "SELL"
        return TradeDecision(side=side, confidence=confidence, proba_buy=float(proba_buy))


def get_strategy(name: str, settings, policy: dict) -> Strategy:
    name = (name or "").strip().lower()
    if name == "zscore" or name == "z_score" or name == "z-score":
        return ZScoreStrategy(lookback_seconds=int(settings.z_score_lookback_seconds), z_threshold=float(settings.z_score_threshold), z_weight=float(settings.z_weight), mode=settings.z_combination_mode)
    if name == "momentum" or name == "mom":
        return MomentumStrategy(lookback_seconds=int(settings.momentum_lookback_seconds), momentum_threshold=float(settings.momentum_threshold), momentum_weight=float(settings.momentum_weight), mode=settings.momentum_mode)
    # default fallback
    return DefaultStrategy()
