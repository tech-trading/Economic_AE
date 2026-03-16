from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from src.models import ensemble_predict_proba
from src.mt5_executor import TradeDecision


class Strategy:
    requires_models: bool = True

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


class DonchianBreakoutStrategy(Strategy):
    requires_models: bool = False

    def __init__(
        self,
        lookback_seconds: int = 600,
        breakout_buffer_pips: float = 0.2,
        min_channel_pips: float = 1.0,
        confirm_ticks: int = 1,
        trigger_quantile: float = 0.80,
    ):
        self.lookback_seconds = lookback_seconds
        self.breakout_buffer_pips = breakout_buffer_pips
        self.min_channel_pips = min_channel_pips
        self.confirm_ticks = max(1, int(confirm_ticks))
        self.trigger_quantile = float(np.clip(trigger_quantile, 0.55, 0.95))

    @staticmethod
    def _pip_size(symbol: str) -> float:
        sym = str(symbol or "").upper()
        return 0.01 if "JPY" in sym else 0.0001

    def _window(self, ticks: pd.DataFrame) -> pd.DataFrame:
        if ticks is None or ticks.empty:
            return pd.DataFrame()

        df = ticks
        if "time_utc" in df.columns:
            times = df["time_utc"]
            if not times.is_monotonic_increasing:
                df = df.sort_values("time_utc")
                times = df["time_utc"]

            utc_to = times.iat[-1]
            start_time = utc_to - pd.Timedelta(seconds=self.lookback_seconds)
            start_idx = int(times.searchsorted(start_time, side="left"))
            out = df.iloc[start_idx:]
            if out.empty:
                return df.tail(max(30, self.confirm_ticks + 5))
            return out

        return df.tail(300)

    def decide(self, event_row, ticks, bundle, tabular_models, lstm_model, feature_columns, policy, settings):
        window = self._window(ticks)
        if window.empty or len(window) < max(6, self.confirm_ticks + 3):
            return None

        mid = ((window["bid"].astype(float) + window["ask"].astype(float)) / 2.0).dropna()
        if len(mid) < max(6, self.confirm_ticks + 3):
            return None

        pivot = mid.iloc[:-self.confirm_ticks] if len(mid) > self.confirm_ticks else mid
        if pivot.empty:
            return None

        high = float(pivot.max())
        low = float(pivot.min())
        latest_block = mid.tail(self.confirm_ticks)
        latest = float(latest_block.iat[-1])

        pip = self._pip_size(getattr(settings, "symbol", "EURUSD"))
        buffer = float(self.breakout_buffer_pips) * pip
        channel_width = max(1e-12, high - low)
        channel_pips = channel_width / pip
        if channel_pips < float(self.min_channel_pips):
            return None

        buy_break = bool((latest_block > (high + buffer)).all())
        sell_break = bool((latest_block < (low - buffer)).all())
        channel_pos = float(np.clip((latest - low) / channel_width, 0.0, 1.0))
        buy_zone = channel_pos >= self.trigger_quantile
        sell_zone = channel_pos <= (1.0 - self.trigger_quantile)

        if not buy_break and not sell_break:
            buy_break = buy_zone
            sell_break = sell_zone

        if buy_break == sell_break:
            return None

        ema_fast = float(mid.ewm(span=20, adjust=False).mean().iat[-1])
        ema_slow = float(mid.ewm(span=50, adjust=False).mean().iat[-1])

        trend_factor = 1.0
        if buy_break:
            if ema_fast <= ema_slow:
                trend_factor = 0.95
            distance = max(0.0, latest - (high + buffer))
            side = "BUY"
            direction = 1.0
        else:
            if ema_fast >= ema_slow:
                trend_factor = 0.95
            distance = max(0.0, (low - buffer) - latest)
            side = "SELL"
            direction = -1.0

        strength = float(distance / channel_width)
        edge_strength = max(strength, abs(channel_pos - 0.5) * 2.0)
        confidence = float(np.clip((0.55 + min(0.35, edge_strength * 0.45)) * trend_factor, 0.55, 0.93))
        if confidence < float(policy.get("decision_threshold", 0.5)):
            return None

        proba_buy = float(np.clip(0.5 + direction * min(0.49, 0.28 + strength), 0.01, 0.99))
        return TradeDecision(side=side, confidence=confidence, proba_buy=proba_buy)


def get_strategy(name: str, settings, policy: dict) -> Strategy:
    name = (name or "").strip().lower()
    if name == "zscore" or name == "z_score" or name == "z-score":
        return ZScoreStrategy(lookback_seconds=int(settings.z_score_lookback_seconds), z_threshold=float(settings.z_score_threshold), z_weight=float(settings.z_weight), mode=settings.z_combination_mode)
    if name == "momentum" or name == "mom":
        return MomentumStrategy(lookback_seconds=int(settings.momentum_lookback_seconds), momentum_threshold=float(settings.momentum_threshold), momentum_weight=float(settings.momentum_weight), mode=settings.momentum_mode)
    if name in {"donchian", "breakout", "turtle", "donchian_breakout"}:
        return DonchianBreakoutStrategy(
            lookback_seconds=int(settings.donchian_lookback_seconds),
            breakout_buffer_pips=float(settings.donchian_breakout_buffer_pips),
            min_channel_pips=float(settings.donchian_min_channel_pips),
            confirm_ticks=int(settings.donchian_confirm_ticks),
            trigger_quantile=float(settings.donchian_trigger_quantile),
        )
    # default fallback
    return DefaultStrategy()
