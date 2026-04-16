from __future__ import annotations

import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any

from src.models import ensemble_predict_proba
from src.mt5_executor import TradeDecision
from src.fundamental_agent import FundamentalNewsLLMEngine


class Strategy:
    requires_models: bool = True
    requires_event: bool = True

    def decide(self, event_row: pd.Series, ticks: pd.DataFrame, bundle: Any, tabular_models, lstm_model, feature_columns, policy: dict, settings) -> TradeDecision | None:  # pragma: no cover - simple interface
        raise NotImplementedError()


class AgentManagedStrategy(Strategy):
    """Adapter that runs any strategy behind a unified agent interface."""

    def __init__(self, agent_name: str, strategy: Strategy):
        self.agent_name = str(agent_name or "unknown")
        self.strategy = strategy
        self.requires_models = bool(getattr(strategy, "requires_models", True))
        self.requires_event = bool(getattr(strategy, "requires_event", True))
        self.calls = 0
        self.decisions = 0
        self.last_decision_side: str | None = None

    def decide(self, event_row, ticks, bundle, tabular_models, lstm_model, feature_columns, policy, settings):
        self.calls += 1
        decision = self.strategy.decide(event_row, ticks, bundle, tabular_models, lstm_model, feature_columns, policy, settings)
        if decision is not None:
            self.decisions += 1
            self.last_decision_side = str(getattr(decision, "side", ""))
        return decision

    def get_agent_status(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "strategy_class": self.strategy.__class__.__name__,
            "calls": int(self.calls),
            "decisions": int(self.decisions),
            "last_decision_side": self.last_decision_side,
            "requires_models": bool(self.requires_models),
            "requires_event": bool(self.requires_event),
        }


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


class EmaRsiTrendStrategy(Strategy):
    requires_models: bool = False
    requires_event: bool = False

    def __init__(
        self,
        fast_span: int = 21,
        slow_span: int = 55,
        rsi_period: int = 14,
        rsi_buy_level: float = 56.0,
        rsi_sell_level: float = 44.0,
        min_separation_pips: float = 0.20,
        momentum_lookback_ticks: int = 20,
        min_momentum_pips: float = 0.25,
        vol_period: int = 40,
        min_vol_pips: float = 0.05,
        signal_cooldown_seconds: int = 180,
    ):
        self.fast_span = max(3, int(fast_span))
        self.slow_span = max(self.fast_span + 2, int(slow_span))
        self.rsi_period = max(5, int(rsi_period))
        self.rsi_buy_level = float(np.clip(rsi_buy_level, 50.0, 80.0))
        self.rsi_sell_level = float(np.clip(rsi_sell_level, 20.0, 50.0))
        self.min_separation_pips = max(0.0, float(min_separation_pips))
        self.momentum_lookback_ticks = max(3, int(momentum_lookback_ticks))
        self.min_momentum_pips = max(0.0, float(min_momentum_pips))
        self.vol_period = max(8, int(vol_period))
        self.min_vol_pips = max(0.0, float(min_vol_pips))
        self.signal_cooldown_seconds = max(0, int(signal_cooldown_seconds))
        self._last_signal_side: str | None = None
        self._last_signal_ts: pd.Timestamp | None = None

    @staticmethod
    def _pip_size(symbol: str) -> float:
        sym = str(symbol or "").upper()
        return 0.01 if "JPY" in sym else 0.0001

    def _compute_rsi(self, prices: pd.Series) -> float:
        if prices is None or len(prices) < self.rsi_period + 2:
            return 50.0
        delta = prices.diff().dropna()
        gain = delta.clip(lower=0.0)
        loss = (-delta.clip(upper=0.0))
        avg_gain = gain.ewm(alpha=1.0 / self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.rsi_period, adjust=False).mean()
        rs = avg_gain.iloc[-1] / max(1e-12, avg_loss.iloc[-1])
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(np.clip(rsi, 0.0, 100.0))

    def decide(self, event_row, ticks, bundle, tabular_models, lstm_model, feature_columns, policy, settings):
        if ticks is None or ticks.empty or len(ticks) < max(self.slow_span + 5, self.vol_period + 5):
            return None

        df = ticks
        if "time_utc" in df.columns and not df["time_utc"].is_monotonic_increasing:
            df = df.sort_values("time_utc")

        mid = ((df["bid"].astype(float) + df["ask"].astype(float)) / 2.0).dropna()
        if len(mid) < max(self.slow_span + 5, self.vol_period + 5):
            return None

        pip = self._pip_size(getattr(settings, "symbol", "EURUSD"))
        ema_fast = mid.ewm(span=self.fast_span, adjust=False).mean()
        ema_slow = mid.ewm(span=self.slow_span, adjust=False).mean()

        ema_gap_pips = float((ema_fast.iloc[-1] - ema_slow.iloc[-1]) / pip)
        rsi = self._compute_rsi(mid)

        lb = min(len(mid) - 1, self.momentum_lookback_ticks)
        momentum_pips = float((mid.iloc[-1] - mid.iloc[-1 - lb]) / pip) if lb > 0 else 0.0

        vol_pips = float((mid.diff().abs().rolling(self.vol_period).mean().iloc[-1]) / pip)
        if not np.isfinite(vol_pips) or vol_pips < self.min_vol_pips:
            return None

        buy_ok = (
            (ema_gap_pips >= self.min_separation_pips)
            and (rsi >= self.rsi_buy_level)
            and (momentum_pips >= self.min_momentum_pips)
        )
        sell_ok = (
            (ema_gap_pips <= -self.min_separation_pips)
            and (rsi <= self.rsi_sell_level)
            and (momentum_pips <= -self.min_momentum_pips)
        )

        if buy_ok == sell_ok:
            return None

        side = "BUY" if buy_ok else "SELL"
        direction = 1.0 if buy_ok else -1.0
        strength_gap = min(1.0, abs(ema_gap_pips) / max(1e-6, self.min_separation_pips + 0.25))
        strength_mom = min(1.0, abs(momentum_pips) / max(1e-6, self.min_momentum_pips + 0.35))
        rsi_edge = abs(rsi - 50.0) / 50.0
        confidence = float(np.clip(0.52 + 0.18 * strength_gap + 0.16 * strength_mom + 0.12 * rsi_edge, 0.52, 0.94))
        if confidence < float(policy.get("decision_threshold", 0.5)):
            return None

        current_ts = pd.Timestamp.now(tz="UTC")
        if "time_utc" in df.columns:
            ts = pd.to_datetime(df["time_utc"].iloc[-1], utc=True, errors="coerce")
            if pd.notna(ts):
                current_ts = ts

        if (
            self.signal_cooldown_seconds > 0
            and self._last_signal_side == side
            and self._last_signal_ts is not None
        ):
            elapsed = float((current_ts - self._last_signal_ts).total_seconds())
            if elapsed < float(self.signal_cooldown_seconds):
                return None

        proba_buy = float(np.clip(0.5 + direction * min(0.45, 0.20 + 0.30 * confidence), 0.01, 0.99))
        self._last_signal_side = side
        self._last_signal_ts = current_ts
        return TradeDecision(side=side, confidence=confidence, proba_buy=proba_buy)


class TurtleAtrBreakoutStrategy(Strategy):
    requires_models: bool = False
    requires_event: bool = False

    def __init__(
        self,
        lookback_seconds: int = 3600,
        breakout_buffer_pips: float = 0.10,
        min_channel_pips: float = 0.02,
        confirm_ticks: int = 1,
        atr_period_ticks: int = 120,
        min_atr_pips: float = 0.08,
        trigger_quantile: float = 0.85,
        trend_ema_span: int = 180,
        max_extension_atr: float = 2.50,
        signal_cooldown_seconds: int = 240,
    ):
        self.lookback_seconds = max(120, int(lookback_seconds))
        self.breakout_buffer_pips = max(0.0, float(breakout_buffer_pips))
        self.min_channel_pips = max(0.1, float(min_channel_pips))
        self.confirm_ticks = max(1, int(confirm_ticks))
        self.atr_period_ticks = max(20, int(atr_period_ticks))
        self.min_atr_pips = max(0.0, float(min_atr_pips))
        self.trigger_quantile = float(np.clip(trigger_quantile, 0.65, 0.95))
        self.trend_ema_span = max(20, int(trend_ema_span))
        self.max_extension_atr = max(0.5, float(max_extension_atr))
        self.signal_cooldown_seconds = max(0, int(signal_cooldown_seconds))
        self._last_signal_side: str | None = None
        self._last_signal_ts: pd.Timestamp | None = None

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
                return df.tail(max(300, self.atr_period_ticks + self.confirm_ticks + 20))
            return out

        return df.tail(max(300, self.atr_period_ticks + self.confirm_ticks + 20))

    def decide(self, event_row, ticks, bundle, tabular_models, lstm_model, feature_columns, policy, settings):
        window = self._window(ticks)
        min_rows = max(self.atr_period_ticks + 5, self.confirm_ticks + 5, self.trend_ema_span + 5)
        if window.empty or len(window) < min_rows:
            return None

        mid = ((window["bid"].astype(float) + window["ask"].astype(float)) / 2.0).dropna()
        if len(mid) < min_rows:
            return None

        pip = self._pip_size(getattr(settings, "symbol", "EURUSD"))

        pivot = mid.iloc[:-self.confirm_ticks] if len(mid) > self.confirm_ticks else mid
        if pivot.empty:
            return None
        high = float(pivot.max())
        low = float(pivot.min())
        latest_block = mid.tail(self.confirm_ticks)
        latest = float(latest_block.iat[-1])

        channel_width = max(1e-12, high - low)
        channel_pips = float(channel_width / pip)
        if channel_pips < self.min_channel_pips:
            return None

        tr = mid.diff().abs().fillna(0.0)
        atr_price = float(tr.rolling(self.atr_period_ticks).mean().iloc[-1])
        if not np.isfinite(atr_price) or atr_price <= 0:
            return None
        atr_pips = float(atr_price / pip)
        if atr_pips < self.min_atr_pips:
            return None

        ema_trend = float(mid.ewm(span=self.trend_ema_span, adjust=False).mean().iat[-1])
        extension_norm = abs(latest - ema_trend) / max(1e-12, max(channel_width, 2.0 * atr_price))
        if extension_norm > self.max_extension_atr:
            return None

        buffer = self.breakout_buffer_pips * pip
        buy_break = bool((latest_block > (high + buffer)).all())
        sell_break = bool((latest_block < (low - buffer)).all())

        # Fallback zone logic: enables sparse, trend-aligned entries when strict breakout confirmation is absent.
        channel_pos = float(np.clip((latest - low) / channel_width, 0.0, 1.0))
        buy_zone = channel_pos >= self.trigger_quantile
        sell_zone = channel_pos <= (1.0 - self.trigger_quantile)
        if not buy_break and not sell_break:
            buy_break = buy_zone
            sell_break = sell_zone

        # trend filter
        if buy_break and latest < ema_trend:
            buy_break = False
        if sell_break and latest > ema_trend:
            sell_break = False

        if buy_break == sell_break:
            return None

        side = "BUY" if buy_break else "SELL"
        direction = 1.0 if buy_break else -1.0

        now_ts = pd.Timestamp.now(tz="UTC")
        if "time_utc" in window.columns:
            ts = pd.to_datetime(window["time_utc"].iloc[-1], utc=True, errors="coerce")
            if pd.notna(ts):
                now_ts = ts

        if (
            self.signal_cooldown_seconds > 0
            and self._last_signal_side == side
            and self._last_signal_ts is not None
        ):
            elapsed = float((now_ts - self._last_signal_ts).total_seconds())
            if elapsed < float(self.signal_cooldown_seconds):
                return None

        breakout_dist = max(0.0, abs(latest - (high if buy_break else low)) / max(1e-12, channel_width))
        vol_factor = min(1.0, atr_pips / max(self.min_atr_pips + 1e-6, 2.0 * self.min_atr_pips))
        confidence = float(np.clip(0.56 + 0.25 * min(1.0, breakout_dist) + 0.12 * vol_factor, 0.56, 0.92))
        if confidence < float(policy.get("decision_threshold", 0.5)):
            return None

        proba_buy = float(np.clip(0.5 + direction * min(0.46, 0.22 + 0.26 * confidence), 0.01, 0.99))
        self._last_signal_side = side
        self._last_signal_ts = now_ts
        return TradeDecision(side=side, confidence=confidence, proba_buy=proba_buy)


class FundamentalLLMStrategy(Strategy):
    requires_models: bool = False
    requires_event: bool = False

    def __init__(self, settings):
        self.engine = FundamentalNewsLLMEngine(settings=settings)
        self.signal_cooldown_seconds = max(0, int(getattr(settings, "fundamental_signal_cooldown_seconds", 300)))
        self.min_confidence = float(np.clip(getattr(settings, "fundamental_min_confidence", 0.60), 0.50, 0.95))
        self.decision_threshold_override = float(getattr(settings, "fundamental_decision_threshold", -1.0))
        self.allow_same_side_on_news_change = bool(getattr(settings, "fundamental_allow_same_side_on_news_change", True))
        self._last_signal_side: str | None = None
        self._last_signal_ts: pd.Timestamp | None = None
        self._last_news_changed: bool = False
        self._last_news_signature: str = ""
        self._last_analysis_source: str = ""

    def get_last_signal_meta(self) -> dict[str, Any]:
        return {
            "news_changed": bool(self._last_news_changed),
            "news_signature": str(self._last_news_signature),
            "analysis_source": str(self._last_analysis_source),
        }

    def decide(self, event_row, ticks, bundle, tabular_models, lstm_model, feature_columns, policy, settings):
        now_ts = pd.Timestamp.now(tz="UTC")
        if ticks is not None and not ticks.empty and "time_utc" in ticks.columns:
            ts = pd.to_datetime(ticks["time_utc"].iloc[-1], utc=True, errors="coerce")
            if pd.notna(ts):
                now_ts = ts

        symbol_for_analysis = str(getattr(settings, "symbol", "EURUSD"))
        if isinstance(event_row, pd.Series):
            maybe_symbol = str(event_row.get("symbol", "")).strip()
            if maybe_symbol:
                symbol_for_analysis = maybe_symbol

        event_ctx: dict[str, Any] = {}
        if isinstance(event_row, pd.Series):
            event_ctx = {
                "name": str(event_row.get("name", "")),
                "currency": str(event_row.get("currency", "")),
                "importance": event_row.get("importance", ""),
            }

        try:
            result = self.engine.analyze(symbol=symbol_for_analysis, event_context=event_ctx)
        except Exception:
            return None

        action = str(getattr(result, "action", "HOLD")).upper()
        confidence = float(np.clip(getattr(result, "confidence", 0.0), 0.0, 1.0))
        self._last_news_changed = bool(getattr(result, "news_changed", False))
        self._last_news_signature = str(getattr(result, "news_signature", ""))
        self._last_analysis_source = str(getattr(result, "analysis_source", ""))
        if action not in {"BUY", "SELL"}:
            return None

        override = float(self.decision_threshold_override)
        if override > 0.0:
            threshold = max(self.min_confidence, float(np.clip(override, 0.50, 0.95)))
        else:
            threshold = max(float(policy.get("decision_threshold", 0.5)), self.min_confidence)
        if confidence < threshold:
            return None

        if (
            self.signal_cooldown_seconds > 0
            and self._last_signal_side == action
            and self._last_signal_ts is not None
        ):
            elapsed = float((now_ts - self._last_signal_ts).total_seconds())
            if elapsed < float(self.signal_cooldown_seconds):
                if self.allow_same_side_on_news_change and self._last_news_changed:
                    pass
                else:
                    return None

        direction = 1.0 if action == "BUY" else -1.0
        proba_buy = float(np.clip(0.5 + direction * min(0.49, 0.10 + 0.40 * confidence), 0.01, 0.99))
        self._last_signal_side = action
        self._last_signal_ts = now_ts
        return TradeDecision(side=action, confidence=confidence, proba_buy=proba_buy)


class AgenticHybridStrategy(Strategy):
    requires_models: bool = False
    requires_event: bool = False

    def __init__(self, settings, policy: dict):
        self.policy = policy
        self.learning_rate = float(np.clip(getattr(settings, "agentic_learning_rate", 0.20), 0.01, 1.0))
        self.explore_prob = float(np.clip(getattr(settings, "agentic_explore_prob", 0.10), 0.0, 0.5))
        self.min_agent_confidence = float(np.clip(getattr(settings, "agentic_min_confidence", 0.56), 0.50, 0.95))
        self.min_fallback_confidence = float(np.clip(getattr(settings, "agentic_min_fallback_confidence", 0.53), 0.50, 0.90))
        self.decision_threshold_override = float(getattr(settings, "agentic_decision_threshold", -1.0))
        self.dynamic_threshold = bool(getattr(settings, "agentic_dynamic_threshold", True))
        self.dynamic_threshold_floor = float(np.clip(getattr(settings, "agentic_dynamic_threshold_floor", 0.54), 0.50, 0.90))
        self.dynamic_threshold_cap = float(np.clip(getattr(settings, "agentic_dynamic_threshold_cap", 0.74), 0.55, 0.98))
        self.require_agent_agreement = bool(getattr(settings, "agentic_require_agent_agreement", True))
        self.max_spread_pips = max(0.0, float(getattr(settings, "agentic_max_spread_pips", 2.5)))
        self.use_fundamental_fallback = bool(getattr(settings, "agentic_use_fundamental_fallback", False))
        self.reward_horizon_seconds = max(10, int(getattr(settings, "agentic_reward_horizon_seconds", 45)))
        self.reward_target_pips = max(0.5, float(getattr(settings, "agentic_reward_target_pips", 1.2)))
        self.signal_cooldown_seconds = max(0, int(getattr(settings, "agentic_signal_cooldown_seconds", 180)))

        self.state_path = Path(str(getattr(settings, "agentic_state_path", "models/agentic_state.json")))
        self.weights = {
            "ema_rsi": 1.0,
            "donchian": 1.0,
            "turtle_atr": 1.0,
        }
        self.agent_counts = {"ema_rsi": 0, "donchian": 0, "turtle_atr": 0}
        self.pending_trades: list[dict[str, Any]] = []
        self._last_signal_side: str | None = None
        self._last_signal_ts: pd.Timestamp | None = None
        self.fundamental_fallback = FundamentalLLMStrategy(settings=settings) if self.use_fundamental_fallback else None

        self.ema_agent = EmaRsiTrendStrategy(
            fast_span=int(settings.ema_fast_span),
            slow_span=int(settings.ema_slow_span),
            rsi_period=int(settings.ema_rsi_period),
            rsi_buy_level=float(settings.ema_rsi_buy_level),
            rsi_sell_level=float(settings.ema_rsi_sell_level),
            min_separation_pips=float(settings.ema_min_separation_pips),
            momentum_lookback_ticks=int(settings.ema_momentum_lookback_ticks),
            min_momentum_pips=float(settings.ema_min_momentum_pips),
            vol_period=int(settings.ema_vol_period),
            min_vol_pips=float(settings.ema_min_vol_pips),
        )
        self.donchian_agent = DonchianBreakoutStrategy(
            lookback_seconds=int(settings.donchian_lookback_seconds),
            breakout_buffer_pips=float(settings.donchian_breakout_buffer_pips),
            min_channel_pips=float(settings.donchian_min_channel_pips),
            confirm_ticks=int(settings.donchian_confirm_ticks),
            trigger_quantile=float(settings.donchian_trigger_quantile),
            session_filter=False,
            sessions="london,ny",
        )
        self.turtle_agent = TurtleAtrBreakoutStrategy(
            lookback_seconds=int(getattr(settings, "turtle_lookback_seconds", 1200)),
            breakout_buffer_pips=float(getattr(settings, "turtle_breakout_buffer_pips", 0.30)),
            min_channel_pips=float(getattr(settings, "turtle_min_channel_pips", 1.20)),
            confirm_ticks=int(getattr(settings, "turtle_confirm_ticks", 2)),
            atr_period_ticks=int(getattr(settings, "turtle_atr_period_ticks", 120)),
            min_atr_pips=float(getattr(settings, "turtle_min_atr_pips", 0.08)),
            trigger_quantile=float(getattr(settings, "turtle_trigger_quantile", 0.85)),
            trend_ema_span=int(getattr(settings, "turtle_trend_ema_span", 180)),
            max_extension_atr=float(getattr(settings, "turtle_max_extension_atr", 2.50)),
            signal_cooldown_seconds=int(getattr(settings, "turtle_signal_cooldown_seconds", 240)),
        )

        self._load_state()

    @staticmethod
    def _pip_size(symbol: str) -> float:
        sym = str(symbol or "").upper()
        return 0.01 if "JPY" in sym else 0.0001

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            obj = json.loads(self.state_path.read_text(encoding="utf-8"))
            w = obj.get("weights", {})
            c = obj.get("counts", {})
            for k in self.weights:
                if k in w:
                    self.weights[k] = float(w[k])
                if k in c:
                    self.agent_counts[k] = int(c[k])
        except Exception:
            return

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "weights": self.weights,
                "counts": self.agent_counts,
            }
            self.state_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception:
            return

    def _update_rewards(self, current_time: pd.Timestamp, current_mid: float, pip: float) -> None:
        if not self.pending_trades:
            return

        still_open: list[dict[str, Any]] = []
        changed = False
        for tr in self.pending_trades:
            due = tr["due_time"]
            if current_time < due:
                still_open.append(tr)
                continue

            direction = 1.0 if tr["side"] == "BUY" else -1.0
            ret_pips = ((current_mid - tr["entry_mid"]) * direction) / max(1e-12, pip)
            reward = float(np.tanh(ret_pips / self.reward_target_pips))

            k = tr["agent"]
            old_w = float(self.weights.get(k, 1.0))
            new_w = float(np.clip(old_w + (self.learning_rate * reward), 0.20, 5.00))
            self.weights[k] = new_w
            self.agent_counts[k] = int(self.agent_counts.get(k, 0)) + 1
            changed = True

        self.pending_trades = still_open
        if changed:
            self._save_state()

    def _choose_agent(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if len(candidates) == 1:
            return candidates[0]

        if random.random() < self.explore_prob:
            return random.choice(candidates)

        total_w = float(sum(max(1e-6, self.weights.get(c["agent"], 1.0)) for c in candidates))
        scored = []
        for c in candidates:
            w_norm = float(self.weights.get(c["agent"], 1.0)) / max(1e-9, total_w)
            edge = abs(float(c["decision"].proba_buy) - 0.5)
            score = (0.70 * w_norm) + (0.25 * float(c["decision"].confidence)) + (0.05 * edge)
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _consensus_decision(self, candidates: list[dict[str, Any]]) -> TradeDecision | None:
        if not candidates:
            return None

        by_side = {"BUY": [], "SELL": []}
        for c in candidates:
            side = str(c["decision"].side).upper()
            if side in by_side:
                by_side[side].append(c)

        buy_count = len(by_side["BUY"])
        sell_count = len(by_side["SELL"])
        if buy_count == 0 and sell_count == 0:
            return None

        if self.require_agent_agreement and max(buy_count, sell_count) < 2:
            return None

        def side_score(items: list[dict[str, Any]]) -> float:
            if not items:
                return 0.0
            s = 0.0
            for item in items:
                k = str(item["agent"])
                w = max(1e-6, float(self.weights.get(k, 1.0)))
                c = float(item["decision"].confidence)
                s += w * c
            return s

        buy_score = side_score(by_side["BUY"])
        sell_score = side_score(by_side["SELL"])
        total = max(1e-9, buy_score + sell_score)

        if buy_score == sell_score:
            return None

        side = "BUY" if buy_score > sell_score else "SELL"
        conf = float(np.clip(max(buy_score, sell_score) / total, 0.50, 0.96))
        proba_buy = float(np.clip(0.5 + ((buy_score - sell_score) / (2.0 * total)), 0.01, 0.99))
        return TradeDecision(side=side, confidence=conf, proba_buy=proba_buy)

    def _apply_dynamic_threshold(self, base_threshold: float, mid: pd.Series, pip: float, spread_pips: float) -> float:
        thr = float(base_threshold)
        if not self.dynamic_threshold or len(mid) < 40:
            return float(np.clip(thr, self.dynamic_threshold_floor, self.dynamic_threshold_cap))

        returns_pips = (mid.diff().dropna() / max(1e-12, pip)).astype(float)
        if returns_pips.empty:
            return float(np.clip(thr, self.dynamic_threshold_floor, self.dynamic_threshold_cap))

        vol_now = float(returns_pips.tail(60).std()) if len(returns_pips) >= 10 else float(returns_pips.std())
        if not np.isfinite(vol_now):
            vol_now = 0.0

        if spread_pips > 1.8:
            thr += 0.03
        if vol_now < 0.08:
            thr += 0.03
        elif vol_now < 0.15:
            thr += 0.01
        elif vol_now > 0.45:
            thr -= 0.02
        elif vol_now > 0.30:
            thr -= 0.01

        return float(np.clip(thr, self.dynamic_threshold_floor, self.dynamic_threshold_cap))

    def decide(self, event_row, ticks, bundle, tabular_models, lstm_model, feature_columns, policy, settings):
        if ticks is None or ticks.empty:
            return None

        df = ticks
        if "time_utc" in df.columns and not df["time_utc"].is_monotonic_increasing:
            df = df.sort_values("time_utc")

        mid = ((df["bid"].astype(float) + df["ask"].astype(float)) / 2.0).dropna()
        if mid.empty:
            return None

        if "time_utc" in df.columns:
            now_ts = pd.to_datetime(df["time_utc"].iloc[-1], utc=True, errors="coerce")
            if pd.isna(now_ts):
                now_ts = pd.Timestamp.now(tz="UTC")
        else:
            now_ts = pd.Timestamp.now(tz="UTC")

        pip = self._pip_size(getattr(settings, "symbol", "EURUSD"))
        spread_pips = 0.0
        if {"bid", "ask"}.issubset(set(df.columns)):
            try:
                spread_pips = float((float(df["ask"].iat[-1]) - float(df["bid"].iat[-1])) / max(1e-12, pip))
            except Exception:
                spread_pips = 0.0

        if self.max_spread_pips > 0:
            try:
                if spread_pips > self.max_spread_pips:
                    return None
            except Exception:
                pass

        current_mid = float(mid.iloc[-1])
        self._update_rewards(now_ts, current_mid, pip)

        candidates: list[dict[str, Any]] = []
        soft_candidates: list[dict[str, Any]] = []
        dec_ema = self.ema_agent.decide(event_row, df, bundle, tabular_models, lstm_model, feature_columns, policy, settings)
        if dec_ema is not None:
            c_ema = {"agent": "ema_rsi", "decision": dec_ema}
            if float(dec_ema.confidence) >= self.min_agent_confidence:
                candidates.append(c_ema)
            elif float(dec_ema.confidence) >= self.min_fallback_confidence:
                soft_candidates.append(c_ema)

        dec_don = self.donchian_agent.decide(event_row, df, bundle, tabular_models, lstm_model, feature_columns, policy, settings)
        if dec_don is not None:
            c_don = {"agent": "donchian", "decision": dec_don}
            if float(dec_don.confidence) >= self.min_agent_confidence:
                candidates.append(c_don)
            elif float(dec_don.confidence) >= self.min_fallback_confidence:
                soft_candidates.append(c_don)

        dec_turtle = self.turtle_agent.decide(event_row, df, bundle, tabular_models, lstm_model, feature_columns, policy, settings)
        if dec_turtle is not None:
            c_turtle = {"agent": "turtle_atr", "decision": dec_turtle}
            if float(dec_turtle.confidence) >= self.min_agent_confidence:
                candidates.append(c_turtle)
            elif float(dec_turtle.confidence) >= self.min_fallback_confidence:
                soft_candidates.append(c_turtle)

        if not candidates:
            candidates = soft_candidates

        if not candidates and self.fundamental_fallback is not None:
            try:
                dec_fund = self.fundamental_fallback.decide(event_row, df, bundle, tabular_models, lstm_model, feature_columns, policy, settings)
            except Exception:
                dec_fund = None
            if dec_fund is not None and float(dec_fund.confidence) >= self.min_fallback_confidence:
                candidates.append({"agent": "fundamental", "decision": dec_fund})

        if not candidates:
            return None

        decision: TradeDecision
        selected_agent: str
        consensus = self._consensus_decision(candidates)
        if consensus is not None:
            same_side = [c for c in candidates if str(c["decision"].side).upper() == str(consensus.side).upper()]
            selected = self._choose_agent(same_side if same_side else candidates)
            selected_agent = str(selected["agent"])
            conf = max(float(consensus.confidence), float(selected["decision"].confidence))
            decision = TradeDecision(side=str(consensus.side), confidence=float(np.clip(conf, 0.5, 0.97)), proba_buy=float(consensus.proba_buy))
        else:
            selected = self._choose_agent(candidates)
            selected_agent = str(selected["agent"])
            decision = selected["decision"]

        if self.decision_threshold_override > 0.0:
            threshold = float(np.clip(self.decision_threshold_override, 0.50, 0.95))
        else:
            threshold = float(self.policy.get("decision_threshold", 0.5))
        threshold = self._apply_dynamic_threshold(threshold, mid, pip, spread_pips)
        if float(decision.confidence) < threshold:
            return None

        if (
            self.signal_cooldown_seconds > 0
            and self._last_signal_side == str(decision.side)
            and self._last_signal_ts is not None
        ):
            elapsed = float((now_ts - self._last_signal_ts).total_seconds())
            if elapsed < float(self.signal_cooldown_seconds):
                return None

        self.pending_trades.append(
            {
                "agent": selected_agent,
                "side": str(decision.side),
                "entry_mid": current_mid,
                "due_time": now_ts + pd.Timedelta(seconds=self.reward_horizon_seconds),
            }
        )
        self._last_signal_side = str(decision.side)
        self._last_signal_ts = now_ts
        return decision


class DonchianBreakoutStrategy(Strategy):
    requires_models: bool = False

    def __init__(
        self,
        lookback_seconds: int = 600,
        breakout_buffer_pips: float = 0.2,
        min_channel_pips: float = 1.0,
        confirm_ticks: int = 1,
        trigger_quantile: float = 0.80,
        session_filter: bool = False,
        sessions: str = "london,ny",
    ):
        self.lookback_seconds = lookback_seconds
        self.breakout_buffer_pips = breakout_buffer_pips
        self.min_channel_pips = min_channel_pips
        self.confirm_ticks = max(1, int(confirm_ticks))
        self.trigger_quantile = float(np.clip(trigger_quantile, 0.55, 0.95))
        self.session_filter = bool(session_filter)
        self.sessions = {s.strip().lower() for s in str(sessions).split(",") if s.strip()}

    @staticmethod
    def _is_in_ny_london_window(ts: pd.Timestamp, sessions: set[str]) -> bool:
        h = int(ts.hour)
        windows = {
            "london": (6, 11),
            "ny": (12, 17),
            "newyork": (12, 17),
        }
        for s in sessions:
            if s not in windows:
                continue
            h0, h1 = windows[s]
            if h0 <= h <= h1:
                return True
        return False

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

        if self.session_filter:
            evt_ts = pd.NaT
            if isinstance(event_row, pd.Series) and "date_utc" in event_row.index:
                evt_ts = pd.to_datetime(event_row.get("date_utc"), utc=True, errors="coerce")
            if pd.isna(evt_ts):
                evt_ts = pd.to_datetime(window["time_utc"].iat[-1], utc=True, errors="coerce")
            if pd.notna(evt_ts) and not self._is_in_ny_london_window(evt_ts, self.sessions or {"london", "ny"}):
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


def _build_strategy(name: str, settings, policy: dict) -> Strategy:
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
            session_filter=bool(settings.donchian_session_filter),
            sessions=str(settings.donchian_sessions),
        )
    if name in {"turtle_atr", "atr_breakout", "vol_breakout", "turtle_atr_breakout"}:
        return TurtleAtrBreakoutStrategy(
            lookback_seconds=int(getattr(settings, "turtle_lookback_seconds", 1200)),
            breakout_buffer_pips=float(getattr(settings, "turtle_breakout_buffer_pips", 0.30)),
            min_channel_pips=float(getattr(settings, "turtle_min_channel_pips", 1.20)),
            confirm_ticks=int(getattr(settings, "turtle_confirm_ticks", 2)),
            atr_period_ticks=int(getattr(settings, "turtle_atr_period_ticks", 120)),
            min_atr_pips=float(getattr(settings, "turtle_min_atr_pips", 0.08)),
            trigger_quantile=float(getattr(settings, "turtle_trigger_quantile", 0.85)),
            trend_ema_span=int(getattr(settings, "turtle_trend_ema_span", 180)),
            max_extension_atr=float(getattr(settings, "turtle_max_extension_atr", 2.50)),
            signal_cooldown_seconds=int(getattr(settings, "turtle_signal_cooldown_seconds", 240)),
        )
    if name in {"fundamental", "fundamental_llm", "macro_llm", "news_llm"}:
        return FundamentalLLMStrategy(settings=settings)
    if name in {"donchian_nylondon", "donchian_session", "donchian_ny_london"}:
        return DonchianBreakoutStrategy(
            lookback_seconds=int(settings.donchian_lookback_seconds),
            breakout_buffer_pips=float(settings.donchian_breakout_buffer_pips),
            min_channel_pips=float(settings.donchian_min_channel_pips),
            confirm_ticks=int(settings.donchian_confirm_ticks),
            trigger_quantile=float(settings.donchian_trigger_quantile),
            session_filter=True,
            sessions="london,ny",
        )
    if name in {"ema_rsi", "ema_rsi_trend", "ema_rsi_active", "crossover_rsi"}:
        return EmaRsiTrendStrategy(
            fast_span=int(settings.ema_fast_span),
            slow_span=int(settings.ema_slow_span),
            rsi_period=int(settings.ema_rsi_period),
            rsi_buy_level=float(settings.ema_rsi_buy_level),
            rsi_sell_level=float(settings.ema_rsi_sell_level),
            min_separation_pips=float(settings.ema_min_separation_pips),
            momentum_lookback_ticks=int(settings.ema_momentum_lookback_ticks),
            min_momentum_pips=float(settings.ema_min_momentum_pips),
            vol_period=int(settings.ema_vol_period),
            min_vol_pips=float(settings.ema_min_vol_pips),
            signal_cooldown_seconds=int(getattr(settings, "ema_signal_cooldown_seconds", 180)),
        )
    if name in {"agentic", "agentic_hybrid", "agentic_ai", "multi_agent"}:
        return AgenticHybridStrategy(settings=settings, policy=policy)
    # default fallback
    return DefaultStrategy()


def is_agent_managed(strategy: Strategy) -> bool:
    return isinstance(strategy, AgentManagedStrategy)


def list_supported_strategies() -> list[str]:
    return [
        "default",
        "zscore",
        "momentum",
        "donchian",
        "donchian_nylondon",
        "turtle_atr",
        "fundamental_llm",
        "ema_rsi",
        "agentic_hybrid",
    ]


def get_strategy(name: str, settings, policy: dict) -> Strategy:
    raw_name = (name or "").strip().lower()
    base = _build_strategy(raw_name, settings, policy)

    manage_all = bool(getattr(settings, "agent_manage_all_strategies", True))
    if not manage_all:
        return base

    if isinstance(base, AgentManagedStrategy):
        return base

    agent_name = raw_name if raw_name else "default"
    return AgentManagedStrategy(agent_name=agent_name, strategy=base)
