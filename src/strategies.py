from __future__ import annotations

import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any

from src.models import ensemble_predict_proba
from src.mt5_executor import TradeDecision


class Strategy:
    requires_models: bool = True
    requires_event: bool = True

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

        proba_buy = float(np.clip(0.5 + direction * min(0.45, 0.20 + 0.30 * confidence), 0.01, 0.99))
        return TradeDecision(side=side, confidence=confidence, proba_buy=proba_buy)


class AgenticHybridStrategy(Strategy):
    requires_models: bool = False
    requires_event: bool = False

    def __init__(self, settings, policy: dict):
        self.policy = policy
        self.learning_rate = float(np.clip(getattr(settings, "agentic_learning_rate", 0.20), 0.01, 1.0))
        self.explore_prob = float(np.clip(getattr(settings, "agentic_explore_prob", 0.10), 0.0, 0.5))
        self.min_agent_confidence = float(np.clip(getattr(settings, "agentic_min_confidence", 0.56), 0.50, 0.95))
        self.reward_horizon_seconds = max(10, int(getattr(settings, "agentic_reward_horizon_seconds", 45)))
        self.reward_target_pips = max(0.5, float(getattr(settings, "agentic_reward_target_pips", 1.2)))

        self.state_path = Path(str(getattr(settings, "agentic_state_path", "models/agentic_state.json")))
        self.weights = {
            "ema_rsi": 1.0,
            "donchian": 1.0,
        }
        self.agent_counts = {"ema_rsi": 0, "donchian": 0}
        self.pending_trades: list[dict[str, Any]] = []

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
        current_mid = float(mid.iloc[-1])
        self._update_rewards(now_ts, current_mid, pip)

        candidates: list[dict[str, Any]] = []
        dec_ema = self.ema_agent.decide(event_row, df, bundle, tabular_models, lstm_model, feature_columns, policy, settings)
        if dec_ema is not None and float(dec_ema.confidence) >= self.min_agent_confidence:
            candidates.append({"agent": "ema_rsi", "decision": dec_ema})

        dec_don = self.donchian_agent.decide(event_row, df, bundle, tabular_models, lstm_model, feature_columns, policy, settings)
        if dec_don is not None and float(dec_don.confidence) >= self.min_agent_confidence:
            candidates.append({"agent": "donchian", "decision": dec_don})

        if not candidates:
            return None

        selected = self._choose_agent(candidates)
        decision = selected["decision"]
        if float(decision.confidence) < float(self.policy.get("decision_threshold", 0.5)):
            return None

        self.pending_trades.append(
            {
                "agent": selected["agent"],
                "side": str(decision.side),
                "entry_mid": current_mid,
                "due_time": now_ts + pd.Timedelta(seconds=self.reward_horizon_seconds),
            }
        )
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
            session_filter=bool(settings.donchian_session_filter),
            sessions=str(settings.donchian_sessions),
        )
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
        )
    if name in {"agentic", "agentic_hybrid", "agentic_ai", "multi_agent"}:
        return AgenticHybridStrategy(settings=settings, policy=policy)
    # default fallback
    return DefaultStrategy()
