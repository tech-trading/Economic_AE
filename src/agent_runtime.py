from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.mt5_executor import TradeDecision


@dataclass
class RuntimeVerdict:
    allow_trade: bool
    reason: str = ""
    confidence_floor: float = 0.0


class TradingAgentRuntime:
    """Runtime orchestration layer for agents/trading/*.md roles.

    This is a lightweight executable framework that maps each documented agent role
    to practical runtime checks and annotations in live trading flow.
    """

    def __init__(self, settings, policy: dict[str, float]):
        self.settings = settings
        self.policy = policy
        self.enabled = bool(getattr(settings, "agent_runtime_enabled", True))
        self.strict = bool(getattr(settings, "agent_runtime_strict", True))
        self.min_confidence = float(np.clip(getattr(settings, "agent_runtime_min_confidence", 0.55), 0.5, 0.99))
        self.max_spread_pips = float(max(0.0, getattr(settings, "agent_runtime_max_spread_pips", 2.2)))
        self._last_notes: dict[str, Any] = {}
        self.enabled_agents: dict[str, bool] = {
            "strategy_architect": bool(getattr(settings, "enable_strategy_architect_agent", True)),
            "market_data": bool(getattr(settings, "enable_market_data_agent", True)),
            "backtesting": bool(getattr(settings, "enable_backtesting_agent", True)),
            "risk_manager": bool(getattr(settings, "enable_risk_manager_agent", True)),
            "optimizer": bool(getattr(settings, "enable_optimizer_agent", True)),
            "llm_meta": bool(getattr(settings, "enable_llm_meta_agent", True)),
            "execution": bool(getattr(settings, "enable_execution_agent", True)),
            "monitoring": bool(getattr(settings, "enable_monitoring_agent", True)),
            "qa": bool(getattr(settings, "enable_qa_agent", True)),
        }

    def _is_enabled(self, name: str) -> bool:
        return bool(self.enabled_agents.get(name, True))

    @staticmethod
    def _pip_size(symbol: str) -> float:
        s = str(symbol or "").upper()
        return 0.01 if "JPY" in s else 0.0001

    def _spread_pips_from_ticks(self, ticks: pd.DataFrame, symbol: str) -> float:
        if ticks is None or ticks.empty:
            return 0.0
        if not {"bid", "ask"}.issubset(set(ticks.columns)):
            return 0.0
        pip = self._pip_size(symbol)
        try:
            ask = float(ticks["ask"].iat[-1])
            bid = float(ticks["bid"].iat[-1])
            if ask <= 0 or bid <= 0 or ask < bid:
                return float("inf")
            spread = float((ask - bid) / max(1e-12, pip))
            if not np.isfinite(spread):
                return float("inf")
            # Ignore clearly corrupted snapshots (rare MT5 zeros/spikes).
            if spread > 100.0:
                return float("inf")
            return spread
        except Exception:
            return float("inf")

    def status_note(self) -> str:
        if not self._last_notes:
            return ""
        return ";".join([f"{k}={v}" for k, v in self._last_notes.items() if v not in {"", None}])

    def pre_decision(self, *, event_row: pd.Series, ticks: pd.DataFrame, symbol: str, strategy_name: str) -> RuntimeVerdict:
        if not self.enabled:
            return RuntimeVerdict(allow_trade=True)

        # market_data_agent + qa_agent checks
        if self._is_enabled("market_data") and (ticks is None or ticks.empty):
            self._last_notes = {"orchestrator": "pre_decision", "blocker": "market_data_agent", "reason": "no_ticks"}
            return RuntimeVerdict(allow_trade=not self.strict, reason="no_ticks")
        required = {"time_utc", "bid", "ask"}
        if self._is_enabled("qa") and not required.issubset(set(ticks.columns)):
            self._last_notes = {"orchestrator": "pre_decision", "blocker": "qa_agent", "reason": "missing_tick_columns"}
            return RuntimeVerdict(allow_trade=not self.strict, reason="missing_tick_columns")

        spread_pips = self._spread_pips_from_ticks(ticks, symbol)
        if self._is_enabled("risk_manager") and self.max_spread_pips > 0 and spread_pips > self.max_spread_pips:
            self._last_notes = {
                "orchestrator": "pre_decision",
                "blocker": "risk_manager",
                "reason": "spread_guard",
                "spread_pips": round(spread_pips, 4),
                "max_spread_pips": round(self.max_spread_pips, 4),
            }
            return RuntimeVerdict(allow_trade=not self.strict, reason="spread_guard")

        self._last_notes = {
            "orchestrator": "pre_decision",
            "strategy_architect": strategy_name if self._is_enabled("strategy_architect") else "disabled",
            "market_data_agent": "ok" if self._is_enabled("market_data") else "disabled",
            "risk_manager": "ok" if self._is_enabled("risk_manager") else "disabled",
        }
        return RuntimeVerdict(allow_trade=True)

    def post_decision(
        self,
        *,
        decision: TradeDecision,
        event_row: pd.Series,
        ticks: pd.DataFrame,
        symbol: str,
    ) -> RuntimeVerdict:
        if not self.enabled:
            return RuntimeVerdict(allow_trade=True)

        if decision is None:
            self._last_notes = {"orchestrator": "post_decision", "llm_meta_agent": "no_signal"}
            return RuntimeVerdict(allow_trade=False, reason="no_signal")

        side = str(getattr(decision, "side", "")).upper()
        conf = float(getattr(decision, "confidence", 0.0))
        proba = float(getattr(decision, "proba_buy", 0.5))

        if self._is_enabled("qa") and side not in {"BUY", "SELL"}:
            self._last_notes = {"orchestrator": "post_decision", "blocker": "qa_agent", "reason": "invalid_side"}
            return RuntimeVerdict(allow_trade=not self.strict, reason="invalid_side")

        if self._is_enabled("qa") and not (0.0 <= proba <= 1.0):
            self._last_notes = {"orchestrator": "post_decision", "blocker": "qa_agent", "reason": "invalid_proba"}
            return RuntimeVerdict(allow_trade=not self.strict, reason="invalid_proba")

        floor = max(float(self.policy.get("decision_threshold", 0.5)), self.min_confidence)
        if self._is_enabled("risk_manager") and conf < floor:
            self._last_notes = {
                "orchestrator": "post_decision",
                "blocker": "risk_manager",
                "reason": "confidence_floor",
                "confidence": round(conf, 4),
                "floor": round(floor, 4),
            }
            return RuntimeVerdict(allow_trade=not self.strict, reason="confidence_floor", confidence_floor=floor)

        self._last_notes = {
            "orchestrator": "post_decision",
            "llm_meta_agent": "checked" if self._is_enabled("llm_meta") else "disabled",
            "qa_agent": "ok" if self._is_enabled("qa") else "disabled",
            "risk_manager": "ok" if self._is_enabled("risk_manager") else "disabled",
        }
        return RuntimeVerdict(allow_trade=True, confidence_floor=floor)

    def pre_execution(self, *, decision: TradeDecision, symbol: str, event_id: str, now: datetime) -> RuntimeVerdict:
        if not self.enabled:
            return RuntimeVerdict(allow_trade=True)

        if self._is_enabled("execution") and decision is None:
            self._last_notes = {"orchestrator": "pre_execution", "blocker": "execution_agent", "reason": "missing_decision"}
            return RuntimeVerdict(allow_trade=False, reason="missing_decision")

        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        self._last_notes = {
            "orchestrator": "pre_execution",
            "execution_agent": "ok" if self._is_enabled("execution") else "disabled",
            "monitoring_agent": "tracking" if self._is_enabled("monitoring") else "disabled",
            "event_id": event_id,
            "symbol": symbol,
        }
        return RuntimeVerdict(allow_trade=True)
