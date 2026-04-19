from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.agent_runtime import TradingAgentRuntime
from src.mt5_executor import TradeDecision


class TradingAgentRuntimeTest(unittest.TestCase):
    def test_pre_decision_blocks_invalid_tick_schema(self) -> None:
        settings = SimpleNamespace(
            agent_runtime_enabled=True,
            agent_runtime_strict=True,
            agent_runtime_min_confidence=0.55,
            agent_runtime_max_spread_pips=2.2,
        )
        policy = {"decision_threshold": 0.60, "no_trade_band": 0.05}
        rt = TradingAgentRuntime(settings=settings, policy=policy)

        ticks = pd.DataFrame({"bid": [1.1, 1.2], "ask": [1.1001, 1.2001]})
        verdict = rt.pre_decision(
            event_row=pd.Series({"event_id": "evt1"}),
            ticks=ticks,
            symbol="EURUSD",
            strategy_name="driven_trading_agentic_system",
        )

        self.assertFalse(verdict.allow_trade)
        self.assertEqual(verdict.reason, "missing_tick_columns")

    def test_post_decision_blocks_low_confidence(self) -> None:
        settings = SimpleNamespace(
            agent_runtime_enabled=True,
            agent_runtime_strict=True,
            agent_runtime_min_confidence=0.70,
            agent_runtime_max_spread_pips=2.2,
        )
        policy = {"decision_threshold": 0.60, "no_trade_band": 0.05}
        rt = TradingAgentRuntime(settings=settings, policy=policy)

        ticks = pd.DataFrame(
            {
                "time_utc": pd.date_range("2026-01-01", periods=5, freq="s", tz="UTC"),
                "bid": np.linspace(1.1000, 1.1004, 5),
                "ask": np.linspace(1.1001, 1.1005, 5),
            }
        )
        decision = TradeDecision(side="BUY", confidence=0.62, proba_buy=0.64)
        verdict = rt.post_decision(
            decision=decision,
            event_row=pd.Series({"event_id": "evt2"}),
            ticks=ticks,
            symbol="EURUSD",
        )

        self.assertFalse(verdict.allow_trade)
        self.assertEqual(verdict.reason, "confidence_floor")
        self.assertGreaterEqual(verdict.confidence_floor, 0.70)

    def test_pre_execution_allows_valid_decision(self) -> None:
        settings = SimpleNamespace(
            agent_runtime_enabled=True,
            agent_runtime_strict=True,
            agent_runtime_min_confidence=0.55,
            agent_runtime_max_spread_pips=2.2,
        )
        policy = {"decision_threshold": 0.60, "no_trade_band": 0.05}
        rt = TradingAgentRuntime(settings=settings, policy=policy)

        verdict = rt.pre_execution(
            decision=TradeDecision(side="SELL", confidence=0.80, proba_buy=0.20),
            symbol="EURUSD",
            event_id="evt3",
            now=datetime.now(timezone.utc),
        )

        self.assertTrue(verdict.allow_trade)

    def test_disabling_qa_agent_bypasses_schema_block(self) -> None:
        settings = SimpleNamespace(
            agent_runtime_enabled=True,
            agent_runtime_strict=True,
            agent_runtime_min_confidence=0.55,
            agent_runtime_max_spread_pips=2.2,
            enable_qa_agent=False,
            enable_market_data_agent=True,
            enable_risk_manager_agent=True,
            enable_strategy_architect_agent=True,
            enable_backtesting_agent=True,
            enable_optimizer_agent=True,
            enable_llm_meta_agent=True,
            enable_execution_agent=True,
            enable_monitoring_agent=True,
        )
        policy = {"decision_threshold": 0.60, "no_trade_band": 0.05}
        rt = TradingAgentRuntime(settings=settings, policy=policy)

        # Missing time_utc: with QA disabled, this should not block.
        ticks = pd.DataFrame({"bid": [1.1, 1.2], "ask": [1.1001, 1.2001]})
        verdict = rt.pre_decision(
            event_row=pd.Series({"event_id": "evt4"}),
            ticks=ticks,
            symbol="EURUSD",
            strategy_name="driven",
        )

        self.assertTrue(verdict.allow_trade)

    def test_disabling_risk_manager_bypasses_low_confidence_block(self) -> None:
        settings = SimpleNamespace(
            agent_runtime_enabled=True,
            agent_runtime_strict=True,
            agent_runtime_min_confidence=0.80,
            agent_runtime_max_spread_pips=2.2,
            enable_risk_manager_agent=False,
            enable_qa_agent=True,
            enable_market_data_agent=True,
            enable_strategy_architect_agent=True,
            enable_backtesting_agent=True,
            enable_optimizer_agent=True,
            enable_llm_meta_agent=True,
            enable_execution_agent=True,
            enable_monitoring_agent=True,
        )
        policy = {"decision_threshold": 0.60, "no_trade_band": 0.05}
        rt = TradingAgentRuntime(settings=settings, policy=policy)

        ticks = pd.DataFrame(
            {
                "time_utc": pd.date_range("2026-01-01", periods=5, freq="s", tz="UTC"),
                "bid": np.linspace(1.1000, 1.1004, 5),
                "ask": np.linspace(1.1001, 1.1005, 5),
            }
        )
        decision = TradeDecision(side="BUY", confidence=0.62, proba_buy=0.64)
        verdict = rt.post_decision(
            decision=decision,
            event_row=pd.Series({"event_id": "evt5"}),
            ticks=ticks,
            symbol="EURUSD",
        )

        self.assertTrue(verdict.allow_trade)


if __name__ == "__main__":
    unittest.main()
