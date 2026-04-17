from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.mt5_executor import TradeDecision
from src.strategies import DrivenTradingAgenticSystem


class DrivenTradingAgenticSystemTest(unittest.TestCase):
    def test_selects_best_candidate_and_tracks_pending_trade(self) -> None:
        settings = SimpleNamespace(
            symbol="EURUSD",
            driven_llm_enabled=False,
            driven_llm_mode="off",
            driven_explore_prob=0.0,
            driven_min_agent_confidence=0.55,
            driven_max_spread_pips=3.0,
            driven_state_path="models/test_driven_state.json",
            driven_decision_threshold_floor=0.55,
            driven_decision_threshold_cap=0.82,
        )
        policy = {"decision_threshold": 0.60, "no_trade_band": 0.05}
        system = DrivenTradingAgenticSystem(settings=settings, policy=policy)

        # Deterministic candidate set: make Donchian clearly preferred.
        system.weights["donchian"] = 5.0
        system.weights["ema_rsi"] = 1.0

        system.default_agent.decide = lambda *args, **kwargs: None
        system.zscore_agent.decide = lambda *args, **kwargs: None
        system.momentum_agent.decide = lambda *args, **kwargs: None
        system.turtle_agent.decide = lambda *args, **kwargs: None
        system.ema_agent.decide = lambda *args, **kwargs: TradeDecision(side="BUY", confidence=0.70, proba_buy=0.66)
        system.donchian_agent.decide = lambda *args, **kwargs: TradeDecision(side="SELL", confidence=0.82, proba_buy=0.32)

        ticks = pd.DataFrame(
            {
                "time_utc": pd.date_range("2026-01-01", periods=120, freq="s", tz="UTC"),
                "bid": np.linspace(1.1000, 1.1010, 120),
                "ask": np.linspace(1.1001, 1.1011, 120),
            }
        )
        event_row = pd.Series({"event_id": "evt_1", "symbol": "EURUSD"})
        bundle = SimpleNamespace(X_tabular=pd.DataFrame(), X_seq=np.zeros((1, 1, 1), dtype=np.float32))

        decision = system.decide(
            event_row=event_row,
            ticks=ticks,
            bundle=bundle,
            tabular_models={},
            lstm_model=None,
            feature_columns=[],
            policy=policy,
            settings=settings,
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.side, "SELL")
        self.assertGreaterEqual(float(decision.confidence), 0.60)
        self.assertEqual(len(system.pending_trades), 1)
        self.assertEqual(system.pending_trades[0]["agent"], "donchian")


if __name__ == "__main__":
    unittest.main()
