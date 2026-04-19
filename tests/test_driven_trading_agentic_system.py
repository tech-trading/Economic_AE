from __future__ import annotations

import os
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

    def test_llm_confirm_mode_can_veto_conflicting_signal(self) -> None:
        settings = SimpleNamespace(
            symbol="EURUSD",
            driven_llm_enabled=True,
            driven_llm_mode="confirm",
            driven_llm_min_confidence=0.62,
            driven_llm_veto_gap=0.08,
            driven_explore_prob=0.0,
            driven_min_agent_confidence=0.55,
            driven_max_spread_pips=3.0,
            driven_state_path="models/test_driven_state_llm.json",
            driven_decision_threshold_floor=0.55,
            driven_decision_threshold_cap=0.82,
        )
        policy = {"decision_threshold": 0.60, "no_trade_band": 0.05}
        system = DrivenTradingAgenticSystem(settings=settings, policy=policy)

        system.default_agent.decide = lambda *args, **kwargs: None
        system.zscore_agent.decide = lambda *args, **kwargs: None
        system.momentum_agent.decide = lambda *args, **kwargs: None
        system.turtle_agent.decide = lambda *args, **kwargs: None
        system.ema_agent.decide = lambda *args, **kwargs: None
        system.donchian_agent.decide = lambda *args, **kwargs: TradeDecision(side="BUY", confidence=0.66, proba_buy=0.64)

        assert system.llm_agent is not None
        system.llm_agent.decide = lambda *args, **kwargs: TradeDecision(side="SELL", confidence=0.90, proba_buy=0.22)

        ticks = pd.DataFrame(
            {
                "time_utc": pd.date_range("2026-02-01", periods=120, freq="s", tz="UTC"),
                "bid": np.linspace(1.1000, 1.1010, 120),
                "ask": np.linspace(1.1001, 1.1011, 120),
            }
        )
        event_row = pd.Series({"event_id": "evt_veto", "symbol": "EURUSD"})
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

        self.assertIsNone(decision)
        self.assertEqual(len(system.pending_trades), 0)

    def test_underperforming_agent_is_temporarily_disabled(self) -> None:
        state_path = "models/test_driven_state_disable.json"
        if os.path.exists(state_path):
            os.remove(state_path)

        settings = SimpleNamespace(
            symbol="EURUSD",
            driven_llm_enabled=False,
            driven_llm_mode="off",
            driven_state_path=state_path,
            driven_min_samples_disable=5,
            driven_disable_threshold=-0.10,
            driven_disable_cooldown_minutes=30,
        )
        policy = {"decision_threshold": 0.60, "no_trade_band": 0.05}
        system = DrivenTradingAgenticSystem(settings=settings, policy=policy)

        system.stats["donchian"]["trades"] = 4
        system.stats["donchian"]["wins"] = 1
        system.stats["donchian"]["losses"] = 3
        system.stats["donchian"]["avg_reward"] = -0.20
        system.stats["donchian"]["disable_until"] = ""

        now_ts = pd.Timestamp("2026-03-01T10:00:00Z")
        pip = system._pip_size(settings.symbol)
        system.pending_trades.append(
            {
                "agent": "donchian",
                "side": "BUY",
                "entry_mid": 1.2000,
                "due_time": now_ts - pd.Timedelta(seconds=1),
                "cost_pips": 0.10,
            }
        )

        system._update_rewards(current_time=now_ts, current_mid=1.1980, pip=pip)

        st = system.stats["donchian"]
        self.assertGreaterEqual(int(st["trades"]), 5)
        self.assertLess(float(st["avg_reward"]), -0.10)
        self.assertTrue(str(st.get("disable_until", "")))
        self.assertTrue(system._is_agent_disabled("donchian", now_ts))

    def test_llm_blend_mode_can_flip_side_on_strong_conflict(self) -> None:
        settings = SimpleNamespace(
            symbol="EURUSD",
            driven_llm_enabled=True,
            driven_llm_mode="blend",
            driven_llm_min_confidence=0.62,
            driven_llm_veto_gap=0.08,
            driven_explore_prob=0.0,
            driven_min_agent_confidence=0.55,
            driven_max_spread_pips=3.0,
            driven_state_path="models/test_driven_state_blend.json",
            driven_decision_threshold_floor=0.55,
            driven_decision_threshold_cap=0.82,
        )
        policy = {"decision_threshold": 0.60, "no_trade_band": 0.05}
        system = DrivenTradingAgenticSystem(settings=settings, policy=policy)

        system.default_agent.decide = lambda *args, **kwargs: None
        system.zscore_agent.decide = lambda *args, **kwargs: None
        system.momentum_agent.decide = lambda *args, **kwargs: None
        system.turtle_agent.decide = lambda *args, **kwargs: None
        system.ema_agent.decide = lambda *args, **kwargs: None
        # Base technical signal is BUY but weak edge.
        system.donchian_agent.decide = lambda *args, **kwargs: TradeDecision(side="BUY", confidence=0.74, proba_buy=0.54)

        assert system.llm_agent is not None
        # Strong SELL conviction should dominate blend and flip side.
        system.llm_agent.decide = lambda *args, **kwargs: TradeDecision(side="SELL", confidence=0.95, proba_buy=0.02)

        ticks = pd.DataFrame(
            {
                "time_utc": pd.date_range("2026-04-01", periods=120, freq="s", tz="UTC"),
                "bid": np.linspace(1.1500, 1.1510, 120),
                "ask": np.linspace(1.1501, 1.1511, 120),
            }
        )
        event_row = pd.Series({"event_id": "evt_blend", "symbol": "EURUSD"})
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
        self.assertLess(float(decision.proba_buy), 0.50)
        self.assertEqual(len(system.pending_trades), 1)


if __name__ == "__main__":
    unittest.main()
