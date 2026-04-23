from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.mt5_executor import TradeDecision
from src.strategies import DrivenTradingAgenticSystem


class DrivenActivationLogicTest(unittest.TestCase):
    def test_no_signal_relaxation_allows_trade_after_silence(self) -> None:
        state_path = "models/test_driven_activation_state.json"
        if os.path.exists(state_path):
            os.remove(state_path)

        settings = SimpleNamespace(
            symbol="EURUSD",
            driven_llm_enabled=False,
            driven_llm_mode="off",
            driven_state_path=state_path,
            driven_decision_threshold_floor=0.55,
            driven_decision_threshold_cap=0.82,
            driven_no_signal_relax_seconds=60,
            driven_max_relaxation=0.10,
            driven_min_agent_confidence=0.55,
            driven_explore_prob=0.0,
            driven_max_spread_pips=3.0,
        )
        policy = {"decision_threshold": 0.66, "no_trade_band": 0.05}
        system = DrivenTradingAgenticSystem(settings=settings, policy=policy)
        for name in system.stats:
            system.stats[name]["disable_until"] = ""

        # Force only one candidate with confidence below base threshold but above relaxed threshold.
        system.default_agent.decide = lambda *args, **kwargs: None
        system.zscore_agent.decide = lambda *args, **kwargs: None
        system.momentum_agent.decide = lambda *args, **kwargs: None
        system.turtle_agent.decide = lambda *args, **kwargs: None
        system.ema_agent.decide = lambda *args, **kwargs: None
        system.donchian_agent.decide = lambda *args, **kwargs: TradeDecision(side="BUY", confidence=0.62, proba_buy=0.62)

        now = pd.Timestamp("2026-04-20T08:00:00Z")
        system._last_signal_ts = now - pd.Timedelta(seconds=300)

        ticks = pd.DataFrame(
            {
                "time_utc": pd.date_range(now - pd.Timedelta(seconds=119), periods=120, freq="s", tz="UTC"),
                "bid": np.linspace(1.1000, 1.1010, 120),
                "ask": np.linspace(1.1001, 1.1011, 120),
            }
        )
        event_row = pd.Series({"event_id": "evt_relax", "symbol": "EURUSD"})
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
        self.assertEqual(decision.side, "BUY")

    def test_regime_and_score_metadata_populated(self) -> None:
        settings = SimpleNamespace(
            symbol="EURUSD",
            driven_llm_enabled=False,
            driven_llm_mode="off",
            driven_state_path="models/test_driven_activation_state_regime.json",
            driven_decision_threshold_floor=0.55,
            driven_decision_threshold_cap=0.82,
            driven_min_agent_confidence=0.55,
            driven_explore_prob=0.0,
            driven_max_spread_pips=3.0,
        )
        policy = {"decision_threshold": 0.60, "no_trade_band": 0.05}
        system = DrivenTradingAgenticSystem(settings=settings, policy=policy)

        system.default_agent.decide = lambda *args, **kwargs: None
        system.zscore_agent.decide = lambda *args, **kwargs: None
        system.momentum_agent.decide = lambda *args, **kwargs: None
        system.turtle_agent.decide = lambda *args, **kwargs: None
        system.ema_agent.decide = lambda *args, **kwargs: TradeDecision(side="BUY", confidence=0.72, proba_buy=0.70)
        system.donchian_agent.decide = lambda *args, **kwargs: TradeDecision(side="BUY", confidence=0.66, proba_buy=0.62)

        ticks = pd.DataFrame(
            {
                "time_utc": pd.date_range("2026-04-20", periods=120, freq="s", tz="UTC"),
                "bid": np.linspace(1.1000, 1.1020, 120),
                "ask": np.linspace(1.1001, 1.1021, 120),
            }
        )
        event_row = pd.Series({"event_id": "evt_regime", "symbol": "EURUSD"})
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
        status = system.get_agent_status()
        self.assertTrue(str(status.get("regime", "")))
        scores = status.get("scores", [])
        self.assertTrue(isinstance(scores, list) and len(scores) >= 1)

    def test_exit_profile_overrides_are_set_for_regime(self) -> None:
        settings = SimpleNamespace(
            symbol="EURUSD",
            driven_llm_enabled=False,
            driven_llm_mode="off",
            driven_state_path="models/test_driven_activation_state_exit_profile.json",
            driven_decision_threshold_floor=0.55,
            driven_decision_threshold_cap=0.82,
            driven_min_agent_confidence=0.55,
            driven_explore_prob=0.0,
            driven_max_spread_pips=3.0,
            driven_regime_tp_low_vol_pips=3.2,
            driven_regime_tp_mean_rev_pips=4.2,
            driven_regime_tp_trend_pips=6.0,
            driven_regime_tp_high_vol_pips=7.0,
            driven_regime_tp_event_risk_pips=3.6,
            driven_regime_tp_confidence_bonus_pips=1.0,
            driven_regime_tp_spread_penalty_factor=0.35,
            driven_regime_trail_ratio=0.55,
            driven_regime_trail_activation_ratio=0.45,
            driven_regime_trail_min_pips=1.2,
        )
        policy = {"decision_threshold": 0.60, "no_trade_band": 0.05}
        system = DrivenTradingAgenticSystem(settings=settings, policy=policy)

        system.default_agent.decide = lambda *args, **kwargs: None
        system.zscore_agent.decide = lambda *args, **kwargs: None
        system.momentum_agent.decide = lambda *args, **kwargs: None
        system.turtle_agent.decide = lambda *args, **kwargs: None
        system.ema_agent.decide = lambda *args, **kwargs: TradeDecision(side="BUY", confidence=0.78, proba_buy=0.76)
        system.donchian_agent.decide = lambda *args, **kwargs: None

        ticks = pd.DataFrame(
            {
                "time_utc": pd.date_range("2026-04-20", periods=140, freq="s", tz="UTC"),
                "bid": np.linspace(1.1000, 1.1030, 140),
                "ask": np.linspace(1.1001, 1.1031, 140),
            }
        )
        event_row = pd.Series({"event_id": "evt_exit_profile", "symbol": "EURUSD"})
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
        self.assertIsNotNone(decision.tp_pips_override)
        self.assertIsNotNone(decision.trailing_stop_pips_override)
        self.assertIsNotNone(decision.trailing_activation_pips_override)
        self.assertGreater(float(decision.tp_pips_override), 0.0)


if __name__ == "__main__":
    unittest.main()
