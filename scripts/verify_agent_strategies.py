from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.config import settings
from src.strategies import get_strategy, is_agent_managed, list_supported_strategies


class DummyTabularModel:
    def predict_proba(self, X):
        n = len(X)
        probs = np.zeros((n, 2), dtype=np.float32)
        probs[:, 0] = 0.40
        probs[:, 1] = 0.60
        return probs


def _build_ticks(n_rows: int = 900) -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    times = pd.date_range(base, periods=n_rows, freq="s", tz="UTC")

    drift = np.linspace(0.0, 0.0009, n_rows)
    noise = np.random.normal(loc=0.0, scale=0.00003, size=n_rows)
    mid = 1.1000 + drift + noise.cumsum() * 0.05

    spread = 0.00008
    bid = mid - spread / 2.0
    ask = mid + spread / 2.0

    return pd.DataFrame(
        {
            "time_utc": times,
            "bid": bid.astype(float),
            "ask": ask.astype(float),
        }
    )


def _build_bundle(feature_columns: list[str]) -> SimpleNamespace:
    tab = pd.DataFrame([[0.2, 0.1, -0.3]], columns=feature_columns)
    seq = np.ones((1, 20, 1), dtype=np.float32)
    return SimpleNamespace(X_tabular=tab, X_seq=seq)


def main() -> int:
    np.random.seed(7)

    cfg = SimpleNamespace(**vars(settings))
    cfg.agent_manage_all_strategies = True
    cfg.agentic_state_path = os.path.join("models", "agentic_state_verify_tmp.json")

    policy = {
        "decision_threshold": 0.50,
        "no_trade_band": 0.00,
    }

    feature_columns = ["f1", "f2", "f3"]
    tabular_models = {"dummy": DummyTabularModel()}
    bundle = _build_bundle(feature_columns)
    ticks = _build_ticks()

    event_row = pd.Series(
        {
            "event_id": "verify_1",
            "date_utc": pd.Timestamp("2026-01-01T00:15:00Z").isoformat(),
            "name": "verification_event",
            "currency": "EUR",
            "importance": 2,
        }
    )

    names = list_supported_strategies()
    rows = []
    failed = False

    for name in names:
        try:
            strat = get_strategy(name, cfg, policy)
            managed = is_agent_managed(strat)
            if not managed:
                failed = True
                rows.append((name, "FAIL", "not_agent_managed"))
                continue

            decision = strat.decide(event_row, ticks, bundle, tabular_models, None, feature_columns, policy, cfg)
            status = "OK"
            detail = "none"
            if decision is not None:
                detail = f"side={decision.side},confidence={float(decision.confidence):.4f}"

            rows.append((name, status, detail))
        except Exception as ex:
            failed = True
            rows.append((name, "FAIL", str(ex)))

    print("strategy,status,detail")
    for name, status, detail in rows:
        print(f"{name},{status},{detail}")

    tmp_state = cfg.agentic_state_path
    if isinstance(tmp_state, str) and os.path.exists(tmp_state):
        try:
            os.remove(tmp_state)
        except OSError:
            pass

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
