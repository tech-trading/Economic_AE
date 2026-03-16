"""Simple backtest runner to compare strategies over stored events and ticks.

Usage: run this script from repo root with the project's virtualenv active.
"""
from __future__ import annotations

import pandas as pd
import os

from src.config import settings
from src.models import load_artifacts
from src.strategies import get_strategy
from src.feature_engineering import build_event_dataset
import json
import os


def load_events(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_ticks(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time_utc"], infer_datetime_format=True)
    # normalize to UTC tz to match feature builder expectations
    try:
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    except Exception:
        pass
    return df


def main():
    events = load_events(settings.events_csv)
    ticks = load_ticks(settings.market_csv) if os.path.exists(settings.market_csv) else pd.DataFrame()

    tabular, lstm, feature_columns = load_artifacts(settings.model_dir)

    # Load optimized policy from models if available, otherwise fall back to settings
    policy_path = os.path.join(settings.model_dir, "trading_policy.json")
    if os.path.exists(policy_path):
        try:
            with open(policy_path, "r", encoding="utf-8") as f:
                policy = json.load(f)
        except Exception:
            policy = {
                "decision_threshold": settings.decision_threshold,
                "no_trade_band": settings.no_trade_band,
            }
    else:
        policy = {
            "decision_threshold": settings.decision_threshold,
            "no_trade_band": settings.no_trade_band,
        }

    strat = get_strategy(settings.strategy, settings, policy)

    # Build datasets once for all events (efficient)
    bundle = build_event_dataset(events, ticks, lookback_seconds=settings.lookback_seconds)
    signals = []
    print(f"Built bundle: samples={getattr(bundle.X_tabular, 'shape', None)}")
    if bundle.X_tabular.empty:
        print("No samples built for backtest. Exiting.")
    else:
        # index events by id for lookup
        events_by_id = events.set_index("event_id")

        # prepare time index for ticks to allow fast slicing
        times = None
        if not ticks.empty:
            ticks = ticks.sort_values("time_utc")
            times = ticks["time_utc"]

        for i in range(bundle.X_tabular.shape[0]):
            ev_id = bundle.event_ids.iloc[i]
            try:
                ev_row = events_by_id.loc[ev_id]
            except Exception:
                ev_row = None

            event_time = bundle.event_times.iloc[i]

            # fast slice of ticks up to event_time
            if times is not None:
                idx = int(times.searchsorted(event_time, side="right"))
                ticks_up_to = ticks.iloc[:idx].copy()
            else:
                ticks_up_to = pd.DataFrame()

            dec = strat.decide(ev_row if ev_row is not None else pd.Series(dtype=object), ticks_up_to, bundle, tabular, lstm, feature_columns, policy, settings)
            if dec is not None:
                signals.append({"event_id": ev_id, "time": str(event_time), "side": dec.side, "confidence": dec.confidence, "proba_buy": getattr(dec, "proba_buy", 0.5)})
            else:
                # optional: could collect why filtered; for now we count silent drops
                pass

    df = pd.DataFrame(signals)
    print(f"Samples processed: {getattr(bundle.X_tabular, 'shape', None)} | Signals generated: {len(df)}")
    out = os.path.join(settings.data_dir, "backtest_signals.csv")
    os.makedirs(settings.data_dir, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Backtest finished. Signals: {len(df)}. Saved to {out}")


if __name__ == "__main__":
    main()
