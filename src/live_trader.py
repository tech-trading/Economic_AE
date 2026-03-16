from __future__ import annotations

import time
from datetime import datetime, timezone
import os

import numpy as np
import pandas as pd

from src.calendar_sources import fetch_and_store_events
from src.config import settings
from src.feature_engineering import build_event_dataset
from src.models import ensemble_predict_proba, load_artifacts
from src.mt5_executor import MT5Executor, TradeDecision
from src.policy import load_policy
from src.strategies import get_strategy


class LiveTrader:
    def __init__(self) -> None:
        self.executor = MT5Executor()
        self.tabular_models, self.lstm_model, self.feature_columns = load_artifacts(settings.model_dir)
        self.policy = load_policy(
            settings.model_dir,
            default_threshold=settings.decision_threshold,
            default_no_trade_band=settings.no_trade_band,
        )

        # strategy selection
        self.strategy = get_strategy(settings.strategy, settings, self.policy)

        if not self.tabular_models and self.lstm_model is None:
            raise RuntimeError("No models loaded. Train first.")

    def run(self) -> None:
        self.executor.initialize()

        try:
            mode = "PAPER" if settings.paper_trading else "LIVE"
            print(f"Starting live trader in {mode} mode with policy {self.policy}...")
            events = fetch_and_store_events(days_ahead=14)
            if events.empty:
                print("No relevant upcoming events found.")
                return

            already_traded_event_ids: set[str] = set()

            while True:
                now = datetime.now(timezone.utc)

                upcoming = events[events["date_utc"].map(pd.Timestamp) > pd.Timestamp(now)]
                if upcoming.empty:
                    print("No more events in memory. Refreshing calendar...")
                    events = fetch_and_store_events(days_ahead=14)
                    time.sleep(3)
                    continue

                next_event = upcoming.iloc[0]
                event_id = str(next_event["event_id"])
                event_time = pd.to_datetime(next_event["date_utc"], utc=True).to_pydatetime()
                trigger_time = event_time - pd.Timedelta(seconds=settings.seconds_before_event)

                self.executor.apply_trailing_stop(settings.symbol)

                if event_id not in already_traded_event_ids and now >= trigger_time and now < event_time:
                    decision = self._build_decision(next_event)
                    if decision is not None:
                        if decision.confidence < self.policy["decision_threshold"]:
                            time.sleep(1)
                            continue

                        if abs(decision.proba_buy - 0.5) < self.policy["no_trade_band"]:
                            time.sleep(1)
                            continue

                        if settings.paper_trading:
                            self._record_paper_trade(event_id, next_event, decision)
                            print(f"[PAPER] Signal for event {event_id}: {decision}")
                        else:
                            open_positions = self.executor.count_open_positions(settings.symbol)
                            if open_positions >= settings.max_open_positions:
                                print(
                                    f"Skipping event {event_id}: open positions {open_positions} "
                                    f">= MAX_OPEN_POSITIONS={settings.max_open_positions}"
                                )
                                already_traded_event_ids.add(event_id)
                                time.sleep(1)
                                continue

                            print(f"Sending order for event {event_id}: {decision}")
                            self.executor.send_market_order(settings.symbol, decision)

                        already_traded_event_ids.add(event_id)

                time.sleep(1)

        finally:
            self.executor.shutdown()

    def _build_decision(self, event_row: pd.Series) -> TradeDecision | None:
        ticks = self.executor.get_recent_ticks(settings.symbol, seconds=settings.lookback_seconds + 120)
        if ticks.empty:
            return None

        event_df = pd.DataFrame([event_row.to_dict()])
        bundle = build_event_dataset(event_df, ticks, lookback_seconds=settings.lookback_seconds)
        if bundle.X_tabular.empty:
            return None
        # delegate to selected strategy
        return self.strategy.decide(event_row, ticks, bundle, self.tabular_models, self.lstm_model, self.feature_columns, self.policy, settings)

    def _record_paper_trade(self, event_id: str, event_row: pd.Series, decision: TradeDecision) -> None:
        os.makedirs(settings.data_dir, exist_ok=True)
        path = os.path.join(settings.data_dir, "paper_trades.csv")
        row = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "event_id": event_id,
            "event_time_utc": str(event_row.get("date_utc", "")),
            "event_name": str(event_row.get("name", "")),
            "event_currency": str(event_row.get("currency", "")),
            "event_importance": event_row.get("importance", ""),
            "symbol": settings.symbol,
            "side": decision.side,
            "confidence": decision.confidence,
            "proba_buy": getattr(decision, "proba_buy", 0.5),
            "mode": "PAPER",
        }

        df = pd.DataFrame([row])
        if os.path.exists(path):
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)
