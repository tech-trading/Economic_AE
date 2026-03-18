from __future__ import annotations

import time
from datetime import datetime, timezone
import json
import os
from types import SimpleNamespace

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

        if self.strategy.requires_models and not self.tabular_models and self.lstm_model is None:
            raise RuntimeError("No models loaded. Train first.")

    def run(self) -> None:
        self.executor.initialize()

        try:
            mode = "PAPER" if settings.paper_trading else "LIVE"
            print(f"Starting live trader in {mode} mode with policy {self.policy}...")
            events = self._refresh_events()
            last_refresh_utc = datetime.now(timezone.utc)
            last_eventless_eval_utc = datetime.min.replace(tzinfo=timezone.utc)

            already_traded_event_ids: set[str] = set()

            while True:
                now = datetime.now(timezone.utc)

                if (now - last_refresh_utc).total_seconds() >= max(10, settings.live_calendar_refresh_seconds):
                    events = self._refresh_events()
                    last_refresh_utc = now

                self.executor.apply_trailing_stop(settings.symbol)

                if not self.strategy.requires_event:
                    elapsed = (now - last_eventless_eval_utc).total_seconds()
                    if elapsed >= max(5, int(settings.eventless_eval_seconds)):
                        pseudo_event = pd.Series(
                            {
                                "event_id": f"eventless_{int(now.timestamp())}",
                                "date_utc": now.isoformat(),
                                "name": "Eventless strategy tick",
                                "currency": str(settings.symbol)[:3],
                                "importance": int(settings.event_min_importance),
                            }
                        )
                        decision = self._build_decision(pseudo_event)
                        if decision is not None:
                            eventless_id = str(pseudo_event["event_id"])
                            if settings.paper_trading:
                                self._record_paper_trade(eventless_id, pseudo_event, decision)
                                self._log_activity(action="paper_signal_eventless", event_id=eventless_id, detail=f"side={decision.side},confidence={decision.confidence:.4f}")
                                print(f"[PAPER] Eventless signal: {decision}")
                            else:
                                open_positions = self.executor.count_open_positions(settings.symbol)
                                if open_positions >= settings.max_open_positions:
                                    self._log_activity(action="skip_max_open_positions", event_id=eventless_id, detail=f"open_positions={open_positions}")
                                else:
                                    self.executor.send_market_order(settings.symbol, decision)
                                    self._log_activity(action="order_sent_eventless", event_id=eventless_id, detail=f"side={decision.side},confidence={decision.confidence:.4f}")
                                    print(f"Sending eventless order: {decision}")
                        else:
                            self._log_activity(action="eventless_no_decision", detail="strategy_returned_none")
                        last_eventless_eval_utc = now
                    time.sleep(max(1, settings.live_loop_sleep_seconds))
                    continue

                upcoming = events[events["date_utc"].map(pd.Timestamp) > pd.Timestamp(now)]
                if upcoming.empty:
                    print("No relevant upcoming events found. Waiting for next refresh...")
                    self._log_activity(action="no_upcoming_events", detail="calendar_empty_after_filter")
                    time.sleep(max(2, settings.live_calendar_refresh_seconds))
                    continue

                next_event = upcoming.iloc[0]
                event_id = str(next_event["event_id"])
                event_time = pd.to_datetime(next_event["date_utc"], utc=True).to_pydatetime()
                trigger_time = event_time - pd.Timedelta(seconds=settings.seconds_before_event)

                if event_id not in already_traded_event_ids and now >= trigger_time and now < event_time:
                    decision = self._build_decision(next_event)
                    if decision is not None:
                        if decision.confidence < self.policy["decision_threshold"]:
                            self._log_activity(action="skip_threshold", event_id=event_id, detail=f"confidence={decision.confidence:.4f}")
                            time.sleep(max(1, settings.live_loop_sleep_seconds))
                            continue

                        if abs(decision.proba_buy - 0.5) < self.policy["no_trade_band"]:
                            self._log_activity(action="skip_no_trade_band", event_id=event_id, detail=f"proba_buy={decision.proba_buy:.4f}")
                            time.sleep(max(1, settings.live_loop_sleep_seconds))
                            continue

                        if settings.paper_trading:
                            self._record_paper_trade(event_id, next_event, decision)
                            self._log_activity(action="paper_signal", event_id=event_id, detail=f"side={decision.side},confidence={decision.confidence:.4f}")
                            print(f"[PAPER] Signal for event {event_id}: {decision}")
                        else:
                            open_positions = self.executor.count_open_positions(settings.symbol)
                            if open_positions >= settings.max_open_positions:
                                print(
                                    f"Skipping event {event_id}: open positions {open_positions} "
                                    f">= MAX_OPEN_POSITIONS={settings.max_open_positions}"
                                )
                                self._log_activity(action="skip_max_open_positions", event_id=event_id, detail=f"open_positions={open_positions}")
                                already_traded_event_ids.add(event_id)
                                time.sleep(max(1, settings.live_loop_sleep_seconds))
                                continue

                            print(f"Sending order for event {event_id}: {decision}")
                            self.executor.send_market_order(settings.symbol, decision)
                            self._log_activity(action="order_sent", event_id=event_id, detail=f"side={decision.side},confidence={decision.confidence:.4f}")

                        already_traded_event_ids.add(event_id)
                    else:
                        self._log_activity(action="skip_no_decision", event_id=event_id, detail="strategy_returned_none")

                time.sleep(max(1, settings.live_loop_sleep_seconds))

        finally:
            self.executor.shutdown()

    def _refresh_events(self) -> pd.DataFrame:
        try:
            events = fetch_and_store_events(days_ahead=14)
            if events.empty:
                self._log_activity(action="calendar_refresh", detail="events=0")
            else:
                self._log_activity(action="calendar_refresh", detail=f"events={len(events)}")
            return events
        except Exception as ex:
            print(f"Calendar refresh failed: {ex}")
            self._log_activity(action="calendar_refresh_error", detail=str(ex)[:200])
            return pd.DataFrame()

    def _log_activity(self, action: str, event_id: str | None = None, detail: str = "") -> None:
        os.makedirs(settings.data_dir, exist_ok=True)
        path = settings.live_activity_csv
        row = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "PAPER" if settings.paper_trading else "LIVE",
            "strategy": settings.strategy,
            "symbol": settings.symbol,
            "action": action,
            "event_id": event_id or "",
            "detail": detail,
            "policy": json.dumps(self.policy, ensure_ascii=True),
        }
        df = pd.DataFrame([row])
        if os.path.exists(path):
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)

    def _build_decision(self, event_row: pd.Series) -> TradeDecision | None:
        ticks = self.executor.get_recent_ticks(settings.symbol, seconds=settings.lookback_seconds + 120)
        if ticks.empty:
            return None

        if self.strategy.requires_models:
            event_df = pd.DataFrame([event_row.to_dict()])
            bundle = build_event_dataset(event_df, ticks, lookback_seconds=settings.lookback_seconds)
            if bundle.X_tabular.empty:
                return None
        else:
            bundle = SimpleNamespace(X_tabular=pd.DataFrame(), X_seq=np.zeros((1, 1, 1), dtype=np.float32))
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
