from __future__ import annotations

import time
from datetime import datetime, timezone
import json
import os
from collections import deque
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
        self._recent_order_times: deque[datetime] = deque()
        self._last_order_side: str | None = None
        self._last_order_utc: datetime | None = None
        self._active_symbols: set[str] = {str(settings.symbol)}

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

                for sym in sorted(self._active_symbols):
                    try:
                        self.executor.apply_trailing_stop(sym)
                    except Exception:
                        continue

                if not self.strategy.requires_event:
                    elapsed = (now - last_eventless_eval_utc).total_seconds()
                    if elapsed >= max(5, int(settings.eventless_eval_seconds)):
                        impact_event = self._next_impact_event(events, now)
                        pseudo_event = pd.Series(
                            {
                                "event_id": f"eventless_{int(now.timestamp())}",
                                "date_utc": now.isoformat(),
                                "name": "Eventless strategy tick",
                                "currency": str(settings.symbol)[:3],
                                "importance": int(settings.event_min_importance),
                            }
                        )
                        context_event = impact_event if impact_event is not None else pseudo_event
                        target_symbol, routing_reason = self._resolve_symbol_for_event(context_event)
                        decision = self._build_decision(context_event, target_symbol)
                        if decision is not None:
                            eventless_id = str(context_event.get("event_id", pseudo_event["event_id"]))
                            allow_same_side_override = False
                            meta_getter = getattr(self.strategy, "get_last_signal_meta", None)
                            if callable(meta_getter):
                                try:
                                    meta = meta_getter() or {}
                                    allow_same_side_override = bool(meta.get("news_changed", False))
                                except Exception:
                                    allow_same_side_override = False
                            if settings.paper_trading:
                                self._record_paper_trade(eventless_id, context_event, decision, target_symbol)
                                self._log_activity(action="paper_signal_eventless", event_id=eventless_id, detail=f"side={decision.side},confidence={decision.confidence:.4f},symbol={target_symbol},route={routing_reason}", symbol=target_symbol)
                                print(f"[PAPER] Eventless signal: {decision}")
                            else:
                                open_positions = self.executor.count_open_positions(target_symbol)
                                if open_positions >= settings.max_open_positions:
                                    self._log_activity(action="skip_max_open_positions", event_id=eventless_id, detail=f"open_positions={open_positions},symbol={target_symbol},route={routing_reason}", symbol=target_symbol)
                                elif not self._can_send_order(now, str(decision.side), eventless_id, allow_same_side_override=allow_same_side_override):
                                    pass
                                else:
                                    self.executor.send_market_order(target_symbol, decision)
                                    self._on_order_sent(now, str(decision.side))
                                    self._active_symbols.add(target_symbol)
                                    self._log_activity(action="order_sent_eventless", event_id=eventless_id, detail=f"side={decision.side},confidence={decision.confidence:.4f},symbol={target_symbol},route={routing_reason}", symbol=target_symbol)
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
                    target_symbol, routing_reason = self._resolve_symbol_for_event(next_event)
                    decision = self._build_decision(next_event, target_symbol)
                    if decision is not None:
                        allow_same_side_override = False
                        meta_getter = getattr(self.strategy, "get_last_signal_meta", None)
                        if callable(meta_getter):
                            try:
                                meta = meta_getter() or {}
                                allow_same_side_override = bool(meta.get("news_changed", False))
                            except Exception:
                                allow_same_side_override = False
                        if decision.confidence < self.policy["decision_threshold"]:
                            self._log_activity(action="skip_threshold", event_id=event_id, detail=f"confidence={decision.confidence:.4f},symbol={target_symbol}", symbol=target_symbol)
                            time.sleep(max(1, settings.live_loop_sleep_seconds))
                            continue

                        if abs(decision.proba_buy - 0.5) < self.policy["no_trade_band"]:
                            self._log_activity(action="skip_no_trade_band", event_id=event_id, detail=f"proba_buy={decision.proba_buy:.4f},symbol={target_symbol}", symbol=target_symbol)
                            time.sleep(max(1, settings.live_loop_sleep_seconds))
                            continue

                        if settings.paper_trading:
                            self._record_paper_trade(event_id, next_event, decision, target_symbol)
                            self._log_activity(action="paper_signal", event_id=event_id, detail=f"side={decision.side},confidence={decision.confidence:.4f},symbol={target_symbol},route={routing_reason}", symbol=target_symbol)
                            print(f"[PAPER] Signal for event {event_id}: {decision}")
                        else:
                            open_positions = self.executor.count_open_positions(target_symbol)
                            if open_positions >= settings.max_open_positions:
                                print(
                                    f"Skipping event {event_id} ({target_symbol}): open positions {open_positions} "
                                    f">= MAX_OPEN_POSITIONS={settings.max_open_positions}"
                                )
                                self._log_activity(action="skip_max_open_positions", event_id=event_id, detail=f"open_positions={open_positions},symbol={target_symbol},route={routing_reason}", symbol=target_symbol)
                                already_traded_event_ids.add(event_id)
                                time.sleep(max(1, settings.live_loop_sleep_seconds))
                                continue
                            if not self._can_send_order(now, str(decision.side), event_id, allow_same_side_override=allow_same_side_override):
                                time.sleep(max(1, settings.live_loop_sleep_seconds))
                                continue

                            print(f"Sending order for event {event_id} [{target_symbol}] route={routing_reason}: {decision}")
                            self.executor.send_market_order(target_symbol, decision)
                            self._on_order_sent(now, str(decision.side))
                            self._active_symbols.add(target_symbol)
                            self._log_activity(action="order_sent", event_id=event_id, detail=f"side={decision.side},confidence={decision.confidence:.4f},symbol={target_symbol},route={routing_reason}", symbol=target_symbol)

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

    def _log_activity(self, action: str, event_id: str | None = None, detail: str = "", symbol: str | None = None) -> None:
        os.makedirs(settings.data_dir, exist_ok=True)
        path = settings.live_activity_csv
        agent_detail = self._agent_status_detail()
        full_detail = detail
        if agent_detail:
            full_detail = f"{detail} | {agent_detail}" if detail else agent_detail
        row = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "PAPER" if settings.paper_trading else "LIVE",
            "strategy": settings.strategy,
            "symbol": str(symbol or settings.symbol),
            "action": action,
            "event_id": event_id or "",
            "detail": full_detail,
            "policy": json.dumps(self.policy, ensure_ascii=True),
        }
        df = pd.DataFrame([row])
        if os.path.exists(path):
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)

    def _agent_status_detail(self) -> str:
        getter = getattr(self.strategy, "get_agent_status", None)
        if not callable(getter):
            return ""

        try:
            st = getter() or {}
        except Exception:
            return ""

        agent_name = str(st.get("agent_name", ""))
        strategy_class = str(st.get("strategy_class", ""))
        calls = int(st.get("calls", 0))
        decisions = int(st.get("decisions", 0))
        last_side = st.get("last_decision_side")
        return (
            f"agent={agent_name};strategy_class={strategy_class};"
            f"calls={calls};decisions={decisions};last_side={last_side}"
        )

    def _build_decision(self, event_row: pd.Series, symbol: str) -> TradeDecision | None:
        ticks = self.executor.get_recent_ticks(symbol, seconds=settings.lookback_seconds + 120)
        if ticks.empty:
            return None

        evt = event_row.copy() if isinstance(event_row, pd.Series) else pd.Series(dtype=object)
        evt["symbol"] = str(symbol)

        if self.strategy.requires_models:
            event_df = pd.DataFrame([evt.to_dict()])
            bundle = build_event_dataset(event_df, ticks, lookback_seconds=settings.lookback_seconds)
            if bundle.X_tabular.empty:
                return None
        else:
            bundle = SimpleNamespace(X_tabular=pd.DataFrame(), X_seq=np.zeros((1, 1, 1), dtype=np.float32))
        # delegate to selected strategy
        return self.strategy.decide(evt, ticks, bundle, self.tabular_models, self.lstm_model, self.feature_columns, self.policy, settings)

    @staticmethod
    def _parse_symbol_map(raw: str) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        txt = str(raw or "").strip()
        if not txt:
            return out
        for block in txt.split(";"):
            part = block.strip()
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            key = k.strip().upper()
            vals = [x.strip().upper() for x in str(v).split("|") if x.strip()]
            if key and vals:
                out[key] = vals
        return out

    def _pick_tradeable_symbol(self, candidates: list[str]) -> str | None:
        for sym in candidates:
            s = str(sym or "").strip().upper()
            if not s:
                continue
            try:
                if self.executor.is_symbol_tradeable(s):
                    return s
            except Exception:
                continue
        return None

    @staticmethod
    def _prioritize_symbols(candidates: list[str], base_symbol: str, prefer_non_default: bool) -> list[str]:
        dedup: list[str] = []
        seen: set[str] = set()
        for sym in candidates:
            s = str(sym or "").strip().upper()
            if not s or s in seen:
                continue
            seen.add(s)
            dedup.append(s)

        if not prefer_non_default:
            return dedup

        base = str(base_symbol or "").strip().upper()
        non_base = [s for s in dedup if s != base]
        base_tail = [s for s in dedup if s == base]
        return non_base + base_tail

    def _resolve_symbol_for_event(self, event_row: pd.Series) -> tuple[str, str]:
        base_symbol = str(settings.symbol).upper()
        ccy = str(event_row.get("currency", "")).strip().upper()
        name = str(event_row.get("name", "")).strip().lower()
        prefer_non_default = bool(getattr(settings, "impact_prefer_non_default", True))

        kw_map = self._parse_symbol_map(getattr(settings, "impact_keyword_symbol_map", ""))
        for kw, syms in kw_map.items():
            if kw.lower() in name:
                ranked = self._prioritize_symbols(syms, base_symbol=base_symbol, prefer_non_default=prefer_non_default)
                pick = self._pick_tradeable_symbol(ranked)
                if pick:
                    return pick, f"keyword:{kw.lower()}"

        ccy_map = self._parse_symbol_map(getattr(settings, "impact_symbol_map", ""))
        if ccy in ccy_map:
            ranked = self._prioritize_symbols(ccy_map[ccy], base_symbol=base_symbol, prefer_non_default=prefer_non_default)
            pick = self._pick_tradeable_symbol(ranked)
            if pick:
                return pick, f"currency:{ccy}"

        if bool(getattr(settings, "impact_symbol_fallback_to_default", True)):
            return base_symbol, "fallback:default"

        pick = self._pick_tradeable_symbol([base_symbol])
        return (pick or base_symbol), "fallback:default"

    def _next_impact_event(self, events: pd.DataFrame, now: datetime) -> pd.Series | None:
        if events is None or events.empty:
            return None
        if "date_utc" not in events.columns:
            return None
        aux = events.copy()
        aux["date_utc"] = pd.to_datetime(aux["date_utc"], utc=True, errors="coerce")
        aux = aux.dropna(subset=["date_utc"])
        aux = aux[aux["date_utc"] > pd.Timestamp(now)]
        if "importance" in aux.columns:
            imp = pd.to_numeric(aux["importance"], errors="coerce").fillna(0)
            aux = aux[imp >= float(settings.event_min_importance)]
        if aux.empty:
            return None
        aux = aux.sort_values(["date_utc"], ascending=True)
        return aux.iloc[0]

    def _prune_recent_orders(self, now: datetime) -> None:
        while self._recent_order_times and (now - self._recent_order_times[0]).total_seconds() > 3600:
            self._recent_order_times.popleft()

    def _can_send_order(self, now: datetime, side: str, event_id: str, allow_same_side_override: bool = False) -> bool:
        self._prune_recent_orders(now)

        min_gap = max(0, int(getattr(settings, "min_seconds_between_trades", 0)))
        if self._last_order_utc is not None and min_gap > 0:
            elapsed = float((now - self._last_order_utc).total_seconds())
            if elapsed < min_gap:
                self._log_activity(action="skip_trade_cooldown", event_id=event_id, detail=f"elapsed={elapsed:.1f},min_gap={min_gap}")
                return False

        max_hour = max(0, int(getattr(settings, "max_trades_per_hour", 0)))
        if max_hour > 0 and len(self._recent_order_times) >= max_hour:
            self._log_activity(action="skip_trade_rate_limit", event_id=event_id, detail=f"trades_last_hour={len(self._recent_order_times)},max={max_hour}")
            return False

        same_side_gap = max(0, int(getattr(settings, "same_side_cooldown_seconds", 0)))
        if (
            same_side_gap > 0
            and self._last_order_side == str(side).upper()
            and self._last_order_utc is not None
            and not allow_same_side_override
        ):
            elapsed_side = float((now - self._last_order_utc).total_seconds())
            if elapsed_side < same_side_gap:
                self._log_activity(action="skip_same_side_cooldown", event_id=event_id, detail=f"side={side},elapsed={elapsed_side:.1f},cooldown={same_side_gap}")
                return False

        return True

    def _on_order_sent(self, now: datetime, side: str) -> None:
        self._recent_order_times.append(now)
        self._last_order_utc = now
        self._last_order_side = str(side).upper()

    def _record_paper_trade(self, event_id: str, event_row: pd.Series, decision: TradeDecision, symbol: str) -> None:
        os.makedirs(settings.data_dir, exist_ok=True)
        path = os.path.join(settings.data_dir, "paper_trades.csv")
        row = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "event_id": event_id,
            "event_time_utc": str(event_row.get("date_utc", "")),
            "event_name": str(event_row.get("name", "")),
            "event_currency": str(event_row.get("currency", "")),
            "event_importance": event_row.get("importance", ""),
            "symbol": str(symbol),
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
