from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from src.config import settings


@dataclass
class TradeDecision:
    side: str
    confidence: float
    proba_buy: float = 0.5
    sl_pips_override: float | None = None
    tp_pips_override: float | None = None
    trailing_stop_pips_override: float | None = None
    trailing_activation_pips_override: float | None = None


class MT5Executor:
    def __init__(self) -> None:
        self._initialized = False

    def initialize(self) -> None:
        if settings.mt5_login > 0 and settings.mt5_password and settings.mt5_server:
            ok = mt5.initialize(
                login=settings.mt5_login,
                password=settings.mt5_password,
                server=settings.mt5_server,
            )
        else:
            ok = mt5.initialize()

        if not ok:
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

        self._initialized = True

    def shutdown(self) -> None:
        if self._initialized:
            mt5.shutdown()
            self._initialized = False

    def get_recent_ticks(self, symbol: str, seconds: int) -> pd.DataFrame:
        utc_to = datetime.now(timezone.utc)
        utc_from = utc_to - pd.Timedelta(seconds=seconds)

        ticks = mt5.copy_ticks_range(symbol, utc_from, utc_to, mt5.COPY_TICKS_INFO)
        if ticks is None or len(ticks) == 0:
            ticks = mt5.copy_ticks_range(symbol, utc_from, utc_to, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            return pd.DataFrame(columns=["time_utc", "bid", "ask"])

        df = pd.DataFrame(ticks)
        df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
        out = df[["time_utc", "bid", "ask"]].copy()
        out["bid"] = pd.to_numeric(out["bid"], errors="coerce")
        out["ask"] = pd.to_numeric(out["ask"], errors="coerce")
        out = out.dropna(subset=["time_utc", "bid", "ask"])
        out = out[(out["bid"] > 0.0) & (out["ask"] > 0.0) & (out["ask"] >= out["bid"])]
        if out.empty:
            return pd.DataFrame(columns=["time_utc", "bid", "ask"])
        return out.sort_values("time_utc").reset_index(drop=True)

    def is_symbol_tradeable(self, symbol: str) -> bool:
        sym = str(symbol or "").strip()
        if not sym:
            return False
        info = mt5.symbol_info(sym)
        if info is None:
            return False
        if not info.visible:
            mt5.symbol_select(sym, True)
            info = mt5.symbol_info(sym)
            if info is None:
                return False
        return True

    def send_market_order(self, symbol: str, decision: TradeDecision) -> dict:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol not found in MT5: {symbol}")

        if not info.visible:
            mt5.symbol_select(symbol, True)

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError("Cannot get symbol tick")

        spread_points = int((tick.ask - tick.bid) / info.point)
        if spread_points > settings.max_spread_points:
            raise RuntimeError(f"Spread too high: {spread_points} points")

        spread_pips = float((tick.ask - tick.bid) / max(1e-12, info.point * 10.0))

        is_buy = decision.side.upper() == "BUY"
        price = tick.ask if is_buy else tick.bid

        sl_pips_cfg = float(settings.stop_loss_pips)
        tp_pips_cfg = float(settings.take_profit_pips)
        sl_override = float(getattr(decision, "sl_pips_override", 0.0) or 0.0)
        if sl_override > 0.0:
            sl_pips = sl_override
            sl_pips = float(np.clip(sl_pips, float(getattr(settings, "stop_loss_min_pips", 1.0)), float(getattr(settings, "stop_loss_max_pips", 20.0))))
        else:
            sl_mode = str(getattr(settings, "stop_loss_mode", "adaptive")).strip().lower()
            if sl_mode == "adaptive":
                sl_pips = self._adaptive_stop_loss_pips(
                    symbol=symbol,
                    decision=decision,
                    spread_pips=spread_pips,
                    info=info,
                )
            else:
                sl_pips = sl_pips_cfg
        sl_pips = max(sl_pips, spread_pips * float(getattr(settings, "min_sl_spread_multiplier", 3.0)))
        tp_override = float(getattr(decision, "tp_pips_override", 0.0) or 0.0)

        tp_mode = str(getattr(settings, "take_profit_mode", "adaptive")).strip().lower()
        if tp_override > 0.0:
            tp_pips = tp_override
            ratio_floor = float(max(0.0, getattr(settings, "tp_sl_ratio_floor", 0.0)))
            if ratio_floor > 0.0:
                tp_pips = max(tp_pips, sl_pips * ratio_floor)
            tp_pips = float(np.clip(tp_pips, float(getattr(settings, "take_profit_min_pips", 2.0)), float(getattr(settings, "take_profit_max_pips", 20.0))))
        elif tp_mode == "adaptive":
            tp_pips = self._adaptive_take_profit_pips(
                symbol=symbol,
                decision=decision,
                sl_pips=sl_pips,
                spread_pips=spread_pips,
                info=info,
            )
            ratio_floor = float(max(0.5, getattr(settings, "tp_sl_ratio_floor", 0.9)))
            tp_pips = max(tp_pips, sl_pips * ratio_floor)
        else:
            tp_pips = max(tp_pips_cfg, sl_pips * float(getattr(settings, "min_tp_sl_ratio_enforced", 1.6)))

        sl_distance = sl_pips * info.point * 10
        tp_distance = tp_pips * info.point * 10

        sl = price - sl_distance if is_buy else price + sl_distance
        tp = price + tp_distance if is_buy else price - tp_distance

        trail_pips = float(getattr(decision, "trailing_stop_pips_override", 0.0) or 0.0)
        if trail_pips <= 0.0:
            trail_pips = float(settings.trailing_stop_pips)
        trail_act_pips = float(getattr(decision, "trailing_activation_pips_override", 0.0) or 0.0)
        if trail_act_pips <= 0.0:
            trail_act_pips = float(getattr(settings, "trailing_activation_pips", 0.0))
        if trail_act_pips <= 0.0:
            trail_act_pips = trail_pips

        comment = self._build_order_comment(
            confidence=float(decision.confidence),
            sl_pips=float(sl_pips),
            tp_pips=float(tp_pips),
            trail_pips=float(trail_pips),
            trail_act_pips=float(trail_act_pips),
        )

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": settings.order_volume,
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 920260311,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"order_send returned None: {mt5.last_error()}")

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Order failed: retcode={result.retcode}, comment={result.comment}")

        res = result._asdict()
        try:
            # Apply trailing stop immediately after opening the order so
            # the new position gets the trailing SL without waiting for
            # the next loop iteration.
            self.apply_trailing_stop(symbol)
        except Exception as ex:
            print(f"apply_trailing_stop after open failed: {ex}")

        return res

    def apply_trailing_stop(self, symbol: str) -> None:
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return

        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            return

        pip_size = self._pip_size_from_info(info)
        default_trail_pips = float(settings.trailing_stop_pips)
        activation_pips_cfg = float(getattr(settings, "trailing_activation_pips", 0.0))
        default_activation_pips = activation_pips_cfg if activation_pips_cfg > 0 else default_trail_pips
        be_offset = float(getattr(settings, "trailing_break_even_offset_pips", 0.2)) * pip_size

        for pos in positions:
            is_buy = pos.type == mt5.POSITION_TYPE_BUY
            current_price = tick.bid if is_buy else tick.ask
            trail_pips, activation_pips = self._extract_trailing_profile(
                str(getattr(pos, "comment", "")),
                default_trail_pips,
                default_activation_pips,
            )
            trail_distance = trail_pips * pip_size

            entry_price = float(getattr(pos, "price_open", current_price))
            if is_buy:
                profit_pips = (current_price - entry_price) / max(1e-12, pip_size)
            else:
                profit_pips = (entry_price - current_price) / max(1e-12, pip_size)
            if profit_pips < activation_pips:
                continue

            target_sl = current_price - trail_distance if is_buy else current_price + trail_distance

            if is_buy:
                target_sl = max(target_sl, entry_price + be_offset)
            else:
                target_sl = min(target_sl, entry_price - be_offset)

            current_sl = float(getattr(pos, "sl", 0.0))
            should_update = (is_buy and target_sl > current_sl) or ((not is_buy) and (current_sl == 0 or target_sl < current_sl))
            if not should_update:
                continue

            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": pos.ticket,
                "sl": target_sl,
                "tp": pos.tp,
            }
            mt5.order_send(modify_request)

    @staticmethod
    def _extract_trailing_profile(comment: str, default_trail_pips: float, default_activation_pips: float) -> tuple[float, float]:
        trail = float(default_trail_pips)
        activation = float(default_activation_pips)
        txt = str(comment or "")
        if not txt:
            return trail, activation
        tokens = [tk for tk in re.split(r"[|_]", txt) if tk]
        for token in tokens:
            tk = token.strip().lower()
            if tk.startswith("ts"):
                try:
                    v = float(tk.replace("ts", "", 1))
                    if v > 0:
                        trail = v
                except Exception:
                    pass
            elif tk.startswith("ta"):
                try:
                    v = float(tk.replace("ta", "", 1))
                    if v > 0:
                        activation = v
                except Exception:
                    pass
            elif tk.startswith("t"):
                try:
                    v = float(tk.replace("t", "", 1)) / 10.0
                    if v > 0:
                        trail = v
                except Exception:
                    pass
            elif tk.startswith("a"):
                try:
                    v = float(tk.replace("a", "", 1)) / 10.0
                    if v > 0:
                        activation = v
                except Exception:
                    pass
        if activation <= 0:
            activation = trail
        return trail, activation

    @staticmethod
    def _build_order_comment(confidence: float, sl_pips: float, tp_pips: float, trail_pips: float, trail_act_pips: float) -> str:
        # MT5 comments are broker-limited (commonly 31 chars), so keep compact ASCII.
        c = int(np.clip(round(confidence * 100.0), 0, 99))
        s = int(np.clip(round(sl_pips * 10.0), 0, 999))
        p = int(np.clip(round(tp_pips * 10.0), 0, 999))
        t = int(np.clip(round(trail_pips * 10.0), 0, 999))
        a = int(np.clip(round(trail_act_pips * 10.0), 0, 999))
        raw = f"ea_c{c}_s{s}_p{p}_t{t}_a{a}"
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", raw)
        return safe[:31]

    @staticmethod
    def _pip_size_from_info(info) -> float:
        digits = int(getattr(info, "digits", 5))
        point = float(getattr(info, "point", 0.00001))
        return point * 10.0 if digits in {3, 5} else point

    def _adaptive_take_profit_pips(self, symbol: str, decision: TradeDecision, sl_pips: float, spread_pips: float, info) -> float:
        min_pips = float(max(0.5, getattr(settings, "take_profit_min_pips", 4.0)))
        max_pips = float(max(min_pips, getattr(settings, "take_profit_max_pips", 12.0)))
        base_pips = float(np.clip(getattr(settings, "take_profit_pips", 8.0), min_pips, max_pips))

        confidence = float(np.clip(getattr(decision, "confidence", 0.5), 0.5, 0.99))
        conf_scale = float((confidence - 0.5) / 0.49)
        conf_bonus = float(np.clip(conf_scale, 0.0, 1.0)) * float(getattr(settings, "take_profit_confidence_bonus_pips", 2.0))

        vol_seconds = int(max(60, getattr(settings, "take_profit_volatility_seconds", 300)))
        vol_mult = float(max(0.2, getattr(settings, "take_profit_volatility_multiplier", 1.2)))

        vol_component = 0.0
        try:
            ticks = self.get_recent_ticks(symbol, seconds=vol_seconds)
            if ticks is not None and not ticks.empty and {"bid", "ask"}.issubset(set(ticks.columns)):
                mid = (ticks["bid"].astype(float) + ticks["ask"].astype(float)) / 2.0
                if len(mid) >= 20:
                    pip_size = self._pip_size_from_info(info)
                    range_pips = float((mid.max() - mid.min()) / max(1e-12, pip_size))
                    vol_component = float(np.clip(0.35 * range_pips * vol_mult, min_pips, max_pips))
        except Exception:
            vol_component = 0.0

        target = max(base_pips, vol_component)
        # Penalize high spread environments so TP can be reached before mean reversion.
        target -= max(0.0, spread_pips - 1.0) * 0.4
        target += conf_bonus

        return float(np.clip(target, min_pips, max_pips))

    def _adaptive_stop_loss_pips(self, symbol: str, decision: TradeDecision, spread_pips: float, info) -> float:
        min_pips = float(max(0.6, getattr(settings, "stop_loss_min_pips", 2.2)))
        max_pips = float(max(min_pips, getattr(settings, "stop_loss_max_pips", 7.5)))
        base_pips = float(np.clip(getattr(settings, "stop_loss_pips", 4.0), min_pips, max_pips))

        confidence = float(np.clip(getattr(decision, "confidence", 0.5), 0.5, 0.99))
        conf_scale = float((confidence - 0.5) / 0.49)
        conf_bonus = float(np.clip(conf_scale, 0.0, 1.0)) * float(getattr(settings, "stop_loss_confidence_bonus_pips", 0.7))

        vol_seconds = int(max(60, getattr(settings, "stop_loss_volatility_seconds", 300)))
        vol_mult = float(max(0.2, getattr(settings, "stop_loss_volatility_multiplier", 1.0)))

        vol_component = 0.0
        try:
            ticks = self.get_recent_ticks(symbol, seconds=vol_seconds)
            if ticks is not None and not ticks.empty and {"bid", "ask"}.issubset(set(ticks.columns)):
                mid = (ticks["bid"].astype(float) + ticks["ask"].astype(float)) / 2.0
                if len(mid) >= 20:
                    pip_size = self._pip_size_from_info(info)
                    range_pips = float((mid.max() - mid.min()) / max(1e-12, pip_size))
                    vol_component = float(np.clip(0.30 * range_pips * vol_mult, min_pips, max_pips))
        except Exception:
            vol_component = 0.0

        target = max(base_pips, vol_component)
        target += max(0.0, spread_pips - 1.0) * 0.25
        target += conf_bonus * 0.5
        return float(np.clip(target, min_pips, max_pips))

    def count_open_positions(self, symbol: str) -> int:
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return 0
        return int(len(positions))

    def get_open_positions(self, symbol: str) -> pd.DataFrame:
        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            return pd.DataFrame()

        df = pd.DataFrame([p._asdict() for p in positions])
        if "time" in df.columns:
            df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
        if "type" in df.columns:
            df["side"] = np.where(df["type"] == mt5.POSITION_TYPE_BUY, "BUY", "SELL")
        return df

    def get_recent_deals(self, symbol: str, days: int = 7) -> pd.DataFrame:
        utc_to = datetime.now(timezone.utc)
        utc_from = utc_to - pd.Timedelta(days=max(1, int(days)))

        deals = mt5.history_deals_get(utc_from, utc_to)
        if deals is None or len(deals) == 0:
            return pd.DataFrame()

        df = pd.DataFrame([d._asdict() for d in deals])
        if "symbol" in df.columns:
            df = df[df["symbol"].astype(str) == str(symbol)].copy()
        if df.empty:
            return df

        if "time" in df.columns:
            df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
        if "type" in df.columns:
            df["side"] = np.where(df["type"] == mt5.DEAL_TYPE_BUY, "BUY", "SELL")
        if "entry" in df.columns:
            entry_map = {
                int(mt5.DEAL_ENTRY_IN): "OPEN",
                int(mt5.DEAL_ENTRY_OUT): "CLOSE",
                int(mt5.DEAL_ENTRY_INOUT): "REVERSE",
                int(mt5.DEAL_ENTRY_OUT_BY): "CLOSE_BY",
            }
            df["entry_label"] = df["entry"].map(entry_map).fillna("OTHER")

        return df.sort_values("time_utc", ascending=False) if "time_utc" in df.columns else df
