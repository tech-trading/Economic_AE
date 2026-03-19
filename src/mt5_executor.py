from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from src.config import settings


@dataclass
class TradeDecision:
    side: str
    confidence: float
    proba_buy: float = 0.5


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

        ticks = mt5.copy_ticks_range(symbol, utc_from, utc_to, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            return pd.DataFrame(columns=["time_utc", "bid", "ask"])

        df = pd.DataFrame(ticks)
        df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df[["time_utc", "bid", "ask"]].copy()

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

        is_buy = decision.side.upper() == "BUY"
        price = tick.ask if is_buy else tick.bid

        sl_distance = settings.stop_loss_pips * info.point * 10
        tp_distance = settings.take_profit_pips * info.point * 10

        sl = price - sl_distance if is_buy else price + sl_distance
        tp = price + tp_distance if is_buy else price - tp_distance

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
            "comment": f"econ_ai_{decision.confidence:.2f}",
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

        trail_distance = settings.trailing_stop_pips * info.point * 10

        for pos in positions:
            is_buy = pos.type == mt5.POSITION_TYPE_BUY
            current_price = tick.bid if is_buy else tick.ask

            target_sl = current_price - trail_distance if is_buy else current_price + trail_distance

            should_update = (is_buy and target_sl > pos.sl) or ((not is_buy) and (pos.sl == 0 or target_sl < pos.sl))
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
