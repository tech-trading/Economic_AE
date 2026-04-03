from __future__ import annotations

from typing import Tuple

from .config import settings


def compute_sl_tp_from_atr(entry_price: float, atr: float, long: bool = True) -> Tuple[float, float]:
    """Devuelve (stop_price, take_profit_price) usando los parámetros en `settings`.

    - `atr` debe estar en las mismas unidades de precio que `entry_price`.
    - Si long=True, SL < entry_price < TP; si long=False, TP < entry_price < SL.
    """
    sl_points = atr * settings.sl_atr_multiplier
    tp_points = sl_points * settings.tp_sl_ratio

    if long:
        stop_price = entry_price - sl_points
        take_profit_price = entry_price + tp_points
    else:
        stop_price = entry_price + sl_points
        take_profit_price = entry_price - tp_points

    return stop_price, take_profit_price


def compute_trailing_params(atr: float) -> Tuple[float, float]:
    """Devuelve (trail_distance, trail_step) en unidades de precio.

    - `trail_distance` es cuánto mantener el trailing stop alejado del precio antes y después
      de activarlo (multiplo de ATR).
    - `trail_step` es cuánto mover el stop cada vez (multiplo de ATR).
    """
    trail_distance = atr * settings.trail_atr_multiplier
    trail_step = atr * settings.trail_step_atr
    return trail_distance, trail_step


# Ejemplo de uso (simple) para backtesters:
#
# from src.risk import compute_sl_tp_from_atr, compute_trailing_params
# entry = 1.1200
# atr = 0.0008  # ejemplo para EURUSD
# stop, tp = compute_sl_tp_from_atr(entry, atr, long=True)
# trail_distance, trail_step = compute_trailing_params(atr)
#
# Esto produce SL/TP en precio y parámetros para el trailing. Ajusta `atr` según tu
# cálculo de volatilidad (ATR en pips/price units) y el `entry` real del trade.
