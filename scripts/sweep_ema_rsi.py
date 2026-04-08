from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.config import settings
from src.feature_engineering import build_event_dataset
from src.strategies import get_strategy


def _compute_eval_returns(bundle, ticks: pd.DataFrame, horizon_seconds: int, min_post_ticks: int) -> np.ndarray:
    times = ticks["time_utc"]
    mids = ((ticks["bid"].astype(float) + ticks["ask"].astype(float)) / 2.0).to_numpy()
    out = np.full(bundle.X_tabular.shape[0], np.nan, dtype=np.float64)

    for i in range(bundle.X_tabular.shape[0]):
        et = pd.to_datetime(bundle.event_times.iloc[i], utc=True, errors="coerce")
        if pd.isna(et):
            continue
        start = int(times.searchsorted(et + pd.Timedelta(seconds=5), side="left"))
        end = int(times.searchsorted(et + pd.Timedelta(seconds=max(10, horizon_seconds)), side="right"))
        if end - start < max(2, int(min_post_ticks)):
            continue
        first_mid = float(mids[start])
        last_mid = float(mids[end - 1])
        if first_mid == 0:
            continue
        out[i] = (last_mid - first_mid) / first_mid

    return out


def evaluate_combo(bundle, ticks, eval_returns, base_settings, policy, combo):
    cfg = SimpleNamespace(**vars(base_settings))
    cfg.agent_manage_all_strategies = False
    cfg.ema_fast_span = int(combo["fast"])
    cfg.ema_slow_span = int(combo["slow"])
    cfg.ema_rsi_period = int(combo["rsi_period"])
    cfg.ema_rsi_buy_level = float(combo["rsi_buy"])
    cfg.ema_rsi_sell_level = float(combo["rsi_sell"])
    cfg.ema_min_separation_pips = float(combo["min_sep"])
    cfg.ema_momentum_lookback_ticks = int(combo["mom_lb"])
    cfg.ema_min_momentum_pips = float(combo["min_mom"])
    cfg.ema_vol_period = int(combo["vol_period"])
    cfg.ema_min_vol_pips = float(combo["min_vol"])
    cfg.ema_signal_cooldown_seconds = int(combo["cooldown"])

    strat = get_strategy("ema_rsi_trend", cfg, policy)
    times = ticks["time_utc"]

    n = 0
    correct = 0
    pnl_sum = 0.0

    for i in range(bundle.X_tabular.shape[0]):
        ev_time = bundle.event_times.iloc[i]
        idx = int(times.searchsorted(ev_time, side="right"))
        dec = strat.decide(pd.Series(dtype=object), ticks.iloc[:idx], None, None, None, None, policy, cfg)
        if dec is None:
            continue

        ret = eval_returns[i]
        if not np.isfinite(ret):
            continue

        n += 1
        if ((dec.side == "BUY") and (ret > 0.0)) or ((dec.side == "SELL") and (ret < 0.0)):
            correct += 1
        pnl_sum += ret if dec.side == "BUY" else -ret

    if n < 15:
        return None

    acc = correct / n
    avg_pnl = pnl_sum / n
    target_signals = 90.0
    score = acc + (200.0 * avg_pnl) - (0.20 * abs(n - target_signals) / target_signals)

    return {
        "score": float(score),
        "accuracy": float(acc),
        "signals": int(n),
        "avg_pnl_proxy": float(avg_pnl),
        "total_pnl_proxy": float(pnl_sum),
        **{k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in combo.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep EMA RSI trend orientado a precision.")
    parser.add_argument("--events-csv", default="data/events.csv")
    parser.add_argument("--market-csv", default="data/market_ticks.csv")
    parser.add_argument("--horizon-minutes", type=float, default=5.0)
    parser.add_argument("--horizon-seconds", type=int, default=None)
    parser.add_argument("--min-post-ticks", type=int, default=5)
    parser.add_argument("--output", default=str(Path(settings.model_dir) / "ema_rsi_sweep_best.json"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    # Backward compatible: if horizon-seconds is provided it takes precedence.
    horizon_seconds = int(args.horizon_seconds) if args.horizon_seconds is not None else int(float(args.horizon_minutes) * 60)

    events = pd.read_csv(args.events_csv)
    ticks = pd.read_csv(args.market_csv, parse_dates=["time_utc"])
    ticks["time_utc"] = pd.to_datetime(ticks["time_utc"], utc=True, errors="coerce")
    ticks = ticks.dropna(subset=["time_utc"]).sort_values("time_utc")

    bundle = build_event_dataset(events, ticks, lookback_seconds=settings.lookback_seconds)
    if bundle.X_tabular.empty:
        raise RuntimeError("No dataset samples for EMA sweep")

    eval_returns = _compute_eval_returns(bundle, ticks, horizon_seconds=horizon_seconds, min_post_ticks=int(args.min_post_ticks))

    if args.quick:
        fasts = [13, 21]
        slows = [55, 89]
        rsi_periods = [14]
        rsi_buys = [56, 58]
        rsi_sells = [44, 42]
        min_seps = [0.05, 0.10, 0.20]
        mom_lbs = [10, 20]
        min_moms = [0.05, 0.10, 0.20]
        vol_periods = [30, 40]
        min_vols = [0.01, 0.03, 0.05]
        cooldowns = [120, 180, 240]
    else:
        fasts = [8, 13, 21, 34]
        slows = [34, 55, 89, 144]
        rsi_periods = [10, 14]
        rsi_buys = [54, 56, 58, 60]
        rsi_sells = [46, 44, 42, 40]
        min_seps = [0.03, 0.05, 0.10, 0.20, 0.40]
        mom_lbs = [8, 10, 20, 30]
        min_moms = [0.03, 0.05, 0.10, 0.20, 0.40]
        vol_periods = [20, 30, 40, 60]
        min_vols = [0.005, 0.01, 0.03, 0.05, 0.08]
        cooldowns = [90, 120, 180, 240, 300]

    combos = []
    for fast in fasts:
        for slow in slows:
            if slow <= fast:
                continue
            for rsi_period in rsi_periods:
                for rsi_buy, rsi_sell in zip(rsi_buys, rsi_sells):
                    for min_sep in min_seps:
                        for mom_lb in mom_lbs:
                            for min_mom in min_moms:
                                for vol_period in vol_periods:
                                    for min_vol in min_vols:
                                        for cooldown in cooldowns:
                                            combos.append(
                                                {
                                                    "fast": fast,
                                                    "slow": slow,
                                                    "rsi_period": rsi_period,
                                                    "rsi_buy": rsi_buy,
                                                    "rsi_sell": rsi_sell,
                                                    "min_sep": min_sep,
                                                    "mom_lb": mom_lb,
                                                    "min_mom": min_mom,
                                                    "vol_period": vol_period,
                                                    "min_vol": min_vol,
                                                    "cooldown": cooldown,
                                                }
                                            )

    policy = {"decision_threshold": settings.decision_threshold, "no_trade_band": settings.no_trade_band}

    scored = []
    for idx, combo in enumerate(combos, start=1):
        row = evaluate_combo(bundle, ticks, eval_returns, settings, policy, combo)
        if row is not None:
            scored.append(row)
        if idx % 300 == 0 or idx == len(combos):
            print(f"progress {idx}/{len(combos)} valid={len(scored)}")

    if not scored:
        raise RuntimeError("EMA sweep without valid combinations")

    scored = sorted(scored, key=lambda x: float(x["score"]), reverse=True)
    best = scored[0]

    out = {
        "best": best,
        "top20": scored[:20],
        "meta": {
            "events_csv": str(args.events_csv),
            "market_csv": str(args.market_csv),
            "horizon_seconds": int(horizon_seconds),
            "horizon_minutes": float(horizon_seconds) / 60.0,
            "min_post_ticks": int(args.min_post_ticks),
            "samples": int(bundle.X_tabular.shape[0]),
            "tested_combinations": len(combos),
            "valid_combinations": len(scored),
            "quick": bool(args.quick),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    cmp_dir = Path(settings.data_dir) / "comparison"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scored).to_csv(cmp_dir / "ema_rsi_sweep_results.csv", index=False)

    print(json.dumps(best, indent=2))
    print(f"Saved sweep result to {out_path}")


if __name__ == "__main__":
    main()
