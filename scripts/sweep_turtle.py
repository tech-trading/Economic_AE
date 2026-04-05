from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.config import settings
from src.feature_engineering import build_event_dataset
from src.strategies import get_strategy


def evaluate_combo(
    bundle,
    ticks: pd.DataFrame,
    base_settings,
    policy: dict,
    combo: dict[str, float | int],
) -> dict[str, float | int] | None:
    cfg = SimpleNamespace(**vars(base_settings))
    cfg.turtle_lookback_seconds = int(combo["lookback"])
    cfg.turtle_breakout_buffer_pips = float(combo["buffer"])
    cfg.turtle_min_channel_pips = float(combo["min_channel"])
    cfg.turtle_confirm_ticks = int(combo["confirm_ticks"])
    cfg.turtle_atr_period_ticks = int(combo["atr_period"])
    cfg.turtle_min_atr_pips = float(combo["min_atr"])
    cfg.turtle_trend_ema_span = int(combo["trend_ema"])
    cfg.turtle_max_extension_atr = float(combo["max_ext"])
    cfg.turtle_signal_cooldown_seconds = int(combo["cooldown"])

    strat = get_strategy("turtle_atr", cfg, policy)

    times = ticks["time_utc"]
    results: list[tuple[bool, float]] = []

    for i in range(bundle.X_tabular.shape[0]):
        event_time = bundle.event_times.iloc[i]
        idx = int(times.searchsorted(event_time, side="right"))

        decision = strat.decide(
            pd.Series(dtype=object),
            ticks.iloc[:idx],
            bundle,
            None,
            None,
            None,
            policy,
            cfg,
        )
        if decision is None:
            continue

        ret_post = float(bundle.ret_post[i])
        is_correct = ((decision.side == "BUY") and (ret_post > 0.0)) or ((decision.side == "SELL") and (ret_post < 0.0))
        pnl_proxy = ret_post if decision.side == "BUY" else -ret_post
        results.append((is_correct, pnl_proxy))

    n = len(results)
    if n < 10:
        return None

    acc = sum(1 for r in results if r[0]) / n
    total = sum(r[1] for r in results)
    avg = total / n

    # Prefer robust sets: better mean pnl, better hit rate, avoid extremely sparse or hyperactive configs.
    signal_penalty = 0.00012 * abs(n - 120) / 120
    score = avg + 0.00008 * (acc - 0.5) - signal_penalty

    return {
        "score": float(score),
        "avg_pnl_proxy": float(avg),
        "accuracy": float(acc),
        "total_pnl_proxy": float(total),
        "signals": int(n),
        "lookback": int(combo["lookback"]),
        "buffer": float(combo["buffer"]),
        "min_channel": float(combo["min_channel"]),
        "confirm_ticks": int(combo["confirm_ticks"]),
        "atr_period": int(combo["atr_period"]),
        "min_atr": float(combo["min_atr"]),
        "trend_ema": int(combo["trend_ema"]),
        "max_ext": float(combo["max_ext"]),
        "cooldown": int(combo["cooldown"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep de hiperparametros para Turtle ATR breakout.")
    parser.add_argument("--events-csv", default=settings.events_csv)
    parser.add_argument("--market-csv", default=settings.market_csv)
    parser.add_argument("--output", default=str(Path(settings.model_dir) / "turtle_sweep_best.json"))
    parser.add_argument("--quick", action="store_true", help="Reduce combinaciones para ejecucion rapida.")
    parser.add_argument("--ultra-quick", action="store_true", help="Barrido minimo para resultado en pocos minutos.")
    args = parser.parse_args()

    events_path = Path(args.events_csv)
    ticks_path = Path(args.market_csv)
    if not events_path.exists() or not ticks_path.exists():
        raise FileNotFoundError(f"Input no encontrado. events={events_path} ticks={ticks_path}")

    events = pd.read_csv(events_path)
    ticks = pd.read_csv(ticks_path, parse_dates=["time_utc"])
    ticks["time_utc"] = pd.to_datetime(ticks["time_utc"], utc=True, errors="coerce")
    ticks = ticks.dropna(subset=["time_utc"]).sort_values("time_utc")

    bundle = build_event_dataset(events, ticks, lookback_seconds=settings.lookback_seconds)
    if bundle.X_tabular.empty:
        raise RuntimeError("No se pudo construir dataset para sweep (X_tabular vacio).")

    if args.ultra_quick:
        lookbacks = [1800, 3600]
        buffers = [0.02, 0.05, 0.10]
        min_channels = [0.005, 0.01, 0.02, 0.05]
        confirms = [2]
        atr_periods = [80, 120]
        min_atrs = [0.03, 0.05, 0.08]
        trend_emas = [140, 180]
        max_exts = [2.2, 2.8]
        cooldowns = [240, 300]
    elif args.quick:
        lookbacks = [1200, 1800, 3600, 7200]
        buffers = [0.02, 0.05, 0.10, 0.20]
        min_channels = [0.005, 0.01, 0.02, 0.05, 0.10]
        confirms = [1, 2]
        atr_periods = [80, 120]
        min_atrs = [0.03, 0.05, 0.08, 0.12]
        trend_emas = [120, 180]
        max_exts = [2.0, 2.5]
        cooldowns = [180, 240, 300]
    else:
        # Normal mode: broad enough for optimization, compact enough to finish in practical time.
        lookbacks = [1200, 1800, 3600, 5400, 7200]
        buffers = [0.02, 0.05, 0.10, 0.20]
        min_channels = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
        confirms = [1, 2]
        atr_periods = [80, 120, 180]
        min_atrs = [0.03, 0.05, 0.08, 0.12]
        trend_emas = [120, 180, 240]
        max_exts = [2.0, 2.5, 3.0]
        cooldowns = [180, 240, 300]

    combos: list[dict[str, float | int]] = []
    for lookback in lookbacks:
        for buffer in buffers:
            for min_channel in min_channels:
                for confirm_ticks in confirms:
                    for atr_period in atr_periods:
                        for min_atr in min_atrs:
                            for trend_ema in trend_emas:
                                for max_ext in max_exts:
                                    for cooldown in cooldowns:
                                        combos.append(
                                            {
                                                "lookback": lookback,
                                                "buffer": buffer,
                                                "min_channel": min_channel,
                                                "confirm_ticks": confirm_ticks,
                                                "atr_period": atr_period,
                                                "min_atr": min_atr,
                                                "trend_ema": trend_ema,
                                                "max_ext": max_ext,
                                                "cooldown": cooldown,
                                            }
                                        )

    policy = {
        "decision_threshold": settings.decision_threshold,
        "no_trade_band": settings.no_trade_band,
    }

    scored: list[dict[str, float | int]] = []
    for idx, combo in enumerate(combos, start=1):
        row = evaluate_combo(bundle, ticks, settings, policy, combo)
        if row is not None:
            scored.append(row)
        if idx % 500 == 0 or idx == len(combos):
            print(f"progress {idx}/{len(combos)} valid={len(scored)}")

    if not scored:
        raise RuntimeError("Sweep sin resultados validos (ninguna combinacion supero el minimo de senales).")

    scored = sorted(scored, key=lambda x: float(x["score"]), reverse=True)
    best = scored[0]

    output = {
        "best": best,
        "top20": scored[:20],
        "meta": {
            "events_csv": str(events_path),
            "market_csv": str(ticks_path),
            "samples": int(bundle.X_tabular.shape[0]),
            "tested_combinations": len(combos),
            "valid_combinations": len(scored),
            "quick": bool(args.quick),
            "ultra_quick": bool(args.ultra_quick),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    cmp_dir = Path(settings.data_dir) / "comparison"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scored).to_csv(cmp_dir / "turtle_sweep_results.csv", index=False)

    print(json.dumps(best, indent=2))
    print(f"Saved sweep result to {out_path}")


if __name__ == "__main__":
    main()
