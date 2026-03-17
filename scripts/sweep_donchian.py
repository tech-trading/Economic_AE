from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.config import settings
from src.feature_engineering import build_event_dataset
from src.strategies import get_strategy


def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def evaluate_combo(bundle, ticks: pd.DataFrame, base_settings, policy: dict, combo: dict[str, float | int | str]) -> dict[str, float | int] | None:
    cfg = SimpleNamespace(**vars(base_settings))
    cfg.donchian_lookback_seconds = int(combo["lookback"])
    cfg.donchian_trigger_quantile = float(combo["quantile"])
    cfg.donchian_min_channel_pips = float(combo["min_channel"])
    cfg.donchian_breakout_buffer_pips = float(combo["buffer"])
    cfg.donchian_confirm_ticks = int(combo["confirm_ticks"])
    cfg.donchian_session_filter = bool(combo["session_filter"])
    cfg.donchian_sessions = str(combo["sessions"])

    strategy_name = "donchian_nylondon" if cfg.donchian_session_filter else "donchian"
    strat = get_strategy(strategy_name, cfg, policy)

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
    if n < 40:
        return None

    acc = sum(1 for r in results if r[0]) / n
    total = sum(r[1] for r in results)
    avg = total / n
    # Score robusto: prioriza retorno medio y estabilidad de frecuencia de señales.
    score = avg + 0.00005 * (acc - 0.5) - 0.0002 * abs(n - 140) / 140

    out = {
        "score": float(score),
        "avg_pnl_proxy": float(avg),
        "accuracy": float(acc),
        "total_pnl_proxy": float(total),
        "signals": int(n),
        "lookback": int(combo["lookback"]),
        "quantile": float(combo["quantile"]),
        "min_channel": float(combo["min_channel"]),
        "buffer": float(combo["buffer"]),
        "confirm_ticks": int(combo["confirm_ticks"]),
        "session_filter": bool(combo["session_filter"]),
        "sessions": str(combo["sessions"]),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep de hiperparámetros para estrategia Donchian.")
    parser.add_argument("--events-csv", default=settings.events_csv)
    parser.add_argument("--market-csv", default=settings.market_csv)
    parser.add_argument("--output", default=str(Path(settings.model_dir) / "donchian_sweep_best.json"))
    parser.add_argument("--session-filter", default="both", choices=["both", "on", "off"])
    parser.add_argument("--quick", action="store_true", help="Reduce combinaciones para ejecución rápida.")
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
        raise RuntimeError("No se pudo construir dataset para sweep (X_tabular vacío).")

    session_values: list[bool]
    if args.session_filter == "on":
        session_values = [True]
    elif args.session_filter == "off":
        session_values = [False]
    else:
        session_values = [False, True]

    if args.quick:
        lookbacks = [300, 600, 900]
        quantiles = [0.75, 0.80, 0.85]
        min_channels = [0.1, 0.5, 1.0]
        buffers = [0.0, 0.1]
        confirms = [1]
    else:
        lookbacks = [240, 300, 600, 900, 1200]
        quantiles = [0.70, 0.75, 0.80, 0.85]
        min_channels = [0.1, 0.3, 0.5, 1.0, 1.5]
        buffers = [0.0, 0.1, 0.2, 0.3]
        confirms = [1, 2]

    combos: list[dict[str, float | int | str]] = []
    for lookback in lookbacks:
        for quantile in quantiles:
            for min_channel in min_channels:
                for buffer in buffers:
                    for confirm_ticks in confirms:
                        for session_filter in session_values:
                            combos.append(
                                {
                                    "lookback": lookback,
                                    "quantile": quantile,
                                    "min_channel": min_channel,
                                    "buffer": buffer,
                                    "confirm_ticks": confirm_ticks,
                                    "session_filter": session_filter,
                                    "sessions": "london,ny",
                                }
                            )

    policy = {"decision_threshold": settings.decision_threshold}
    scored: list[dict[str, float | int]] = []
    for combo in combos:
        row = evaluate_combo(bundle, ticks, settings, policy, combo)
        if row is not None:
            scored.append(row)

    if not scored:
        raise RuntimeError("Sweep sin resultados válidos (ninguna combinación superó el mínimo de señales).")

    scored = sorted(scored, key=lambda x: float(x["score"]), reverse=True)
    best = scored[0]
    output = {
        "best": best,
        "top10": scored[:10],
        "meta": {
            "events_csv": str(events_path),
            "market_csv": str(ticks_path),
            "samples": int(bundle.X_tabular.shape[0]),
            "tested_combinations": len(combos),
            "valid_combinations": len(scored),
            "session_filter_mode": args.session_filter,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(best, indent=2))
    print(f"Saved sweep result to {out_path}")


if __name__ == "__main__":
    main()
