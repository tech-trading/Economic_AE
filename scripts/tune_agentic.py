from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from dotenv import dotenv_values

from src.config import settings
from src.policy import load_policy
from src.strategies import get_strategy


def _load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    vals = dotenv_values(str(path))
    return {str(k): str(v) for k, v in vals.items() if k is not None and v is not None}


def _save_env(path: Path, data: dict[str, str]) -> None:
    lines = [f"{k}={v}" for k, v in data.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pip_size(symbol: str) -> float:
    sym = str(symbol or "").upper()
    return 0.01 if "JPY" in sym else 0.0001


def _rows_for_seconds(ticks: pd.DataFrame, seconds: int, default_rows: int = 20) -> int:
    if ticks.empty or "time_utc" not in ticks.columns or len(ticks) < 3:
        return default_rows
    dt = ticks["time_utc"].diff().dt.total_seconds().dropna()
    med = float(dt.median()) if not dt.empty else 1.0
    med = max(0.2, med)
    return max(1, int(round(seconds / med)))


def evaluate_combo(
    *,
    ticks: pd.DataFrame,
    base_settings,
    policy: dict,
    combo: dict[str, float | int],
    eval_horizon_seconds: int,
    min_signals: int,
) -> dict[str, float | int] | None:
    cfg = SimpleNamespace(**vars(base_settings))

    cfg.strategy = "agentic_hybrid"
    cfg.ema_fast_span = int(combo["ema_fast"])
    cfg.ema_slow_span = int(combo["ema_slow"])
    cfg.ema_min_separation_pips = float(combo["ema_sep"])
    cfg.ema_min_momentum_pips = float(combo["ema_mom"])
    cfg.ema_min_vol_pips = float(combo["ema_vol"])

    cfg.donchian_trigger_quantile = float(combo["don_q"])
    cfg.donchian_min_channel_pips = float(combo["don_ch"])
    cfg.donchian_breakout_buffer_pips = float(combo["don_buf"])

    cfg.agentic_learning_rate = float(combo["lr"])
    cfg.agentic_explore_prob = float(combo["exp"])
    cfg.agentic_min_confidence = float(combo["min_conf"])
    cfg.agentic_reward_horizon_seconds = int(combo["reward_h"])
    cfg.agentic_reward_target_pips = float(combo["reward_t"])
    cfg.eventless_eval_seconds = int(combo["eval_sec"])
    cfg.agentic_state_path = "models/agentic_tune_state_tmp.json"

    strat = get_strategy("agentic_hybrid", cfg, policy)

    pip = _pip_size(getattr(cfg, "symbol", "EURUSD"))
    times = ticks["time_utc"]
    mid = ((ticks["bid"].astype(float) + ticks["ask"].astype(float)) / 2.0).astype(float)

    eval_step = _rows_for_seconds(ticks, int(cfg.eventless_eval_seconds), default_rows=20)

    pnl_pips: list[float] = []
    for i in range(300, len(ticks) - 2, eval_step):
        now_t = times.iloc[i]
        start_t = now_t - pd.Timedelta(seconds=int(cfg.lookback_seconds))
        start_idx = int(times.searchsorted(start_t, side="left"))
        window = ticks.iloc[start_idx:i].copy()
        if window.empty:
            continue

        event_row = pd.Series({"date_utc": str(now_t), "event_id": f"tune_{i}"})
        decision = strat.decide(event_row, window, None, None, None, None, policy, cfg)
        if decision is None:
            continue

        fut_t = now_t + pd.Timedelta(seconds=eval_horizon_seconds)
        j = int(times.searchsorted(fut_t, side="left"))
        if j >= len(ticks):
            break

        p0 = float(mid.iloc[i])
        p1 = float(mid.iloc[j])
        if p0 <= 0:
            continue

        direction = 1.0 if decision.side == "BUY" else -1.0
        trade_pips = ((p1 - p0) / pip) * direction
        pnl_pips.append(float(trade_pips))

    if len(pnl_pips) < min_signals:
        return None

    arr = np.array(pnl_pips, dtype=float)
    wins = (arr > 0).sum()
    n = len(arr)
    win_rate = float(wins / n)
    avg_pips = float(arr.mean())
    total_pips = float(arr.sum())
    std = float(arr.std(ddof=0))
    sharpe_like = float(avg_pips / std) if std > 1e-9 else 0.0

    # score robusto: retorno + consistencia + preferencia por mayor muestra
    score = (0.65 * avg_pips) + (0.25 * sharpe_like) + (0.10 * (win_rate - 0.5)) + (0.0005 * n)

    return {
        "score": float(score),
        "signals": int(n),
        "win_rate": float(win_rate),
        "avg_pips": float(avg_pips),
        "total_pips": float(total_pips),
        "sharpe_like": float(sharpe_like),
        **{k: (float(v) if isinstance(v, float) else int(v)) for k, v in combo.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tuning de estrategia agentic_hybrid con aplicación opcional a .env")
    parser.add_argument("--market-csv", default=settings.market_csv)
    parser.add_argument("--output", default=str(Path(settings.model_dir) / "agentic_tuning_best.json"))
    parser.add_argument("--env-path", default=str(Path(".env")))
    parser.add_argument("--apply-env", action="store_true", help="Aplica mejor combinación al archivo .env")
    parser.add_argument("--quick", action="store_true", help="Búsqueda más rápida con menos combinaciones")
    parser.add_argument("--eval-horizon-seconds", type=int, default=60)
    parser.add_argument("--min-signals", type=int, default=20)
    args = parser.parse_args()

    market_path = Path(args.market_csv)
    if not market_path.exists():
        raise FileNotFoundError(f"No existe market csv: {market_path}")

    ticks = pd.read_csv(market_path)
    if "time_utc" not in ticks.columns or "bid" not in ticks.columns or "ask" not in ticks.columns:
        raise ValueError("market csv debe incluir columnas time_utc, bid, ask")

    ticks["time_utc"] = pd.to_datetime(ticks["time_utc"], utc=True, errors="coerce")
    ticks = ticks.dropna(subset=["time_utc", "bid", "ask"]).sort_values("time_utc")
    ticks["bid"] = pd.to_numeric(ticks["bid"], errors="coerce")
    ticks["ask"] = pd.to_numeric(ticks["ask"], errors="coerce")
    ticks = ticks.dropna(subset=["bid", "ask"]).copy()

    policy = load_policy(
        settings.model_dir,
        default_threshold=settings.decision_threshold,
        default_no_trade_band=settings.no_trade_band,
    )

    if args.quick:
        grid = {
            "ema_fast": [13, 21],
            "ema_slow": [34, 55],
            "ema_sep": [0.10, 0.20],
            "ema_mom": [0.10, 0.25],
            "ema_vol": [0.03, 0.05],
            "don_q": [0.75, 0.80],
            "don_ch": [0.10, 0.50],
            "don_buf": [0.00, 0.10],
            "lr": [0.10, 0.20],
            "exp": [0.05, 0.10],
            "min_conf": [0.54, 0.56],
            "reward_h": [45, 60],
            "reward_t": [1.0, 1.5],
            "eval_sec": [10, 20],
        }
    else:
        grid = {
            "ema_fast": [13, 21, 34],
            "ema_slow": [34, 55, 89],
            "ema_sep": [0.10, 0.20, 0.30],
            "ema_mom": [0.10, 0.25, 0.40],
            "ema_vol": [0.03, 0.05, 0.08],
            "don_q": [0.70, 0.75, 0.80, 0.85],
            "don_ch": [0.10, 0.50, 1.00],
            "don_buf": [0.00, 0.10, 0.20],
            "lr": [0.08, 0.15, 0.25],
            "exp": [0.03, 0.08, 0.15],
            "min_conf": [0.53, 0.56, 0.60],
            "reward_h": [30, 45, 60],
            "reward_t": [0.8, 1.2, 1.8],
            "eval_sec": [8, 12, 20],
        }

    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]

    scored: list[dict[str, float | int]] = []
    for combo in combos:
        row = evaluate_combo(
            ticks=ticks,
            base_settings=settings,
            policy=policy,
            combo=combo,
            eval_horizon_seconds=int(args.eval_horizon_seconds),
            min_signals=int(args.min_signals),
        )
        if row is not None:
            scored.append(row)

    if not scored:
        raise RuntimeError("No hubo combinaciones válidas para tuning (sube datos o baja min_signals).")

    scored = sorted(scored, key=lambda x: float(x["score"]), reverse=True)
    best = scored[0]

    out_obj = {
        "best": best,
        "top10": scored[:10],
        "meta": {
            "market_csv": str(market_path),
            "tested_combinations": len(combos),
            "valid_combinations": len(scored),
            "eval_horizon_seconds": int(args.eval_horizon_seconds),
            "min_signals": int(args.min_signals),
            "quick": bool(args.quick),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")

    if args.apply_env:
        env_path = Path(args.env_path)
        env_vals = _load_env(env_path)
        env_vals["STRATEGY"] = "agentic_hybrid"
        env_vals["EMA_FAST_SPAN"] = str(int(best["ema_fast"]))
        env_vals["EMA_SLOW_SPAN"] = str(int(best["ema_slow"]))
        env_vals["EMA_MIN_SEPARATION_PIPS"] = f"{float(best['ema_sep']):.2f}"
        env_vals["EMA_MIN_MOMENTUM_PIPS"] = f"{float(best['ema_mom']):.2f}"
        env_vals["EMA_MIN_VOL_PIPS"] = f"{float(best['ema_vol']):.2f}"
        env_vals["DONCHIAN_TRIGGER_QUANTILE"] = f"{float(best['don_q']):.2f}"
        env_vals["DONCHIAN_MIN_CHANNEL_PIPS"] = f"{float(best['don_ch']):.2f}"
        env_vals["DONCHIAN_BREAKOUT_BUFFER_PIPS"] = f"{float(best['don_buf']):.2f}"
        env_vals["AGENTIC_LEARNING_RATE"] = f"{float(best['lr']):.2f}"
        env_vals["AGENTIC_EXPLORE_PROB"] = f"{float(best['exp']):.2f}"
        env_vals["AGENTIC_MIN_CONFIDENCE"] = f"{float(best['min_conf']):.2f}"
        env_vals["AGENTIC_REWARD_HORIZON_SECONDS"] = str(int(best["reward_h"]))
        env_vals["AGENTIC_REWARD_TARGET_PIPS"] = f"{float(best['reward_t']):.2f}"
        env_vals["EVENTLESS_EVAL_SECONDS"] = str(int(best["eval_sec"]))
        env_vals.setdefault("AGENTIC_STATE_PATH", "models/agentic_state.json")
        _save_env(env_path, env_vals)

    print(json.dumps(best, indent=2))
    print(f"Saved tuning result to {out_path}")
    if args.apply_env:
        print(f"Applied best configuration to {args.env_path}")


if __name__ == "__main__":
    main()
