from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import settings
from src.feature_engineering import build_event_dataset
from src.models import load_artifacts
from src.strategies import get_strategy


def _load_policy() -> dict:
    path = os.path.join(settings.model_dir, "trading_policy.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"decision_threshold": settings.decision_threshold, "no_trade_band": settings.no_trade_band}


def _event_slice_decision(strat, bundle, events_by_id, ticks, idx, policy, cfg):
    ev_id = bundle.event_ids.iloc[idx]
    event_time = pd.to_datetime(bundle.event_times.iloc[idx], utc=True, errors="coerce")
    try:
        ev_row = events_by_id.loc[ev_id]
    except Exception:
        ev_row = pd.Series(dtype=object)

    if ticks.empty:
        ticks_up_to = pd.DataFrame()
    else:
        times = ticks["time_utc"]
        cut = int(times.searchsorted(event_time, side="right")) if pd.notna(event_time) else len(ticks)
        ticks_up_to = ticks.iloc[:cut].copy()

    dec = strat.decide(ev_row, ticks_up_to, bundle, {}, None, [], policy, cfg)
    meta_getter = getattr(strat, "get_last_signal_meta", None)
    meta = meta_getter() if callable(meta_getter) else {}

    return {
        "event_id": ev_id,
        "event_time": str(event_time),
        "decision": None if dec is None else dec.side,
        "confidence": None if dec is None else float(dec.confidence),
        "proba_buy": None if dec is None else float(getattr(dec, "proba_buy", 0.5)),
        "analysis_source": str(meta.get("analysis_source", "")),
        "news_changed": bool(meta.get("news_changed", False)),
        "news_signature": str(meta.get("news_signature", "")),
    }


def run_rolling(bundle, events, ticks, policy, cfg, window_size: int, stride: int, max_events: int) -> pd.DataFrame:
    strat = get_strategy("fundamental_llm", cfg, policy)
    events_by_id = events.set_index("event_id")

    total = int(bundle.X_tabular.shape[0]) if hasattr(bundle, "X_tabular") else 0
    total = min(total, max_events)
    rows = []

    w_id = 0
    start = 0
    while start < total:
        end = min(total, start + window_size)
        for i in range(start, end):
            row = _event_slice_decision(strat, bundle, events_by_id, ticks, i, policy, cfg)
            row["window_id"] = w_id
            row["window_start_idx"] = start
            row["window_end_idx"] = end - 1
            rows.append(row)
        if end >= total:
            break
        start += max(1, stride)
        w_id += 1

    return pd.DataFrame(rows)


def run_threshold_sensitivity(bundle, events, ticks, policy, cfg, max_events: int, thresholds: list[float]) -> pd.DataFrame:
    events_by_id = events.set_index("event_id")
    total = int(bundle.X_tabular.shape[0]) if hasattr(bundle, "X_tabular") else 0
    total = min(total, max_events)
    rows = []

    for th in thresholds:
        cfg_th = SimpleNamespace(**vars(cfg))
        cfg_th.fundamental_decision_threshold = float(th)
        strat = get_strategy("fundamental_llm", cfg_th, policy)

        decisions = []
        for i in range(total):
            r = _event_slice_decision(strat, bundle, events_by_id, ticks, i, policy, cfg_th)
            decisions.append(r)

        df = pd.DataFrame(decisions)
        n = len(df)
        sig = int(df["decision"].notna().sum())
        rows.append(
            {
                "threshold": float(th),
                "rows": int(n),
                "signals": int(sig),
                "buy": int((df["decision"] == "BUY").sum()),
                "sell": int((df["decision"] == "SELL").sum()),
                "no_signal": int(n - sig),
                "avg_confidence": float(df["confidence"].dropna().mean()) if df["confidence"].notna().any() else 0.0,
                "news_changed_flags": int(df["news_changed"].sum()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling + sensitivity evaluation for fundamental_llm strategy")
    parser.add_argument("--max-events", type=int, default=60)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--thresholds", default="0.55,0.60,0.65,0.70,0.75")
    args = parser.parse_args()

    events = pd.read_csv(settings.events_csv)
    ticks = pd.read_csv(settings.market_csv, parse_dates=["time_utc"]) if os.path.exists(settings.market_csv) else pd.DataFrame()
    if not ticks.empty:
        ticks["time_utc"] = pd.to_datetime(ticks["time_utc"], utc=True)
        ticks = ticks.sort_values("time_utc")

    # Keep compatibility with current project artifacts loading.
    _tab, _lstm, _feat = load_artifacts(settings.model_dir)
    bundle = build_event_dataset(events, ticks, lookback_seconds=settings.lookback_seconds)
    policy = _load_policy()

    cfg = SimpleNamespace(**vars(settings))
    cfg.strategy = "fundamental_llm"
    cfg.agent_manage_all_strategies = False

    out_dir = os.path.join(settings.data_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    df_roll = run_rolling(
        bundle=bundle,
        events=events,
        ticks=ticks,
        policy=policy,
        cfg=cfg,
        window_size=max(2, int(args.window_size)),
        stride=max(1, int(args.stride)),
        max_events=max(5, int(args.max_events)),
    )
    roll_path = os.path.join(out_dir, "fundamental_rolling_windows.csv")
    df_roll.to_csv(roll_path, index=False)

    thr = [float(x.strip()) for x in str(args.thresholds).split(",") if x.strip()]
    df_sens = run_threshold_sensitivity(
        bundle=bundle,
        events=events,
        ticks=ticks,
        policy=policy,
        cfg=cfg,
        max_events=max(5, int(args.max_events)),
        thresholds=thr,
    )
    sens_path = os.path.join(out_dir, "fundamental_threshold_sensitivity.csv")
    df_sens.to_csv(sens_path, index=False)

    summary = {
        "max_events": int(args.max_events),
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "thresholds": thr,
        "rolling_rows": int(len(df_roll)),
        "sensitivity_rows": int(len(df_sens)),
        "rolling_path": roll_path,
        "sensitivity_path": sens_path,
    }
    summary_path = os.path.join(out_dir, "fundamental_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print("saved_rolling=", roll_path)
    print("saved_sensitivity=", sens_path)
    print("saved_summary=", summary_path)
    print(df_sens.to_string(index=False))


if __name__ == "__main__":
    main()
