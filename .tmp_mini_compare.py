import os, json
from types import SimpleNamespace
import pandas as pd

from src.config import settings
from src.models import load_artifacts
from src.feature_engineering import build_event_dataset
from src.strategies import get_strategy

N_EVENTS = 4


def load_policy():
    policy_path = os.path.join(settings.model_dir, "trading_policy.json")
    if os.path.exists(policy_path):
        try:
            with open(policy_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"decision_threshold": settings.decision_threshold, "no_trade_band": settings.no_trade_band}


def run_mode(mode_name: str, cfg, bundle, events, ticks, tabular, lstm, feat_cols, policy):
    strat = get_strategy("fundamental_llm", cfg, policy)

    # capture rationale to identify fallback usage
    last_rationale = {"text": ""}
    orig = strat.engine.analyze
    def wrapped(symbol):
        out = orig(symbol)
        last_rationale["text"] = getattr(out, "rationale", "") if out is not None else ""
        return out
    strat.engine.analyze = wrapped

    events_by_id = events.set_index("event_id")
    ticks = ticks.sort_values("time_utc") if not ticks.empty else ticks
    times = ticks["time_utc"] if (not ticks.empty and "time_utc" in ticks.columns) else None

    rows = []
    total = int(getattr(bundle.X_tabular, "shape", [0])[0]) if hasattr(bundle, "X_tabular") else 0
    k = min(N_EVENTS, total)

    for i in range(k):
        ev_id = bundle.event_ids.iloc[i]
        event_time = pd.to_datetime(bundle.event_times.iloc[i], utc=True, errors="coerce")
        try:
            ev_row = events_by_id.loc[ev_id]
        except Exception:
            ev_row = pd.Series(dtype=object)

        if times is not None and pd.notna(event_time):
            idx = int(times.searchsorted(event_time, side="right"))
            ticks_up_to = ticks.iloc[:idx].copy()
        else:
            ticks_up_to = pd.DataFrame()

        dec = strat.decide(ev_row, ticks_up_to, bundle, tabular, lstm, feat_cols, policy, cfg)
        rat = str(last_rationale.get("text", "") or "")
        rows.append({
            "mode": mode_name,
            "i": i,
            "event_id": ev_id,
            "decision": None if dec is None else str(dec.side),
            "confidence": None if dec is None else float(dec.confidence),
            "proba_buy": None if dec is None else float(getattr(dec, "proba_buy", 0.5)),
            "heuristic_flag": ("Heuristic sentiment" in rat) or ("Mixed macro sentiment" in rat) or ("No recent headlines" in rat),
            "rationale_preview": rat[:120],
        })

    return pd.DataFrame(rows)


def main():
    events = pd.read_csv(settings.events_csv)
    ticks = pd.read_csv(settings.market_csv, parse_dates=["time_utc"]) if os.path.exists(settings.market_csv) else pd.DataFrame()
    if not ticks.empty:
        ticks["time_utc"] = pd.to_datetime(ticks["time_utc"], utc=True)

    tabular, lstm, feat_cols = load_artifacts(settings.model_dir)
    bundle = build_event_dataset(events, ticks, lookback_seconds=settings.lookback_seconds)
    policy = load_policy()

    base = SimpleNamespace(**vars(settings))
    base.strategy = "fundamental_llm"
    base.agent_manage_all_strategies = False
    base.gemini_model = "gemini-3.1-pro-preview"
    base.fundamental_llm_max_tokens = 700

    llm_cfg = SimpleNamespace(**vars(base))
    llm_cfg.fundamental_use_heuristic_fallback = False

    heur_cfg = SimpleNamespace(**vars(base))
    heur_cfg.gemini_api_key = ""
    heur_cfg.fundamental_llm_api_key = ""
    heur_cfg.fundamental_use_heuristic_fallback = True

    df_llm = run_mode("llm_only", llm_cfg, bundle, events, ticks, tabular, lstm, feat_cols, policy)
    df_heur = run_mode("heuristic_only", heur_cfg, bundle, events, ticks, tabular, lstm, feat_cols, policy)
    out = pd.concat([df_llm, df_heur], ignore_index=True)

    os.makedirs(os.path.join(settings.data_dir, "comparison"), exist_ok=True)
    out_path = os.path.join(settings.data_dir, "comparison", "mini_compare_fundamental_llm_vs_heuristic.csv")
    out.to_csv(out_path, index=False)

    print("samples_per_mode=", N_EVENTS)
    for mode, g in out.groupby("mode"):
        n = len(g)
        n_sig = int(g["decision"].notna().sum())
        n_buy = int((g["decision"] == "BUY").sum())
        n_sell = int((g["decision"] == "SELL").sum())
        avg_conf = float(g["confidence"].dropna().mean()) if g["confidence"].notna().any() else 0.0
        h_used = int(g["heuristic_flag"].sum())
        print(f"mode={mode} rows={n} signals={n_sig} buy={n_buy} sell={n_sell} avg_conf={avg_conf:.4f} heuristic_flags={h_used}")

    print("saved=", out_path)

if __name__ == "__main__":
    main()
