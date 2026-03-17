from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import settings
from src.feature_engineering import build_event_dataset
from src.strategies import get_strategy


def main() -> None:
    events_path = Path("data/events.csv")
    ticks_path = Path("data/market_ticks.csv")
    out_csv = Path("models/donchian_nylondon_monthly_validation.csv")
    out_json = Path("models/donchian_nylondon_validation_summary.json")

    if not events_path.exists() or not ticks_path.exists():
        raise FileNotFoundError(f"Inputs no disponibles: events={events_path} ticks={ticks_path}")

    events = pd.read_csv(events_path)
    ticks = pd.read_csv(ticks_path, parse_dates=["time_utc"])
    ticks["time_utc"] = pd.to_datetime(ticks["time_utc"], utc=True, errors="coerce")
    ticks = ticks.dropna(subset=["time_utc"]).sort_values("time_utc")

    bundle = build_event_dataset(events, ticks, lookback_seconds=settings.lookback_seconds)
    if bundle.X_tabular.empty:
        raise RuntimeError("No hay muestras para validar (X_tabular vacío).")

    events_by_id = events.set_index("event_id") if "event_id" in events.columns else pd.DataFrame()
    strategy = get_strategy("donchian_nylondon", settings, {"decision_threshold": settings.decision_threshold})

    rows = []
    times = ticks["time_utc"]
    for i in range(bundle.X_tabular.shape[0]):
        event_time = pd.to_datetime(bundle.event_times.iloc[i], utc=True, errors="coerce")
        if pd.isna(event_time):
            continue
        idx = int(times.searchsorted(event_time, side="right"))
        ticks_up_to = ticks.iloc[:idx]

        ev_id = bundle.event_ids.iloc[i]
        ev_row = events_by_id.loc[ev_id] if (not events_by_id.empty and ev_id in events_by_id.index) else pd.Series(dtype=object)

        decision = strategy.decide(
            ev_row,
            ticks_up_to,
            bundle,
            None,
            None,
            None,
            {"decision_threshold": settings.decision_threshold},
            settings,
        )
        if decision is None:
            continue

        ret_post = float(bundle.ret_post[i])
        correct = ((decision.side == "BUY") and (ret_post > 0.0)) or ((decision.side == "SELL") and (ret_post < 0.0))
        pnl_proxy = ret_post if decision.side == "BUY" else -ret_post
        month = event_time.to_period("M").strftime("%Y-%m")

        rows.append(
            {
                "month": month,
                "event_time_utc": str(event_time),
                "event_id": str(ev_id),
                "side": decision.side,
                "confidence": float(decision.confidence),
                "proba_buy": float(getattr(decision, "proba_buy", 0.5)),
                "ret_post": float(ret_post),
                "pnl_proxy": float(pnl_proxy),
                "is_correct": int(correct),
            }
        )

    if not rows:
        raise RuntimeError("La estrategia no generó señales en la validación mensual.")

    raw_df = pd.DataFrame(rows)
    monthly = (
        raw_df.groupby("month", as_index=False)
        .agg(
            signals=("is_correct", "count"),
            accuracy=("is_correct", "mean"),
            total_pnl_proxy=("pnl_proxy", "sum"),
            avg_pnl_proxy=("pnl_proxy", "mean"),
            avg_confidence=("confidence", "mean"),
            buys=("side", lambda s: int((s == "BUY").sum())),
            sells=("side", lambda s: int((s == "SELL").sum())),
        )
        .sort_values("month")
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(out_csv, index=False)

    summary = {
        "strategy": "donchian_nylondon",
        "signals_total": int(raw_df.shape[0]),
        "accuracy": float(raw_df["is_correct"].mean()),
        "total_pnl_proxy": float(raw_df["pnl_proxy"].sum()),
        "avg_pnl_proxy": float(raw_df["pnl_proxy"].mean()),
        "months": int(monthly.shape[0]),
        "output_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved monthly validation to {out_csv}")


if __name__ == "__main__":
    main()
