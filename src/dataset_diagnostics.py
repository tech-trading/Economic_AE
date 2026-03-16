from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.config import settings
from src.feature_engineering import build_event_dataset


def main() -> None:
    events = pd.read_csv(settings.events_csv)
    market = pd.read_csv(settings.market_csv)

    bundle = build_event_dataset(events, market, lookback_seconds=settings.lookback_seconds)
    if bundle.X_tabular.empty:
        raise RuntimeError("No samples found for diagnostics")

    times = pd.to_datetime(bundle.event_times, utc=True)
    month = times.dt.tz_localize(None).dt.to_period("M").astype(str)

    df = pd.DataFrame(
        {
            "month": month,
            "y_direction": bundle.y_direction,
            "y_volatility": bundle.y_volatility,
        }
    )

    diag = (
        df.groupby("month")
        .agg(
            samples=("y_direction", "size"),
            direction_0=("y_direction", lambda s: int((s == 0).sum())),
            direction_1=("y_direction", lambda s: int((s == 1).sum())),
            volatility_0=("y_volatility", lambda s: int((s == 0).sum())),
            volatility_1=("y_volatility", lambda s: int((s == 1).sum())),
        )
        .reset_index()
    )

    diag["direction_pos_rate"] = np.where(diag["samples"] > 0, diag["direction_1"] / diag["samples"], 0.0)
    diag["volatility_pos_rate"] = np.where(diag["samples"] > 0, diag["volatility_1"] / diag["samples"], 0.0)

    os.makedirs(settings.model_dir, exist_ok=True)
    out_path = os.path.join(settings.model_dir, "dataset_monthly_diagnostics.csv")
    diag.to_csv(out_path, index=False)

    print("Dataset monthly diagnostics:")
    print(diag)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
