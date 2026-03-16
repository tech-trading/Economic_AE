from __future__ import annotations

from src.config import settings
from src.data_collection import collect_long_history_from_m1_bars, init_mt5, shutdown_mt5


def main() -> None:
    init_mt5()
    try:
        events_df, market_df = collect_long_history_from_m1_bars(
            months_back=settings.long_history_months,
            interval_hours=settings.synthetic_event_interval_hours,
            out_events_csv="data/events_monthly.csv",
            out_market_csv="data/market_ticks_monthly.csv",
        )
        print(f"Monthly dataset prepared: events={len(events_df)}, market_rows={len(market_df)}")
        print("Files: data/events_monthly.csv, data/market_ticks_monthly.csv")
    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()
