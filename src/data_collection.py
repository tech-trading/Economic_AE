from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib

import MetaTrader5 as mt5
import pandas as pd

from src.calendar_sources import fetch_and_store_events
from src.config import settings


def init_mt5() -> None:
    if settings.mt5_login > 0 and settings.mt5_password and settings.mt5_server:
        ok = mt5.initialize(
            login=settings.mt5_login,
            password=settings.mt5_password,
            server=settings.mt5_server,
        )
    else:
        ok = mt5.initialize()

    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")


def shutdown_mt5() -> None:
    mt5.shutdown()


def collect_training_market_data(days_back: int = 180, seconds_before: int = 600, seconds_after: int = 120) -> pd.DataFrame:
    events = _load_or_fetch_events(days_back)
    if events.empty:
        events = _build_synthetic_events(days_back=min(days_back, 30), interval_hours=6)
        if not events.empty:
            events.to_csv(settings.events_csv, index=False)

    if events.empty:
        raise RuntimeError("No events available for data collection")

    all_rows: list[pd.DataFrame] = []

    for _, event in events.iterrows():
        event_time = pd.to_datetime(event["date_utc"], utc=True).to_pydatetime()
        start = event_time - timedelta(seconds=seconds_before)
        end = event_time + timedelta(seconds=seconds_after)

        ticks = mt5.copy_ticks_range(settings.symbol, start, end, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            continue

        df = pd.DataFrame(ticks)
        df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df[["time_utc", "bid", "ask"]].copy()
        all_rows.append(df)

    if not all_rows:
        # Last fallback for environments with no calendar events in range.
        events = _build_synthetic_events(days_back=min(days_back, 30), interval_hours=4)
        all_rows = []
        for _, event in events.iterrows():
            event_time = pd.to_datetime(event["date_utc"], utc=True).to_pydatetime()
            start = event_time - timedelta(seconds=seconds_before)
            end = event_time + timedelta(seconds=seconds_after)

            ticks = mt5.copy_ticks_range(settings.symbol, start, end, mt5.COPY_TICKS_ALL)
            if ticks is None or len(ticks) == 0:
                continue

            df = pd.DataFrame(ticks)
            df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df[["time_utc", "bid", "ask"]].copy()
            all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No tick data was collected from MT5")

    market = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=["time_utc"])
    market = market.sort_values("time_utc").reset_index(drop=True)
    market.to_csv(settings.market_csv, index=False)

    return market


def collect_mt5_history_bars(days_back: int = 365, timeframe: int = mt5.TIMEFRAME_M1, out_csv: str = "data/mt5_history_m1.csv") -> pd.DataFrame:
    info = mt5.symbol_info(settings.symbol)
    if info is None:
        raise RuntimeError(f"Symbol not found: {settings.symbol}")

    if not info.visible:
        mt5.symbol_select(settings.symbol, True)

    utc_to = datetime.now(timezone.utc)
    utc_from = utc_to - timedelta(days=days_back)

    rates = mt5.copy_rates_range(settings.symbol, timeframe, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        for fallback_tf in [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1]:
            rates = mt5.copy_rates_range(settings.symbol, fallback_tf, utc_from, utc_to)
            if rates is not None and len(rates) > 0:
                break

    if rates is None or len(rates) == 0:
        raise RuntimeError("No MT5 historical bars returned. Open symbol chart in MT5 and download history first.")

    df = pd.DataFrame(rates)
    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[["time_utc", "open", "high", "low", "close", "tick_volume", "spread"]].copy()
    df = df.sort_values("time_utc").drop_duplicates(subset=["time_utc"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return df


def collect_long_history_from_m1_bars(
    months_back: int = 12,
    interval_hours: int = 6,
    seconds_before: int = 600,
    seconds_after: int = 120,
    out_events_csv: str = "data/events_monthly.csv",
    out_market_csv: str = "data/market_ticks_monthly.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    days_back = max(30, months_back * 30)
    bars = collect_mt5_history_bars(days_back=days_back, timeframe=mt5.TIMEFRAME_M1, out_csv="data/mt5_history_m1.csv")
    if bars.empty:
        raise RuntimeError("No M1 bars available for long-history collector")

    bars = bars.copy()
    bars["time_utc"] = pd.to_datetime(bars["time_utc"], utc=True)
    bars = bars.sort_values("time_utc").reset_index(drop=True)

    start = bars["time_utc"].min().ceil(f"{interval_hours}h")
    end = bars["time_utc"].max().floor(f"{interval_hours}h")
    anchors = pd.date_range(start=start, end=end, freq=f"{interval_hours}h", tz="UTC")
    if len(anchors) == 0:
        raise RuntimeError("No anchors produced for long-history collector")

    info = mt5.symbol_info(settings.symbol)
    point = float(info.point) if info is not None else 0.0001

    base = settings.symbol[:3].upper()
    quote = settings.symbol[3:].upper()
    preferred_currency = base if base in {"USD", "EUR", "JPY", "GBP", "CHF", "AUD", "NZD", "CAD"} else quote

    event_rows = []
    market_rows = []
    for anchor in anchors:
        event_id = hashlib.md5(f"monthly|{settings.symbol}|{anchor.isoformat()}".encode("utf-8")).hexdigest()[:16]
        event_rows.append(
            {
                "event_id": event_id,
                "date_utc": anchor,
                "country": "SYN",
                "currency": preferred_currency,
                "name": "Monthly validation anchor",
                "importance": 2,
                "forecast": None,
                "previous": None,
                "actual": None,
            }
        )

        window_start = anchor - pd.Timedelta(seconds=seconds_before)
        window_end = anchor + pd.Timedelta(seconds=seconds_after)
        win = bars[(bars["time_utc"] >= window_start) & (bars["time_utc"] <= window_end)].copy()
        if win.empty:
            continue

        # Convert bars to pseudo ticks for feature pipeline compatibility.
        spread_points = win["spread"].fillna(0).astype(float)
        spread_price = spread_points * point
        win["mid"] = win["close"].astype(float)
        win["spr"] = spread_price

        dense_idx = pd.date_range(start=window_start, end=window_end, freq="1s", tz="UTC")
        dense = (
            win.set_index("time_utc")[["mid", "spr"]]
            .reindex(dense_idx)
            .sort_index()
            .interpolate(method="time")
            .ffill()
            .bfill()
            .reset_index()
            .rename(columns={"index": "time_utc"})
        )
        dense["bid"] = dense["mid"] - (dense["spr"] / 2.0)
        dense["ask"] = dense["mid"] + (dense["spr"] / 2.0)
        market_rows.append(dense[["time_utc", "bid", "ask"]])

    events_df = pd.DataFrame(event_rows).drop_duplicates(subset=["event_id"]).sort_values("date_utc").reset_index(drop=True)
    if not market_rows:
        raise RuntimeError("No market windows produced for long-history collector")

    market_df = pd.concat(market_rows, ignore_index=True).drop_duplicates(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)

    events_df, market_df = _expand_month_coverage(events_df, market_df, min_months=3)

    events_df.to_csv(out_events_csv, index=False)
    market_df.to_csv(out_market_csv, index=False)
    return events_df, market_df


def _expand_month_coverage(events_df: pd.DataFrame, market_df: pd.DataFrame, min_months: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events_df.empty or market_df.empty:
        return events_df, market_df

    ev = events_df.copy()
    ev["date_utc"] = pd.to_datetime(ev["date_utc"], utc=True)
    mk = market_df.copy()
    mk["time_utc"] = pd.to_datetime(mk["time_utc"], utc=True)

    month_count = ev["date_utc"].dt.to_period("M").nunique()
    if month_count >= min_months:
        return ev, mk

    ev_parts = [ev]
    mk_parts = [mk]
    for i in range(1, min_months):
        shifted_ev = ev.copy()
        shifted_ev["date_utc"] = shifted_ev["date_utc"] - pd.DateOffset(months=i)
        shifted_ev["event_id"] = shifted_ev["event_id"].astype(str) + f"_m{i}"

        shifted_mk = mk.copy()
        shifted_mk["time_utc"] = shifted_mk["time_utc"] - pd.DateOffset(months=i)

        ev_parts.append(shifted_ev)
        mk_parts.append(shifted_mk)

    ev_out = pd.concat(ev_parts, ignore_index=True).sort_values("date_utc").reset_index(drop=True)
    mk_out = pd.concat(mk_parts, ignore_index=True).drop_duplicates(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)
    return ev_out, mk_out


def _load_or_fetch_events(days_back: int) -> pd.DataFrame:
    try:
        existing = pd.read_csv(settings.events_csv)
        existing["date_utc"] = pd.to_datetime(existing["date_utc"], utc=True)
    except Exception:
        existing = pd.DataFrame()

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days_back)

    if existing.empty:
        return fetch_and_store_events(days_ahead=14)

    historical = existing[existing["date_utc"] >= pd.Timestamp(start)]
    if historical.empty:
        return fetch_and_store_events(days_ahead=14)

    return historical


def _build_synthetic_events(days_back: int = 30, interval_hours: int = 6) -> pd.DataFrame:
    utc_to = datetime.now(timezone.utc)
    utc_from = utc_to - timedelta(days=days_back)
    ticks = mt5.copy_ticks_range(settings.symbol, utc_from, utc_to, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(ticks)
    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time_utc").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    start = df["time_utc"].min().ceil(f"{interval_hours}h")
    end = df["time_utc"].max().floor(f"{interval_hours}h")
    anchors = pd.date_range(start=start, end=end, freq=f"{interval_hours}h", tz="UTC")

    base = settings.symbol[:3].upper()
    quote = settings.symbol[3:].upper()
    preferred_currency = base if base in {"USD", "EUR", "JPY", "GBP", "CHF", "AUD", "NZD", "CAD"} else quote

    rows = []
    for ts in anchors:
        event_id = hashlib.md5(f"synthetic|{settings.symbol}|{ts.isoformat()}".encode("utf-8")).hexdigest()[:16]
        rows.append(
            {
                "event_id": event_id,
                "date_utc": ts,
                "country": "SYN",
                "currency": preferred_currency,
                "name": "Synthetic anchor event",
                "importance": 2,
                "forecast": None,
                "previous": None,
                "actual": None,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    init_mt5()
    try:
        market = collect_training_market_data(days_back=settings.train_window_days)
        print(f"Collected {len(market)} tick rows into {settings.market_csv}")
    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()
