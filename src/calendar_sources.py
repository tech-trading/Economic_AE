from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from datetime import datetime, timedelta, timezone
from typing import List
import hashlib
import re

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import settings


COUNTRY_TO_CURRENCY = {
    "US": "USD",
    "EU": "EUR",
    "DE": "EUR",
    "FR": "EUR",
    "ES": "EUR",
    "IT": "EUR",
    "NL": "EUR",
    "BE": "EUR",
    "PT": "EUR",
    "IE": "EUR",
    "AT": "EUR",
    "FI": "EUR",
    "GR": "EUR",
    "JP": "JPY",
    "GB": "GBP",
    "UK": "GBP",
    "CH": "CHF",
    "CA": "CAD",
    "AU": "AUD",
    "NZ": "NZD",
    "CN": "CNY",
}


@dataclass
class EconomicEvent:
    event_id: str
    date_utc: datetime
    country: str
    currency: str
    name: str
    importance: int
    forecast: float | None
    previous: float | None
    actual: float | None


class TradingEconomicsCalendarClient:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def fetch_events(self, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
        if not self.api_key:
            raise ValueError("TE_API_KEY is required for Trading Economics API access")

        start_str = start_utc.strftime("%Y-%m-%d")
        end_str = end_utc.strftime("%Y-%m-%d")
        url = f"{self.base_url}/calendar/country/all/{start_str}/{end_str}?c={self.api_key}"

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        raw = response.json()

        rows = []
        for item in raw:
            dt = pd.to_datetime(item.get("Date"), utc=True, errors="coerce")
            if pd.isna(dt):
                continue

            importance = _normalize_importance(item.get("Importance"))
            rows.append(
                {
                    "event_id": str(item.get("CalendarId", "")),
                    "date_utc": dt,
                    "country": item.get("Country", ""),
                    "currency": item.get("Currency", ""),
                    "name": item.get("Event", ""),
                    "importance": importance,
                    "forecast": _to_float(item.get("Forecast")),
                    "previous": _to_float(item.get("Previous")),
                    "actual": _to_float(item.get("Actual")),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        return df.sort_values("date_utc").reset_index(drop=True)


def filter_relevant_events(events: pd.DataFrame, symbol: str, min_importance: int = 2) -> pd.DataFrame:
    if events.empty:
        return events

    base = symbol[:3].upper()
    quote = symbol[3:].upper()

    filtered = events[
        (events["importance"] >= min_importance)
        & (events["currency"].str.upper().isin([base, quote]))
    ].copy()

    include_kw = [x.strip().lower() for x in (settings.event_include_keywords or "").split(",") if x.strip()]
    exclude_kw = [x.strip().lower() for x in (settings.event_exclude_keywords or "").split(",") if x.strip()]

    if include_kw:
        filtered = filtered[
            filtered["name"].astype(str).str.lower().apply(lambda s: any(k in s for k in include_kw))
        ]

    if exclude_kw:
        filtered = filtered[
            ~filtered["name"].astype(str).str.lower().apply(lambda s: any(k in s for k in exclude_kw))
        ]

    return filtered.sort_values("date_utc").reset_index(drop=True)


def fetch_and_store_events(days_ahead: int = 14) -> pd.DataFrame:
    now_utc = datetime.now(timezone.utc)
    end_utc = now_utc + timedelta(days=days_ahead)

    te_key = (settings.te_api_key or "").strip()
    if te_key and not te_key.upper().startswith("YOUR_"):
        client = TradingEconomicsCalendarClient(te_key, settings.te_base_url)
        events = client.fetch_events(now_utc, end_utc)
    else:
        events = scrape_public_calendars(days_ahead=max(days_ahead, 1))

    events = filter_relevant_events(events, settings.symbol, min_importance=settings.event_min_importance)

    if not events.empty:
        _append_unique_events(events, settings.events_csv)
    else:
        cached = _load_cached_events(settings.events_csv, now_utc)
        if not cached.empty:
            return filter_relevant_events(cached, settings.symbol, min_importance=settings.event_min_importance)
        _ensure_events_file(settings.events_csv)

    return events


def scrape_public_calendars(days_ahead: int = 14, start_day_local: datetime | None = None) -> pd.DataFrame:
    # Primary source
    try:
        events = scrape_tradingeconomics_calendar(days_ahead=days_ahead, start_day_local=start_day_local)
    except Exception:
        events = pd.DataFrame()
    if not events.empty:
        return events

    # Fallback 1: Investing.com economic calendar page
    try:
        events = scrape_investing_calendar(days_ahead=days_ahead, start_day_local=start_day_local)
    except Exception:
        events = pd.DataFrame()
    if not events.empty:
        return events

    # Fallback 2: FXStreet economic calendar page
    try:
        events = scrape_fxstreet_calendar(days_ahead=days_ahead, start_day_local=start_day_local)
    except Exception:
        events = pd.DataFrame()
    return events


def scrape_tradingeconomics_calendar(days_ahead: int = 14, start_day_local: datetime | None = None) -> pd.DataFrame:
    local_start = (start_day_local or datetime.now(settings.local_tz)).date()
    local_end = local_start + timedelta(days=max(1, days_ahead))

    url = "https://tradingeconomics.com/calendar"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    }
    response = _calendar_get(url=url, headers=headers)
    tables = pd.read_html(StringIO(response.text))
    if not tables:
        return pd.DataFrame()

    table = _pick_calendar_table(tables)
    if table is None or table.empty:
        return pd.DataFrame()

    df = _normalize_scraped_calendar(table, local_start)
    if df.empty:
        return df

    # Keep only the requested local date window.
    local_ts = pd.to_datetime(df["date_utc"], utc=True, errors="coerce").dt.tz_convert(settings.local_tz)
    mask = (local_ts.dt.date >= local_start) & (local_ts.dt.date <= local_end)
    return df[mask].sort_values("date_utc").reset_index(drop=True)


def scrape_investing_calendar(days_ahead: int = 14, start_day_local: datetime | None = None) -> pd.DataFrame:
    local_start = (start_day_local or datetime.now(settings.local_tz)).date()
    local_end = local_start + timedelta(days=max(1, days_ahead))

    url = "https://www.investing.com/economic-calendar/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.investing.com/",
    }
    response = _calendar_get(url=url, headers=headers)
    tables = pd.read_html(StringIO(response.text))
    return _normalize_generic_calendar_tables(
        tables=tables,
        source="investing",
        local_start=local_start,
        local_end=local_end,
    )


def scrape_fxstreet_calendar(days_ahead: int = 14, start_day_local: datetime | None = None) -> pd.DataFrame:
    local_start = (start_day_local or datetime.now(settings.local_tz)).date()
    local_end = local_start + timedelta(days=max(1, days_ahead))

    url = "https://www.fxstreet.com/economic-calendar"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.fxstreet.com/",
    }
    response = _calendar_get(url=url, headers=headers)
    tables = pd.read_html(StringIO(response.text))
    return _normalize_generic_calendar_tables(
        tables=tables,
        source="fxstreet",
        local_start=local_start,
        local_end=local_end,
    )


def _normalize_generic_calendar_tables(
    tables: list[pd.DataFrame],
    source: str,
    local_start,
    local_end,
) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()

    out_rows: list[dict] = []
    for table in tables:
        if table is None or table.empty or table.shape[1] < 3:
            continue

        cols = [str(c).strip() for c in table.columns]
        cols_map = {str(c).lower(): c for c in cols}
        date_col = _find_col(cols_map, ["date", "day", "fecha", "dia"])
        time_col = _find_col(cols_map, ["time", "hora"])
        event_col = _find_col(cols_map, ["event", "release", "title", "headline", "actual event"])
        ccy_col = _find_col(cols_map, ["currency", "curr", "country", "moneda", "pais"])
        impact_col = _find_col(cols_map, ["impact", "importance", "volatility", "importancia"])
        forecast_col = _find_col(cols_map, ["forecast", "consensus", "estimado"])
        previous_col = _find_col(cols_map, ["previous", "prior", "previo", "anterior"])
        actual_col = _find_col(cols_map, ["actual", "real"])

        if event_col is None or ccy_col is None:
            continue

        current_day = local_start
        for _, row in table.iterrows():
            name = str(row.get(event_col, "")).strip()
            if not name or name.lower() in {"nan", "none"}:
                continue

            if date_col is not None:
                raw_date = str(row.get(date_col, "")).strip()
                parsed_date = pd.to_datetime(raw_date, errors="coerce")
                if not pd.isna(parsed_date):
                    current_day = parsed_date.date()

            t_raw = str(row.get(time_col, "")).strip() if time_col is not None else ""
            dt = _parse_scraped_time(t_raw, current_day)
            if dt is None:
                continue

            dt_local = dt.astimezone(settings.local_tz)
            if dt_local.date() < local_start or dt_local.date() > local_end:
                continue

            ccy_raw = str(row.get(ccy_col, "")).strip().upper()
            currency = ccy_raw if len(ccy_raw) == 3 and ccy_raw.isalpha() else _country_to_currency(ccy_raw)
            if not currency:
                continue

            importance = _normalize_importance(row.get(impact_col)) if impact_col is not None else 2
            event_id = hashlib.md5(f"{source}|{dt.isoformat()}|{currency}|{name}".encode("utf-8")).hexdigest()[:16]
            out_rows.append(
                {
                    "event_id": event_id,
                    "date_utc": pd.Timestamp(dt),
                    "country": ccy_raw,
                    "currency": currency,
                    "name": name,
                    "importance": int(importance),
                    "forecast": _to_float(row.get(forecast_col)) if forecast_col is not None else None,
                    "previous": _to_float(row.get(previous_col)) if previous_col is not None else None,
                    "actual": _to_float(row.get(actual_col)) if actual_col is not None else None,
                }
            )

    if not out_rows:
        return pd.DataFrame()

    df = pd.DataFrame(out_rows)
    return df.drop_duplicates(subset=["event_id"]).sort_values("date_utc").reset_index(drop=True)


def _load_cached_events(path: str, now_utc: datetime) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if df.empty or "date_utc" not in df.columns:
        return pd.DataFrame()

    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True, errors="coerce")
    df = df[df["date_utc"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    return df[df["date_utc"] >= pd.Timestamp(now_utc)].sort_values("date_utc").reset_index(drop=True)


def scrape_tradingeconomics_calendar_day(target_day_local: datetime | None = None) -> pd.DataFrame:
    day = (target_day_local or datetime.now(settings.local_tz)).date()
    url = "https://tradingeconomics.com/calendar"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    }

    response = _calendar_get(url=url, headers=headers)

    tables = pd.read_html(StringIO(response.text))
    if not tables:
        return pd.DataFrame()

    table = _pick_calendar_table(tables)
    if table is None or table.empty:
        return pd.DataFrame()

    df = _normalize_scraped_calendar(table, day)
    if df.empty:
        return df

    return df.sort_values("date_utc").reset_index(drop=True)


def _calendar_get(url: str, headers: dict[str, str]) -> requests.Response:
    session = requests.Session()
    retries = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    resp = session.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp


def _pick_calendar_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    time_regex = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?\s?(AM|PM)?$", re.IGNORECASE)
    for table in tables:
        if table.shape[1] < 7:
            continue
        sample = table.head(50)
        score = 0
        for _, row in sample.iterrows():
            first = str(row.iloc[0]).strip().upper() if table.shape[1] > 0 else ""
            second = str(row.iloc[1]).strip().upper() if table.shape[1] > 1 else ""
            third = str(row.iloc[2]).strip() if table.shape[1] > 2 else ""
            if time_regex.match(first):
                score += 1
            if 1 < len(second) <= 3 and second.isalpha():
                score += 1
            if len(third) > 4:
                score += 1
        if score >= 20:
            return table
    return None


def _normalize_scraped_calendar(table: pd.DataFrame, target_day) -> pd.DataFrame:
    day_pattern = re.compile(r"^[A-Za-z]+\s+[A-Za-z]+\s+\d{1,2}\s+\d{4}$")
    current_day = target_day

    rows = []
    for _, row in table.iterrows():
        first = str(row.iloc[0]).strip() if table.shape[1] > 0 else ""
        second = str(row.iloc[1]).strip().upper() if table.shape[1] > 1 else ""
        event_name = str(row.iloc[2]).strip() if table.shape[1] > 2 else ""

        if day_pattern.match(first):
            try:
                current_day = datetime.strptime(first, "%A %B %d %Y").date()
            except ValueError:
                pass
            continue

        if not event_name:
            continue

        dt = _parse_scraped_time(first, current_day)
        if dt is None:
            continue

        country = second
        currency = _country_to_currency(country)
        if not currency:
            # If site provides currency directly in second column, use it.
            currency = second if len(second) in {3, 4} else ""

        if not currency:
            continue

        importance = 2

        event_id = hashlib.md5(f"{dt.isoformat()}|{currency}|{event_name}".encode("utf-8")).hexdigest()[:16]
        rows.append(
            {
                "event_id": event_id,
                "date_utc": pd.Timestamp(dt),
                "country": country,
                "currency": currency,
                "name": event_name,
                "importance": int(importance),
                "forecast": _to_float(row.iloc[6]) if table.shape[1] > 6 else None,
                "previous": _to_float(row.iloc[4]) if table.shape[1] > 4 else None,
                "actual": _to_float(row.iloc[3]) if table.shape[1] > 3 else None,
            }
        )

    return pd.DataFrame(rows)


def _parse_scraped_time(value: str, day) -> datetime | None:
    text = value.strip().upper()
    if not text:
        return None

    formats = ["%I:%M %p", "%H:%M", "%H:%M:%S"]
    for fmt in formats:
        try:
            t = datetime.strptime(text, fmt).time()
            local_dt = datetime.combine(day, t, tzinfo=settings.local_tz)
            return local_dt.astimezone(timezone.utc)
        except ValueError:
            continue

    return None


def _find_col(cols_map: dict[str, object], keys: list[str]):
    for key in keys:
        for col_lower, original in cols_map.items():
            if key in col_lower:
                return original
    return None


def _country_to_currency(country_code: str) -> str:
    key = str(country_code or "").strip().upper()
    return COUNTRY_TO_CURRENCY.get(key, "")


def _append_unique_events(events: pd.DataFrame, path: str) -> None:
    if events.empty:
        return

    try:
        old = pd.read_csv(path)
        old["date_utc"] = pd.to_datetime(old["date_utc"], utc=True, errors="coerce")
    except Exception:
        old = pd.DataFrame(columns=events.columns)

    combined = pd.concat([old, events], ignore_index=True)
    combined = combined.drop_duplicates(subset=["event_id"]).sort_values("date_utc").reset_index(drop=True)
    combined.to_csv(path, index=False)


def _ensure_events_file(path: str) -> None:
    columns = ["event_id", "date_utc", "country", "currency", "name", "importance", "forecast", "previous", "actual"]
    try:
        existing = pd.read_csv(path)
        if not existing.empty or list(existing.columns) == columns:
            return
    except Exception:
        pass

    pd.DataFrame(columns=columns).to_csv(path, index=False)


def _normalize_importance(value: object) -> int:
    if value is None:
        return 0

    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).strip().lower()
    if not text:
        return 0

    # Trading Economics can return text categories.
    if text in {"low", "1"}:
        return 1
    if text in {"medium", "2"}:
        return 2
    if text in {"high", "3"}:
        return 3

    # Common web labels/symbols.
    if "bull" in text:
        bulls = text.count("bull")
        if bulls >= 3:
            return 3
        if bulls == 2:
            return 2
        if bulls == 1:
            return 1
    stars = text.count("*")
    if stars >= 3:
        return 3
    if stars == 2:
        return 2
    if stars == 1:
        return 1

    return 0


def _to_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
