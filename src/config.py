from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta, timezone
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    te_api_key: str = os.getenv("TE_API_KEY", "")
    te_base_url: str = os.getenv("TE_BASE_URL", "https://api.tradingeconomics.com")

    broker_timezone: str = os.getenv("BROKER_TIMEZONE", "UTC")
    utc_offset_hours: int = int(os.getenv("UTC_OFFSET_HOURS", "-5"))
    symbol: str = os.getenv("SYMBOL", "EURUSD")
    event_min_importance: int = int(os.getenv("EVENT_MIN_IMPORTANCE", "2"))
    event_include_keywords: str = os.getenv("EVENT_INCLUDE_KEYWORDS", "")
    event_exclude_keywords: str = os.getenv("EVENT_EXCLUDE_KEYWORDS", "")
    order_volume: float = float(os.getenv("ORDER_VOLUME", "0.10"))
    max_spread_points: int = int(os.getenv("MAX_SPREAD_POINTS", "25"))

    stop_loss_pips: float = float(os.getenv("STOP_LOSS_PIPS", "12"))
    take_profit_pips: float = float(os.getenv("TAKE_PROFIT_PIPS", "24"))
    trailing_stop_pips: float = float(os.getenv("TRAILING_STOP_PIPS", "8"))
    seconds_before_event: int = int(os.getenv("SECONDS_BEFORE_EVENT", "10"))
    decision_threshold: float = float(os.getenv("DECISION_THRESHOLD", "0.60"))
    no_trade_band: float = float(os.getenv("NO_TRADE_BAND", "0.05"))
    paper_trading: bool = os.getenv("PAPER_TRADING", "true").strip().lower() in {"1", "true", "yes", "y"}
    max_open_positions: int = int(os.getenv("MAX_OPEN_POSITIONS", "1"))

    # Strategy selection and Z-score parameters
    strategy: str = os.getenv("STRATEGY", "default")
    z_score_lookback_seconds: int = int(os.getenv("Z_SCORE_LOOKBACK_SECONDS", "300"))
    z_score_threshold: float = float(os.getenv("Z_SCORE_THRESHOLD", "0.7"))
    z_weight: float = float(os.getenv("Z_WEIGHT", "1.0"))
    z_combination_mode: str = os.getenv("Z_COMBINATION_MODE", "weighted")  # options: weighted, conjunctive

    # Momentum strategy parameters
    momentum_lookback_seconds: int = int(os.getenv("MOMENTUM_LOOKBACK_SECONDS", "300"))
    momentum_threshold: float = float(os.getenv("MOMENTUM_THRESHOLD", "0.0005"))
    momentum_weight: float = float(os.getenv("MOMENTUM_WEIGHT", "1.0"))
    momentum_mode: str = os.getenv("MOMENTUM_MODE", "weighted")

    # Donchian breakout strategy parameters
    donchian_lookback_seconds: int = int(os.getenv("DONCHIAN_LOOKBACK_SECONDS", "600"))
    donchian_breakout_buffer_pips: float = float(os.getenv("DONCHIAN_BREAKOUT_BUFFER_PIPS", "0.2"))
    donchian_min_channel_pips: float = float(os.getenv("DONCHIAN_MIN_CHANNEL_PIPS", "1.0"))
    donchian_confirm_ticks: int = int(os.getenv("DONCHIAN_CONFIRM_TICKS", "1"))
    donchian_trigger_quantile: float = float(os.getenv("DONCHIAN_TRIGGER_QUANTILE", "0.80"))

    data_dir: str = os.getenv("DATA_DIR", "data")
    model_dir: str = os.getenv("MODEL_DIR", "models")
    events_csv: str = os.getenv("EVENTS_CSV", "data/events.csv")
    market_csv: str = os.getenv("MARKET_CSV", "data/market_ticks.csv")

    train_window_days: int = int(os.getenv("TRAIN_WINDOW_DAYS", "180"))
    lookback_seconds: int = int(os.getenv("LOOKBACK_SECONDS", "300"))
    direction_label_mode: str = os.getenv("DIRECTION_LABEL_MODE", "sign")
    strict_monthly_validation: bool = os.getenv("STRICT_MONTHLY_VALIDATION", "false").strip().lower() in {"1", "true", "yes", "y"}
    long_history_months: int = int(os.getenv("LONG_HISTORY_MONTHS", "12"))
    synthetic_event_interval_hours: int = int(os.getenv("SYNTHETIC_EVENT_INTERVAL_HOURS", "6"))

    mt5_login: int = int(os.getenv("MT5_LOGIN", "0"))
    mt5_password: str = os.getenv("MT5_PASSWORD", "")
    mt5_server: str = os.getenv("MT5_SERVER", "")

    @property
    def local_tz(self) -> timezone:
        return timezone(timedelta(hours=self.utc_offset_hours))


settings = Settings()
