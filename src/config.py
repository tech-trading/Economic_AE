from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta, timezone
from dotenv import load_dotenv


load_dotenv(override=True)


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
    min_sl_spread_multiplier: float = float(os.getenv("MIN_SL_SPREAD_MULTIPLIER", "3.0"))
    min_tp_sl_ratio_enforced: float = float(os.getenv("MIN_TP_SL_RATIO_ENFORCED", "1.6"))
    seconds_before_event: int = int(os.getenv("SECONDS_BEFORE_EVENT", "10"))
    decision_threshold: float = float(os.getenv("DECISION_THRESHOLD", "0.60"))
    no_trade_band: float = float(os.getenv("NO_TRADE_BAND", "0.05"))
    paper_trading: bool = os.getenv("PAPER_TRADING", "true").strip().lower() in {"1", "true", "yes", "y"}
    max_open_positions: int = int(os.getenv("MAX_OPEN_POSITIONS", "1"))
    min_seconds_between_trades: int = int(os.getenv("MIN_SECONDS_BETWEEN_TRADES", "120"))
    max_trades_per_hour: int = int(os.getenv("MAX_TRADES_PER_HOUR", "10"))
    same_side_cooldown_seconds: int = int(os.getenv("SAME_SIDE_COOLDOWN_SECONDS", "180"))

    # Strategy selection and Z-score parameters
    strategy: str = os.getenv("STRATEGY", "default")

    # Dynamic symbol routing by impact news (currency/keyword driven)
    impact_symbol_map: str = os.getenv(
        "IMPACT_SYMBOL_MAP",
        "USD=EURUSD|US500|XAUUSD;EUR=EURUSD|GER40;GBP=GBPUSD|UK100;JPY=USDJPY|JP225;CHF=USDCHF;CAD=USDCAD;AUD=AUDUSD;NZD=NZDUSD",
    )
    impact_keyword_symbol_map: str = os.getenv(
        "IMPACT_KEYWORD_SYMBOL_MAP",
        "oil=XTIUSD|WTI;crude=XTIUSD|WTI;brent=BRENT;gold=XAUUSD;silver=XAGUSD;gas=XNGUSD|NATGAS;nasdaq=NAS100|USTEC;sp500=US500|SPX500;dow=US30",
    )
    impact_prefer_non_default: bool = os.getenv("IMPACT_PREFER_NON_DEFAULT", "true").strip().lower() in {"1", "true", "yes", "y"}
    impact_symbol_fallback_to_default: bool = os.getenv("IMPACT_SYMBOL_FALLBACK_TO_DEFAULT", "true").strip().lower() in {"1", "true", "yes", "y"}

    # Fundamental + LLM strategy parameters
    fundamental_news_sources: str = os.getenv(
        "FUNDAMENTAL_NEWS_SOURCES",
        "https://www.investing.com/rss/news_25.rss,https://feeds.reuters.com/reuters/businessNews,https://www.fxstreet.com/rss/news,https://feeds.marketwatch.com/marketwatch/topstories/",
    )
    fundamental_news_lookback_minutes: int = int(os.getenv("FUNDAMENTAL_NEWS_LOOKBACK_MINUTES", "240"))
    fundamental_news_poll_seconds: int = int(os.getenv("FUNDAMENTAL_NEWS_POLL_SECONDS", "20"))
    fundamental_news_timeout_seconds: int = int(os.getenv("FUNDAMENTAL_NEWS_TIMEOUT_SECONDS", "8"))
    fundamental_max_headlines: int = int(os.getenv("FUNDAMENTAL_MAX_HEADLINES", "30"))
    fundamental_max_headlines_per_source: int = int(os.getenv("FUNDAMENTAL_MAX_HEADLINES_PER_SOURCE", "8"))
    fundamental_signal_cooldown_seconds: int = int(os.getenv("FUNDAMENTAL_SIGNAL_COOLDOWN_SECONDS", "300"))
    fundamental_min_confidence: float = float(os.getenv("FUNDAMENTAL_MIN_CONFIDENCE", "0.60"))
    fundamental_decision_threshold: float = float(os.getenv("FUNDAMENTAL_DECISION_THRESHOLD", "-1"))
    fundamental_reanalyze_seconds: int = int(os.getenv("FUNDAMENTAL_REANALYZE_SECONDS", "15"))
    fundamental_allow_same_side_on_news_change: bool = os.getenv("FUNDAMENTAL_ALLOW_SAME_SIDE_ON_NEWS_CHANGE", "true").strip().lower() in {"1", "true", "yes", "y"}
    fundamental_use_heuristic_fallback: bool = os.getenv("FUNDAMENTAL_USE_HEURISTIC_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "y"}
    fundamental_user_agent: str = os.getenv("FUNDAMENTAL_USER_AGENT", "EconomicAE/1.0 (+research)")

    # OpenAI-compatible endpoint (works with OpenAI, Azure OpenAI compatible gateways, and similar APIs)
    fundamental_llm_api_base_url: str = os.getenv("FUNDAMENTAL_LLM_API_BASE_URL", "https://api.openai.com/v1")
    fundamental_llm_api_key: str = os.getenv("FUNDAMENTAL_LLM_API_KEY", "")
    fundamental_llm_model: str = os.getenv("FUNDAMENTAL_LLM_MODEL", "gpt-4o-mini")
    fundamental_llm_temperature: float = float(os.getenv("FUNDAMENTAL_LLM_TEMPERATURE", "0.10"))
    fundamental_llm_max_tokens: int = int(os.getenv("FUNDAMENTAL_LLM_MAX_TOKENS", "250"))
    fundamental_llm_timeout_seconds: int = int(os.getenv("FUNDAMENTAL_LLM_TIMEOUT_SECONDS", "12"))

    # Gemini shortcut variables (used as fallback when FUNDAMENTAL_LLM_API_KEY is empty)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
    gemini_openai_base_url: str = os.getenv("GEMINI_OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")

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
    donchian_session_filter: bool = os.getenv("DONCHIAN_SESSION_FILTER", "false").strip().lower() in {"1", "true", "yes", "y"}
    donchian_sessions: str = os.getenv("DONCHIAN_SESSIONS", "london,ny")

    # Turtle ATR breakout strategy (eventless-capable)
    turtle_lookback_seconds: int = int(os.getenv("TURTLE_LOOKBACK_SECONDS", "3600"))
    turtle_breakout_buffer_pips: float = float(os.getenv("TURTLE_BREAKOUT_BUFFER_PIPS", "0.10"))
    turtle_min_channel_pips: float = float(os.getenv("TURTLE_MIN_CHANNEL_PIPS", "0.02"))
    turtle_confirm_ticks: int = int(os.getenv("TURTLE_CONFIRM_TICKS", "1"))
    turtle_atr_period_ticks: int = int(os.getenv("TURTLE_ATR_PERIOD_TICKS", "120"))
    turtle_min_atr_pips: float = float(os.getenv("TURTLE_MIN_ATR_PIPS", "0.05"))
    turtle_trigger_quantile: float = float(os.getenv("TURTLE_TRIGGER_QUANTILE", "0.85"))
    turtle_trend_ema_span: int = int(os.getenv("TURTLE_TREND_EMA_SPAN", "180"))
    turtle_max_extension_atr: float = float(os.getenv("TURTLE_MAX_EXTENSION_ATR", "2.50"))
    turtle_signal_cooldown_seconds: int = int(os.getenv("TURTLE_SIGNAL_COOLDOWN_SECONDS", "240"))

    # EMA + RSI trend strategy (eventless-capable)
    ema_fast_span: int = int(os.getenv("EMA_FAST_SPAN", "21"))
    ema_slow_span: int = int(os.getenv("EMA_SLOW_SPAN", "55"))
    ema_rsi_period: int = int(os.getenv("EMA_RSI_PERIOD", "14"))
    ema_rsi_buy_level: float = float(os.getenv("EMA_RSI_BUY_LEVEL", "56"))
    ema_rsi_sell_level: float = float(os.getenv("EMA_RSI_SELL_LEVEL", "44"))
    ema_min_separation_pips: float = float(os.getenv("EMA_MIN_SEPARATION_PIPS", "0.20"))
    ema_momentum_lookback_ticks: int = int(os.getenv("EMA_MOMENTUM_LOOKBACK_TICKS", "20"))
    ema_min_momentum_pips: float = float(os.getenv("EMA_MIN_MOMENTUM_PIPS", "0.25"))
    ema_vol_period: int = int(os.getenv("EMA_VOL_PERIOD", "40"))
    ema_min_vol_pips: float = float(os.getenv("EMA_MIN_VOL_PIPS", "0.05"))
    ema_signal_cooldown_seconds: int = int(os.getenv("EMA_SIGNAL_COOLDOWN_SECONDS", "180"))

    # Agentic strategy (multi-agent orchestration)
    agent_manage_all_strategies: bool = os.getenv("AGENT_MANAGE_ALL_STRATEGIES", "true").strip().lower() in {"1", "true", "yes", "y"}
    agentic_learning_rate: float = float(os.getenv("AGENTIC_LEARNING_RATE", "0.20"))
    agentic_explore_prob: float = float(os.getenv("AGENTIC_EXPLORE_PROB", "0.10"))
    agentic_min_confidence: float = float(os.getenv("AGENTIC_MIN_CONFIDENCE", "0.56"))
    agentic_reward_horizon_seconds: int = int(os.getenv("AGENTIC_REWARD_HORIZON_SECONDS", "45"))
    agentic_reward_target_pips: float = float(os.getenv("AGENTIC_REWARD_TARGET_PIPS", "1.20"))
    agentic_state_path: str = os.getenv("AGENTIC_STATE_PATH", "models/agentic_state.json")
    agentic_signal_cooldown_seconds: int = int(os.getenv("AGENTIC_SIGNAL_COOLDOWN_SECONDS", "180"))

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

    live_calendar_refresh_seconds: int = int(os.getenv("LIVE_CALENDAR_REFRESH_SECONDS", "120"))
    live_loop_sleep_seconds: int = int(os.getenv("LIVE_LOOP_SLEEP_SECONDS", "1"))
    eventless_eval_seconds: int = int(os.getenv("EVENTLESS_EVAL_SECONDS", "20"))
    live_activity_csv: str = os.getenv("LIVE_ACTIVITY_CSV", "data/live_activity.csv")

    # --- Risk / exits (ATR-based preferred) ---
    # Riesgo por trade (fracción del capital)
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.005"))

    # Stop loss: multiplicador de ATR (ej. 1.5 × ATR)
    sl_atr_multiplier: float = float(os.getenv("SL_ATR_MULTIPLIER", "1.5"))

    # Take profit: ratio respecto al SL (TP = TP_SL_RATIO × SL)
    tp_sl_ratio: float = float(os.getenv("TP_SL_RATIO", "2.0"))

    # Trailing
    trail_activation_r: float = float(os.getenv("TRAIL_ACTIVATION_R", "1.0"))
    trail_atr_multiplier: float = float(os.getenv("TRAIL_ATR_MULTIPLIER", "0.75"))
    trail_step_atr: float = float(os.getenv("TRAIL_STEP_ATR", "0.5"))

    # Presets por horizonte (opcionales)
    scalp_sl_atr_multiplier: float = float(os.getenv("SCALP_SL_ATR_MULTIPLIER", "0.5"))
    scalp_tp_sl_ratio: float = float(os.getenv("SCALP_TP_SL_RATIO", "1.0"))
    intraday_sl_atr_multiplier: float = float(os.getenv("INTRADAY_SL_ATR_MULTIPLIER", "1.0"))
    intraday_tp_sl_ratio: float = float(os.getenv("INTRADAY_TP_SL_RATIO", "1.5"))
    swing_sl_atr_multiplier: float = float(os.getenv("SWING_SL_ATR_MULTIPLIER", "3.0"))
    swing_tp_sl_ratio: float = float(os.getenv("SWING_TP_SL_RATIO", "2.0"))

    @property
    def local_tz(self) -> timezone:
        return timezone(timedelta(hours=self.utc_offset_hours))


settings = Settings()
