from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model_registry import list_snapshots, restore_snapshot, snapshot_current_models
from src.config import settings
from src.ui.analytics import (
    build_monitor_source as build_monitor_source_impl,
    render_paper_trade_charts as render_paper_trade_charts_impl,
    render_walkforward_charts as render_walkforward_charts_impl,
)
from src.ui.common import (
    load_csv as load_csv_impl,
    parse_datetime_utc as parse_datetime_utc_impl,
    read_if_exists as read_if_exists_impl,
    run_module as run_module_impl,
    run_script as run_script_impl,
)
from src.ui.env import (
    load_env as load_env_impl,
    parse_bool as parse_bool_impl,
    parse_float as parse_float_impl,
    parse_int as parse_int_impl,
    save_env as save_env_impl,
)
from src.ui.history import (
    enrich_trade_history_with_results as enrich_trade_history_with_results_impl,
    render_trade_history_tab as render_trade_history_tab_impl,
)
from src.ui.live_ops import (
    get_live_bot_pid as get_live_bot_pid_impl,
    get_next_trigger_info as get_next_trigger_info_impl,
    load_live_mt5_trades as load_live_mt5_trades_impl,
    render_live_status_panel as render_live_status_panel_impl,
    start_live_bot_process as start_live_bot_process_impl,
    stop_live_bot_process as stop_live_bot_process_impl,
    verify_mt5_connection as verify_mt5_connection_impl,
)
from src.ui.theme import apply_modern_theme as apply_modern_theme_impl, section_card as section_card_impl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
LIVE_PID_PATH = PROJECT_ROOT / "logs/live_bot.pid"


def apply_modern_theme(theme_mode: str = "light") -> None:
    apply_modern_theme_impl(theme_mode)


def section_card(title: str, subtitle: str = "") -> None:
    section_card_impl(title, subtitle)


def load_env() -> dict[str, str]:
    return load_env_impl(ENV_PATH)


def save_env(values: dict[str, str]) -> None:
    save_env_impl(ENV_PATH, values)


def run_module(module: str, extra_env: dict[str, str] | None = None) -> tuple[int, str]:
    return run_module_impl(PROJECT_ROOT, module, extra_env)


def run_script(script_rel_path: str, args: list[str] | None = None, extra_env: dict[str, str] | None = None) -> tuple[int, str]:
    return run_script_impl(PROJECT_ROOT, script_rel_path, args, extra_env)


def read_if_exists(path: Path, n: int = 200) -> str:
    return read_if_exists_impl(path, n)


def load_csv(path: Path) -> pd.DataFrame:
    return load_csv_impl(path)


def parse_datetime_utc(values: pd.Series | list | np.ndarray) -> pd.Series:
    return parse_datetime_utc_impl(values)


def load_live_mt5_trades(symbol: str, history_days: int) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    return load_live_mt5_trades_impl(symbol, history_days)


def build_monitor_source(environment: str, history_days: int) -> tuple[pd.DataFrame, str | None, str]:
    return build_monitor_source_impl(PROJECT_ROOT, environment, history_days)


def parse_int(value: str | None, default: int) -> int:
    return parse_int_impl(value, default)


def parse_float(value: str | None, default: float) -> float:
    return parse_float_impl(value, default)


def parse_bool(value: str | None, default: bool) -> bool:
    return parse_bool_impl(value, default)


def detect_runtime_preset(env_vals: dict[str, str]) -> str:
    strict = str(env_vals.get("AGENT_RUNTIME_STRICT", "true")).strip().lower()
    min_conf = str(env_vals.get("AGENT_RUNTIME_MIN_CONFIDENCE", "0.55")).strip()
    max_spread = str(env_vals.get("AGENT_RUNTIME_MAX_SPREAD_PIPS", "2.2")).strip()
    driven_floor = str(env_vals.get("DRIVEN_DECISION_THRESHOLD_FLOOR", "0.55")).strip()
    llm_mode = str(env_vals.get("DRIVEN_LLM_MODE", "confirm")).strip().lower()
    explore = str(env_vals.get("DRIVEN_EXPLORE_PROB", "0.05")).strip()

    if strict == "true" and min_conf == "0.62" and max_spread == "1.60" and driven_floor == "0.62" and llm_mode == "confirm" and explore == "0.03":
        return "Conservador"
    if strict == "true" and min_conf == "0.55" and max_spread == "2.20" and driven_floor == "0.55" and llm_mode == "confirm" and explore == "0.05":
        return "Balanceado"
    if strict == "false" and min_conf == "0.52" and max_spread == "2.80" and driven_floor == "0.54" and llm_mode == "blend" and explore == "0.12":
        return "Agresivo"
    return "Custom"


def get_next_trigger_info(
    *,
    events_csv_path: Path,
    strategy_mode: str,
    seconds_before_event: int,
    event_min_importance: int,
    utc_offset_hours: float,
    donchian_session_filter: bool,
    donchian_sessions: str,
) -> dict[str, object]:
    return get_next_trigger_info_impl(
        events_csv_path=events_csv_path,
        strategy_mode=strategy_mode,
        seconds_before_event=seconds_before_event,
        event_min_importance=event_min_importance,
        utc_offset_hours=utc_offset_hours,
        donchian_session_filter=donchian_session_filter,
        donchian_sessions=donchian_sessions,
    )


def get_live_bot_pid() -> int | None:
    return get_live_bot_pid_impl(LIVE_PID_PATH)


def start_live_bot_process() -> tuple[bool, str]:
    return start_live_bot_process_impl(PROJECT_ROOT, LIVE_PID_PATH)


def stop_live_bot_process() -> tuple[bool, str]:
    return stop_live_bot_process_impl(LIVE_PID_PATH)


def verify_mt5_connection() -> tuple[bool, str]:
    return verify_mt5_connection_impl()


def render_walkforward_charts(report_path: Path) -> None:
    render_walkforward_charts_impl(report_path)


def render_paper_trade_charts(
    paper_source: Path | pd.DataFrame,
    widget_prefix: str,
    min_signals_sem: int,
    min_edge_sem: float,
    min_conf_sem: float,
    utc_offset_hours: float,
    ny_latam_preset_default: bool,
) -> None:
    render_paper_trade_charts_impl(
        paper_source,
        widget_prefix,
        min_signals_sem,
        min_edge_sem,
        min_conf_sem,
        utc_offset_hours,
        ny_latam_preset_default,
    )


def enrich_trade_history_with_results(trades: pd.DataFrame, market_path: Path) -> pd.DataFrame:
    return enrich_trade_history_with_results_impl(trades, market_path)

def render_live_status_panel(
    live_activity_path: Path,
    daily_report_path: Path,
    *,
    strategy_mode: str,
    events_csv: str,
    seconds_before_event: int,
    event_min_importance: int,
    utc_offset_hours: float,
    donchian_session_filter: bool,
    donchian_sessions: str,
) -> None:
    render_live_status_panel_impl(
        PROJECT_ROOT,
        ENV_PATH,
        LIVE_PID_PATH,
        live_activity_path,
        daily_report_path,
        strategy_mode=strategy_mode,
        events_csv=events_csv,
        seconds_before_event=seconds_before_event,
        event_min_importance=event_min_importance,
        utc_offset_hours=utc_offset_hours,
        donchian_session_filter=donchian_session_filter,
        donchian_sessions=donchian_sessions,
    )


def render_trade_history_tab() -> None:
    render_trade_history_tab_impl(PROJECT_ROOT, ENV_PATH)


def main() -> None:
    st.set_page_config(page_title="Economic AE Control Center", layout="wide")
    env_vals = load_env()
    default_dark = parse_bool(env_vals.get("UI_DARK_MODE"), False)
    if "ui_dark_mode" not in st.session_state:
        st.session_state["ui_dark_mode"] = default_dark

    apply_modern_theme("dark" if st.session_state.get("ui_dark_mode") else "light")

    st.markdown('<div id="theme-switch-anchor"></div>', unsafe_allow_html=True)
    theme_badge_class = "dark" if st.session_state.get("ui_dark_mode") else "light"
    theme_badge_icon = "🌙" if st.session_state.get("ui_dark_mode") else "🌞"
    st.markdown(
        f'<div class="theme-chip {theme_badge_class}"><span class="icon">{theme_badge_icon}</span><span>Tema</span></div>',
        unsafe_allow_html=True,
    )
    st.toggle(
        "Tema",
        key="ui_dark_mode",
        help="Alterna entre tema claro y oscuro.",
        label_visibility="collapsed",
    )
    current_dark = bool(st.session_state.get("ui_dark_mode"))
    persisted_dark = parse_bool(env_vals.get("UI_DARK_MODE"), False)
    if current_dark != persisted_dark:
        env_vals["UI_DARK_MODE"] = "true" if current_dark else "false"
        save_env(env_vals)

    sem_min_signals = parse_int(env_vals.get("SEM_MIN_SIGNALS"), 8)
    sem_min_edge = parse_float(env_vals.get("SEM_MIN_EDGE"), 0.58)
    sem_min_conf = parse_float(env_vals.get("SEM_MIN_CONF"), 0.60)
    utc_offset_hours = parse_float(env_vals.get("UTC_OFFSET_HOURS"), -5.0)
    ny_latam_preset_default = parse_bool(env_vals.get("NY_LATAM_PRESET_DEFAULT"), False)
    paper_mode = parse_bool(env_vals.get("PAPER_TRADING"), settings.paper_trading)
    strategy_mode = (env_vals.get("STRATEGY") or getattr(settings, "strategy", "default") or "default").strip().lower()

    mode_class = "paper" if paper_mode else "live"
    mode_text = "MODO PAPER (PRUEBAS)" if paper_mode else "MODO LIVE (REAL)"
    st.markdown(
        f"""
        <div class="app-hero">
            <h1>Economic AE Control Center</h1>
            <p>Panel de operacion, monitoreo y control de estrategias para EURUSD.</p>
            <span class="mode-pill {mode_class}">{mode_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Modo de ejecución", "PAPER" if paper_mode else "LIVE")
    m2.metric("Símbolo", env_vals.get("SYMBOL", settings.symbol))
    m3.metric("Estrategia", strategy_mode)
    runtime_preset = detect_runtime_preset(env_vals)
    preset_icon = {
        "Conservador": "🛡",
        "Balanceado": "⚖",
        "Agresivo": "⚡",
        "Custom": "⚙",
    }.get(runtime_preset, "⚙")
    preset_style = {
        "Conservador": "background:#e6f7ed;color:#1b5e20;border:1px solid #b7e1c3;",
        "Balanceado": "background:#e8f1ff;color:#0d47a1;border:1px solid #bfd5ff;",
        "Agresivo": "background:#fff3e0;color:#e65100;border:1px solid #ffd8a8;",
        "Custom": "background:#f2f4f7;color:#334155;border:1px solid #d7dee7;",
    }.get(runtime_preset, "background:#f2f4f7;color:#334155;border:1px solid #d7dee7;")
    st.markdown(
        f"""
        <div style=\"margin-top:0.25rem;margin-bottom:0.5rem;\">
            <span style=\"display:inline-block;padding:0.28rem 0.62rem;border-radius:999px;font-weight:600;font-size:0.86rem;{preset_style}\">
                {preset_icon} Preset runtime activo: {runtime_preset}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if paper_mode:
        st.warning("Actualmente estás en PAPER mode. Cambia a LIVE en Configuración para operar real.")
    else:
        st.success("Actualmente estás en LIVE mode (producción real).")

    tab_overview, tab_agentic, tab_live, tab_history, tab_advanced = st.tabs(
        ["Resumen", "Control Agentic", "Operación LIVE", "Histórico Operaciones", "Avanzado"]
    )

    with tab_overview:
        events_path = Path(settings.events_csv)
        if not events_path.is_absolute():
            events_path = PROJECT_ROOT / events_path
        market_path = Path(settings.market_csv)
        if not market_path.is_absolute():
            market_path = PROJECT_ROOT / market_path

        c1, c2, c3 = st.columns(3)
        c1.metric("Events CSV", "OK" if events_path.exists() else "Missing")
        c2.metric("Market CSV", "OK" if market_path.exists() else "Missing")
        c3.metric("Models", "OK" if (PROJECT_ROOT / "models/metadata.json").exists() else "Missing")

        st.subheader("Estrategia activa")
        st.write(f"Modo actual: **{strategy_mode}**")
        if strategy_mode == "zscore":
            st.json(
                {
                    "Z_SCORE_LOOKBACK_SECONDS": parse_int(env_vals.get("Z_SCORE_LOOKBACK_SECONDS"), 300),
                    "Z_SCORE_THRESHOLD": parse_float(env_vals.get("Z_SCORE_THRESHOLD"), 0.7),
                    "Z_WEIGHT": parse_float(env_vals.get("Z_WEIGHT"), 1.0),
                    "Z_COMBINATION_MODE": env_vals.get("Z_COMBINATION_MODE", "weighted"),
                }
            )
        elif strategy_mode == "momentum":
            st.json(
                {
                    "MOMENTUM_LOOKBACK_SECONDS": parse_int(env_vals.get("MOMENTUM_LOOKBACK_SECONDS"), 300),
                    "MOMENTUM_THRESHOLD": parse_float(env_vals.get("MOMENTUM_THRESHOLD"), 0.0005),
                    "MOMENTUM_WEIGHT": parse_float(env_vals.get("MOMENTUM_WEIGHT"), 1.0),
                    "MOMENTUM_MODE": env_vals.get("MOMENTUM_MODE", "weighted"),
                }
            )
        elif strategy_mode == "donchian":
            st.json(
                {
                    "DONCHIAN_LOOKBACK_SECONDS": parse_int(env_vals.get("DONCHIAN_LOOKBACK_SECONDS"), 600),
                    "DONCHIAN_BREAKOUT_BUFFER_PIPS": parse_float(env_vals.get("DONCHIAN_BREAKOUT_BUFFER_PIPS"), 0.2),
                    "DONCHIAN_MIN_CHANNEL_PIPS": parse_float(env_vals.get("DONCHIAN_MIN_CHANNEL_PIPS"), 0.1),
                    "DONCHIAN_CONFIRM_TICKS": parse_int(env_vals.get("DONCHIAN_CONFIRM_TICKS"), 1),
                    "DONCHIAN_TRIGGER_QUANTILE": parse_float(env_vals.get("DONCHIAN_TRIGGER_QUANTILE"), 0.80),
                    "DONCHIAN_SESSION_FILTER": env_vals.get("DONCHIAN_SESSION_FILTER", "false"),
                    "DONCHIAN_SESSIONS": env_vals.get("DONCHIAN_SESSIONS", "london,ny"),
                }
            )
        elif strategy_mode == "donchian_nylondon":
            st.json(
                {
                    "DONCHIAN_LOOKBACK_SECONDS": parse_int(env_vals.get("DONCHIAN_LOOKBACK_SECONDS"), 600),
                    "DONCHIAN_BREAKOUT_BUFFER_PIPS": parse_float(env_vals.get("DONCHIAN_BREAKOUT_BUFFER_PIPS"), 0.2),
                    "DONCHIAN_MIN_CHANNEL_PIPS": parse_float(env_vals.get("DONCHIAN_MIN_CHANNEL_PIPS"), 0.1),
                    "DONCHIAN_CONFIRM_TICKS": parse_int(env_vals.get("DONCHIAN_CONFIRM_TICKS"), 1),
                    "DONCHIAN_TRIGGER_QUANTILE": parse_float(env_vals.get("DONCHIAN_TRIGGER_QUANTILE"), 0.80),
                    "DONCHIAN_SESSION_FILTER": "true",
                    "DONCHIAN_SESSIONS": "london,ny",
                }
            )
        elif strategy_mode in {"turtle_atr", "atr_breakout", "vol_breakout", "turtle_atr_breakout"}:
            st.json(
                {
                    "TURTLE_LOOKBACK_SECONDS": parse_int(env_vals.get("TURTLE_LOOKBACK_SECONDS"), 1200),
                    "TURTLE_BREAKOUT_BUFFER_PIPS": parse_float(env_vals.get("TURTLE_BREAKOUT_BUFFER_PIPS"), 0.30),
                    "TURTLE_MIN_CHANNEL_PIPS": parse_float(env_vals.get("TURTLE_MIN_CHANNEL_PIPS"), 1.20),
                    "TURTLE_CONFIRM_TICKS": parse_int(env_vals.get("TURTLE_CONFIRM_TICKS"), 2),
                    "TURTLE_ATR_PERIOD_TICKS": parse_int(env_vals.get("TURTLE_ATR_PERIOD_TICKS"), 120),
                    "TURTLE_MIN_ATR_PIPS": parse_float(env_vals.get("TURTLE_MIN_ATR_PIPS"), 0.08),
                    "TURTLE_TRIGGER_QUANTILE": parse_float(env_vals.get("TURTLE_TRIGGER_QUANTILE"), 0.85),
                    "TURTLE_TREND_EMA_SPAN": parse_int(env_vals.get("TURTLE_TREND_EMA_SPAN"), 180),
                    "TURTLE_MAX_EXTENSION_ATR": parse_float(env_vals.get("TURTLE_MAX_EXTENSION_ATR"), 2.50),
                    "TURTLE_SIGNAL_COOLDOWN_SECONDS": parse_int(env_vals.get("TURTLE_SIGNAL_COOLDOWN_SECONDS"), 240),
                    "EVENTLESS_EVAL_SECONDS": parse_int(env_vals.get("EVENTLESS_EVAL_SECONDS"), 20),
                }
            )
        elif strategy_mode in {"ema_rsi", "ema_rsi_trend", "ema_rsi_active", "crossover_rsi"}:
            st.json(
                {
                    "EMA_FAST_SPAN": parse_int(env_vals.get("EMA_FAST_SPAN"), 21),
                    "EMA_SLOW_SPAN": parse_int(env_vals.get("EMA_SLOW_SPAN"), 55),
                    "EMA_RSI_PERIOD": parse_int(env_vals.get("EMA_RSI_PERIOD"), 14),
                    "EMA_RSI_BUY_LEVEL": parse_float(env_vals.get("EMA_RSI_BUY_LEVEL"), 56.0),
                    "EMA_RSI_SELL_LEVEL": parse_float(env_vals.get("EMA_RSI_SELL_LEVEL"), 44.0),
                    "EMA_MIN_SEPARATION_PIPS": parse_float(env_vals.get("EMA_MIN_SEPARATION_PIPS"), 0.20),
                    "EMA_MOMENTUM_LOOKBACK_TICKS": parse_int(env_vals.get("EMA_MOMENTUM_LOOKBACK_TICKS"), 20),
                    "EMA_MIN_MOMENTUM_PIPS": parse_float(env_vals.get("EMA_MIN_MOMENTUM_PIPS"), 0.25),
                    "EMA_VOL_PERIOD": parse_int(env_vals.get("EMA_VOL_PERIOD"), 40),
                    "EMA_MIN_VOL_PIPS": parse_float(env_vals.get("EMA_MIN_VOL_PIPS"), 0.05),
                    "EVENTLESS_EVAL_SECONDS": parse_int(env_vals.get("EVENTLESS_EVAL_SECONDS"), 20),
                }
            )
        elif strategy_mode in {"agentic_hybrid", "agentic", "agentic_ai", "multi_agent"}:
            st.json(
                {
                    "AGENTIC_MODE": "multi-agent orchestration (EMA+RSI + Donchian)",
                    "AGENTIC_LEARNING_RATE": parse_float(env_vals.get("AGENTIC_LEARNING_RATE"), 0.20),
                    "AGENTIC_EXPLORE_PROB": parse_float(env_vals.get("AGENTIC_EXPLORE_PROB"), 0.10),
                    "AGENTIC_MIN_CONFIDENCE": parse_float(env_vals.get("AGENTIC_MIN_CONFIDENCE"), 0.56),
                    "AGENTIC_REWARD_HORIZON_SECONDS": parse_int(env_vals.get("AGENTIC_REWARD_HORIZON_SECONDS"), 45),
                    "AGENTIC_REWARD_TARGET_PIPS": parse_float(env_vals.get("AGENTIC_REWARD_TARGET_PIPS"), 1.20),
                    "AGENTIC_STATE_PATH": env_vals.get("AGENTIC_STATE_PATH", "models/agentic_state.json"),
                    "EVENTLESS_EVAL_SECONDS": parse_int(env_vals.get("EVENTLESS_EVAL_SECONDS"), 20),
                }
            )
        elif strategy_mode in {"driven_trading_agentic_system", "driven_agentic", "driven", "multi_strategy_driven"}:
            st.json(
                {
                    "DRIVEN_MODE": "meta-orchestrator with adaptive subagent weighting",
                    "DRIVEN_LEARNING_RATE": parse_float(env_vals.get("DRIVEN_LEARNING_RATE"), 0.15),
                    "DRIVEN_EXPLORE_PROB": parse_float(env_vals.get("DRIVEN_EXPLORE_PROB"), 0.05),
                    "DRIVEN_MIN_AGENT_CONFIDENCE": parse_float(env_vals.get("DRIVEN_MIN_AGENT_CONFIDENCE"), 0.55),
                    "DRIVEN_REWARD_HORIZON_SECONDS": parse_int(env_vals.get("DRIVEN_REWARD_HORIZON_SECONDS"), 75),
                    "DRIVEN_REWARD_TARGET_PIPS": parse_float(env_vals.get("DRIVEN_REWARD_TARGET_PIPS"), 1.4),
                    "DRIVEN_COST_PER_TRADE_PIPS": parse_float(env_vals.get("DRIVEN_COST_PER_TRADE_PIPS"), 0.25),
                    "DRIVEN_MAX_SPREAD_PIPS": parse_float(env_vals.get("DRIVEN_MAX_SPREAD_PIPS"), 2.2),
                    "DRIVEN_SIGNAL_COOLDOWN_SECONDS": parse_int(env_vals.get("DRIVEN_SIGNAL_COOLDOWN_SECONDS"), 120),
                    "DRIVEN_LLM_ENABLED": env_vals.get("DRIVEN_LLM_ENABLED", "true"),
                    "DRIVEN_LLM_MODE": env_vals.get("DRIVEN_LLM_MODE", "confirm"),
                    "DRIVEN_STATE_PATH": env_vals.get("DRIVEN_STATE_PATH", "models/driven_agentic_state.json"),
                    "EVENTLESS_EVAL_SECONDS": parse_int(env_vals.get("EVENTLESS_EVAL_SECONDS"), 20),
                }
            )
        elif strategy_mode in {"fundamental_llm", "fundamental", "macro_llm", "news_llm"}:
            st.json(
                {
                    "FUNDAMENTAL_NEWS_SOURCES": env_vals.get(
                        "FUNDAMENTAL_NEWS_SOURCES",
                        "https://www.investing.com/rss/news_25.rss,https://feeds.reuters.com/reuters/businessNews",
                    ),
                    "FUNDAMENTAL_NEWS_LOOKBACK_MINUTES": parse_int(env_vals.get("FUNDAMENTAL_NEWS_LOOKBACK_MINUTES"), 240),
                    "FUNDAMENTAL_MAX_HEADLINES": parse_int(env_vals.get("FUNDAMENTAL_MAX_HEADLINES"), 30),
                    "FUNDAMENTAL_MIN_CONFIDENCE": parse_float(env_vals.get("FUNDAMENTAL_MIN_CONFIDENCE"), 0.60),
                    "FUNDAMENTAL_SIGNAL_COOLDOWN_SECONDS": parse_int(env_vals.get("FUNDAMENTAL_SIGNAL_COOLDOWN_SECONDS"), 300),
                    "FUNDAMENTAL_LLM_MODEL": env_vals.get("FUNDAMENTAL_LLM_MODEL", "gpt-4o-mini"),
                    "FUNDAMENTAL_LLM_API_BASE_URL": env_vals.get("FUNDAMENTAL_LLM_API_BASE_URL", "https://api.openai.com/v1"),
                    "EVENTLESS_EVAL_SECONDS": parse_int(env_vals.get("EVENTLESS_EVAL_SECONDS"), 20),
                }
            )
        else:
            st.caption("Usando estrategia base de ensemble (tabular + LSTM cuando esté disponible).")

        st.subheader("Último summary de walk-forward")
        summary = read_if_exists(PROJECT_ROOT / "models/walkforward_summary.json")
        st.code(summary or "No disponible")

        render_walkforward_charts(PROJECT_ROOT / "models/walkforward_monthly_report.csv")

        st.subheader("Monitoreo de ejecución")
        monitor_env = st.toggle(
            "Usar ambiente LIVE real (MT5)",
            value=False,
            key="overview_monitor_live_toggle",
            help="Desactivado: usa PAPER (data/paper_trades.csv). Activado: usa deals y posiciones de MT5.",
        )
        monitor_days = st.slider(
            "Ventana LIVE (días)",
            min_value=1,
            max_value=30,
            value=7,
            step=1,
            key="overview_monitor_live_days",
            disabled=not monitor_env,
        )
        monitor_df, monitor_error, monitor_label = build_monitor_source(
            environment="LIVE" if monitor_env else "PAPER",
            history_days=int(monitor_days),
        )
        if monitor_env and not monitor_df.empty and "entry_label" in monitor_df.columns:
            live_scope = st.selectbox(
                "Filtro LIVE",
                options=["Todas", "Solo aperturas", "Solo cierres"],
                index=0,
                key="overview_monitor_live_scope",
            )
            entry = monitor_df["entry_label"].astype(str).str.upper()
            open_mask = entry.isin(["OPEN", "OPEN_POSITION", "REVERSE"])
            close_mask = entry.isin(["CLOSE", "CLOSE_BY", "REVERSE"])
            if live_scope == "Solo aperturas":
                monitor_df = monitor_df[open_mask].copy()
            elif live_scope == "Solo cierres":
                monitor_df = monitor_df[close_mask].copy()

        st.caption(f"Vista analítica sobre los registros disponibles en {monitor_label}")
        if monitor_error:
            st.warning(f"No se pudo cargar fuente LIVE: {monitor_error}")
        render_paper_trade_charts(
            monitor_df,
            widget_prefix="overview",
            min_signals_sem=sem_min_signals,
            min_edge_sem=sem_min_edge,
            min_conf_sem=sem_min_conf,
            utc_offset_hours=utc_offset_hours,
            ny_latam_preset_default=ny_latam_preset_default,
        )

    with tab_agentic:
        st.subheader("Parámetros de trading y filtros")
        section_card(
            "Trading Base",
            "Define simbolo, modo de ejecucion y filtros generales de entrada para evitar operaciones de baja calidad.",
        )

        symbol = st.text_input("Par de divisas", value=env_vals.get("SYMBOL", "EURUSD"))
        min_imp = st.number_input("Importancia mínima de evento", min_value=1, max_value=3, value=int(env_vals.get("EVENT_MIN_IMPORTANCE", "2")))
        include_kw = st.text_input("Incluir eventos por keywords (coma)", value=env_vals.get("EVENT_INCLUDE_KEYWORDS", ""))
        exclude_kw = st.text_input("Excluir eventos por keywords (coma)", value=env_vals.get("EVENT_EXCLUDE_KEYWORDS", ""))
        threshold = st.text_input("Decision threshold", value=env_vals.get("DECISION_THRESHOLD", "0.60"))
        no_trade = st.text_input("No trade band", value=env_vals.get("NO_TRADE_BAND", "0.05"))
        paper = st.selectbox(
            "Modo de ejecución",
            options=["false", "true"],
            index=0 if env_vals.get("PAPER_TRADING", "true").lower() == "false" else 1,
            help="false = LIVE real, true = PAPER pruebas.",
        )
        strategy_options = [
            "default",
            "zscore",
            "momentum",
            "donchian",
            "donchian_nylondon",
            "turtle_atr",
            "ema_rsi_trend",
            "agentic_hybrid",
            "driven_trading_agentic_system",
            "fundamental_llm",
        ]
        strategy = st.selectbox(
            "Estrategia de decisión",
            options=strategy_options,
            index=strategy_options.index(strategy_mode) if strategy_mode in strategy_options else 0,
            help="Selecciona la lógica para generar señal de entrada antes de enviar órdenes.",
        )

        section_card(
            "Motores de Estrategia",
            "Ajusta sensibilidad de ZScore, Momentum, Donchian y EMA/RSI. Modifica solo una familia por vez.",
        )
        z_lookback = st.number_input(
            "Z_SCORE_LOOKBACK_SECONDS",
            min_value=30,
            max_value=7200,
            value=parse_int(env_vals.get("Z_SCORE_LOOKBACK_SECONDS"), 300),
            step=10,
        )
        z_threshold = st.number_input(
            "Z_SCORE_THRESHOLD",
            min_value=0.0,
            max_value=10.0,
            value=parse_float(env_vals.get("Z_SCORE_THRESHOLD"), 0.7),
            step=0.1,
        )
        z_weight = st.number_input(
            "Z_WEIGHT",
            min_value=0.0,
            max_value=10.0,
            value=parse_float(env_vals.get("Z_WEIGHT"), 1.0),
            step=0.1,
        )
        z_mode = st.selectbox(
            "Z_COMBINATION_MODE",
            options=["weighted", "conjunctive"],
            index=0 if (env_vals.get("Z_COMBINATION_MODE", "weighted") == "weighted") else 1,
        )

        mom_lookback = st.number_input(
            "MOMENTUM_LOOKBACK_SECONDS",
            min_value=30,
            max_value=7200,
            value=parse_int(env_vals.get("MOMENTUM_LOOKBACK_SECONDS"), 300),
            step=10,
        )
        mom_threshold = st.number_input(
            "MOMENTUM_THRESHOLD",
            min_value=0.0,
            max_value=0.05,
            value=parse_float(env_vals.get("MOMENTUM_THRESHOLD"), 0.0005),
            step=0.0001,
            format="%.4f",
        )
        mom_weight = st.number_input(
            "MOMENTUM_WEIGHT",
            min_value=0.0,
            max_value=10.0,
            value=parse_float(env_vals.get("MOMENTUM_WEIGHT"), 1.0),
            step=0.1,
        )
        mom_mode = st.selectbox(
            "MOMENTUM_MODE",
            options=["weighted", "conjunctive"],
            index=0 if (env_vals.get("MOMENTUM_MODE", "weighted") == "weighted") else 1,
        )

        don_lookback = st.number_input(
            "DONCHIAN_LOOKBACK_SECONDS",
            min_value=60,
            max_value=14400,
            value=parse_int(env_vals.get("DONCHIAN_LOOKBACK_SECONDS"), 600),
            step=30,
        )
        don_buffer = st.number_input(
            "DONCHIAN_BREAKOUT_BUFFER_PIPS",
            min_value=0.0,
            max_value=20.0,
            value=parse_float(env_vals.get("DONCHIAN_BREAKOUT_BUFFER_PIPS"), 0.2),
            step=0.1,
        )
        don_channel = st.number_input(
            "DONCHIAN_MIN_CHANNEL_PIPS",
            min_value=0.1,
            max_value=100.0,
            value=parse_float(env_vals.get("DONCHIAN_MIN_CHANNEL_PIPS"), 0.1),
            step=0.5,
        )
        don_confirm = st.number_input(
            "DONCHIAN_CONFIRM_TICKS",
            min_value=1,
            max_value=20,
            value=parse_int(env_vals.get("DONCHIAN_CONFIRM_TICKS"), 1),
            step=1,
        )
        don_quantile = st.number_input(
            "DONCHIAN_TRIGGER_QUANTILE",
            min_value=0.55,
            max_value=0.95,
            value=parse_float(env_vals.get("DONCHIAN_TRIGGER_QUANTILE"), 0.80),
            step=0.01,
            format="%.2f",
        )
        don_sessions = st.multiselect(
            "DONCHIAN_SESSIONS",
            options=["london", "ny"],
            default=[s for s in str(env_vals.get("DONCHIAN_SESSIONS", "london,ny")).split(",") if s in {"london", "ny"}] or ["london", "ny"],
            help="Solo aplica cuando DONCHIAN_SESSION_FILTER=true o estrategia donchian_nylondon.",
        )
        don_session_filter = st.selectbox(
            "DONCHIAN_SESSION_FILTER",
            options=["false", "true"],
            index=1 if parse_bool(env_vals.get("DONCHIAN_SESSION_FILTER"), False) else 0,
        )
        st.markdown("### Parámetros EMA + RSI (estrategia activa)")
        ema_fast_span = st.number_input(
            "EMA_FAST_SPAN",
            min_value=3,
            max_value=200,
            value=parse_int(env_vals.get("EMA_FAST_SPAN"), 21),
            step=1,
        )
        ema_slow_span = st.number_input(
            "EMA_SLOW_SPAN",
            min_value=5,
            max_value=400,
            value=parse_int(env_vals.get("EMA_SLOW_SPAN"), 55),
            step=1,
        )
        ema_rsi_period = st.number_input(
            "EMA_RSI_PERIOD",
            min_value=5,
            max_value=100,
            value=parse_int(env_vals.get("EMA_RSI_PERIOD"), 14),
            step=1,
        )
        ema_rsi_buy_level = st.number_input(
            "EMA_RSI_BUY_LEVEL",
            min_value=50.0,
            max_value=90.0,
            value=parse_float(env_vals.get("EMA_RSI_BUY_LEVEL"), 56.0),
            step=0.5,
        )
        ema_rsi_sell_level = st.number_input(
            "EMA_RSI_SELL_LEVEL",
            min_value=10.0,
            max_value=50.0,
            value=parse_float(env_vals.get("EMA_RSI_SELL_LEVEL"), 44.0),
            step=0.5,
        )
        ema_min_sep = st.number_input(
            "EMA_MIN_SEPARATION_PIPS",
            min_value=0.0,
            max_value=20.0,
            value=parse_float(env_vals.get("EMA_MIN_SEPARATION_PIPS"), 0.20),
            step=0.05,
            format="%.2f",
        )
        ema_mom_lb = st.number_input(
            "EMA_MOMENTUM_LOOKBACK_TICKS",
            min_value=3,
            max_value=500,
            value=parse_int(env_vals.get("EMA_MOMENTUM_LOOKBACK_TICKS"), 20),
            step=1,
        )
        ema_min_mom = st.number_input(
            "EMA_MIN_MOMENTUM_PIPS",
            min_value=0.0,
            max_value=20.0,
            value=parse_float(env_vals.get("EMA_MIN_MOMENTUM_PIPS"), 0.25),
            step=0.05,
            format="%.2f",
        )
        ema_vol_period = st.number_input(
            "EMA_VOL_PERIOD",
            min_value=8,
            max_value=500,
            value=parse_int(env_vals.get("EMA_VOL_PERIOD"), 40),
            step=1,
        )
        ema_min_vol = st.number_input(
            "EMA_MIN_VOL_PIPS",
            min_value=0.0,
            max_value=10.0,
            value=parse_float(env_vals.get("EMA_MIN_VOL_PIPS"), 0.05),
            step=0.01,
            format="%.2f",
        )
        eventless_eval_seconds = st.number_input(
            "EVENTLESS_EVAL_SECONDS",
            min_value=5,
            max_value=600,
            value=parse_int(env_vals.get("EVENTLESS_EVAL_SECONDS"), 20),
            step=1,
            help="Frecuencia de evaluación en segundos cuando la estrategia opera sin eventos.",
        )
        section_card(
            "Agentic IA",
            "Controla aprendizaje online, exploracion y horizonte de recompensa del orquestador multiagente.",
        )
        agentic_learning_rate = st.number_input(
            "AGENTIC_LEARNING_RATE",
            min_value=0.01,
            max_value=1.0,
            value=parse_float(env_vals.get("AGENTIC_LEARNING_RATE"), 0.20),
            step=0.01,
            format="%.2f",
        )
        agentic_explore_prob = st.number_input(
            "AGENTIC_EXPLORE_PROB",
            min_value=0.0,
            max_value=0.5,
            value=parse_float(env_vals.get("AGENTIC_EXPLORE_PROB"), 0.10),
            step=0.01,
            format="%.2f",
        )
        agentic_min_conf = st.number_input(
            "AGENTIC_MIN_CONFIDENCE",
            min_value=0.50,
            max_value=0.95,
            value=parse_float(env_vals.get("AGENTIC_MIN_CONFIDENCE"), 0.56),
            step=0.01,
            format="%.2f",
        )
        agentic_horizon = st.number_input(
            "AGENTIC_REWARD_HORIZON_SECONDS",
            min_value=10,
            max_value=600,
            value=parse_int(env_vals.get("AGENTIC_REWARD_HORIZON_SECONDS"), 45),
            step=1,
        )
        agentic_target_pips = st.number_input(
            "AGENTIC_REWARD_TARGET_PIPS",
            min_value=0.1,
            max_value=20.0,
            value=parse_float(env_vals.get("AGENTIC_REWARD_TARGET_PIPS"), 1.20),
            step=0.1,
            format="%.2f",
        )
        agentic_state_path = st.text_input(
            "AGENTIC_STATE_PATH",
            value=env_vals.get("AGENTIC_STATE_PATH", "models/agentic_state.json"),
            help="Archivo donde Agentic IA guarda pesos aprendidos entre reinicios.",
        )
        section_card(
            "Driven Trading Agentic System",
            "Meta-orquestador multiagente con aprendizaje online, desactivación adaptativa y capa LLM opcional.",
        )
        driven_state_path = st.text_input(
            "DRIVEN_STATE_PATH",
            value=env_vals.get("DRIVEN_STATE_PATH", "models/driven_agentic_state.json"),
            help="Estado persistente del meta-orquestador driven.",
        )
        driven_learning_rate = st.number_input(
            "DRIVEN_LEARNING_RATE",
            min_value=0.01,
            max_value=1.0,
            value=parse_float(env_vals.get("DRIVEN_LEARNING_RATE"), 0.15),
            step=0.01,
            format="%.2f",
        )
        driven_explore_prob = st.number_input(
            "DRIVEN_EXPLORE_PROB",
            min_value=0.0,
            max_value=0.5,
            value=parse_float(env_vals.get("DRIVEN_EXPLORE_PROB"), 0.05),
            step=0.01,
            format="%.2f",
        )
        driven_min_agent_conf = st.number_input(
            "DRIVEN_MIN_AGENT_CONFIDENCE",
            min_value=0.50,
            max_value=0.95,
            value=parse_float(env_vals.get("DRIVEN_MIN_AGENT_CONFIDENCE"), 0.55),
            step=0.01,
            format="%.2f",
        )
        driven_min_samples_disable = st.number_input(
            "DRIVEN_MIN_SAMPLES_DISABLE",
            min_value=5,
            max_value=300,
            value=parse_int(env_vals.get("DRIVEN_MIN_SAMPLES_DISABLE"), 14),
            step=1,
        )
        driven_disable_threshold = st.number_input(
            "DRIVEN_DISABLE_THRESHOLD",
            min_value=-0.95,
            max_value=0.20,
            value=parse_float(env_vals.get("DRIVEN_DISABLE_THRESHOLD"), -0.18),
            step=0.01,
            format="%.2f",
        )
        driven_disable_cooldown_minutes = st.number_input(
            "DRIVEN_DISABLE_COOLDOWN_MINUTES",
            min_value=5,
            max_value=720,
            value=parse_int(env_vals.get("DRIVEN_DISABLE_COOLDOWN_MINUTES"), 45),
            step=5,
        )
        driven_reward_horizon = st.number_input(
            "DRIVEN_REWARD_HORIZON_SECONDS",
            min_value=15,
            max_value=1800,
            value=parse_int(env_vals.get("DRIVEN_REWARD_HORIZON_SECONDS"), 75),
            step=1,
        )
        driven_reward_target_pips = st.number_input(
            "DRIVEN_REWARD_TARGET_PIPS",
            min_value=0.1,
            max_value=20.0,
            value=parse_float(env_vals.get("DRIVEN_REWARD_TARGET_PIPS"), 1.4),
            step=0.1,
            format="%.2f",
        )
        driven_cost_per_trade_pips = st.number_input(
            "DRIVEN_COST_PER_TRADE_PIPS",
            min_value=0.0,
            max_value=10.0,
            value=parse_float(env_vals.get("DRIVEN_COST_PER_TRADE_PIPS"), 0.25),
            step=0.01,
            format="%.2f",
        )
        driven_latency_penalty_pips = st.number_input(
            "DRIVEN_LATENCY_PENALTY_PIPS",
            min_value=0.0,
            max_value=10.0,
            value=parse_float(env_vals.get("DRIVEN_LATENCY_PENALTY_PIPS"), 0.08),
            step=0.01,
            format="%.2f",
        )
        driven_max_spread_pips = st.number_input(
            "DRIVEN_MAX_SPREAD_PIPS",
            min_value=0.0,
            max_value=20.0,
            value=parse_float(env_vals.get("DRIVEN_MAX_SPREAD_PIPS"), 2.2),
            step=0.1,
            format="%.2f",
        )
        driven_corr_window = st.number_input(
            "DRIVEN_CORR_WINDOW",
            min_value=20,
            max_value=1000,
            value=parse_int(env_vals.get("DRIVEN_CORR_WINDOW"), 80),
            step=5,
        )
        driven_corr_penalty = st.number_input(
            "DRIVEN_CORR_PENALTY",
            min_value=0.0,
            max_value=1.0,
            value=parse_float(env_vals.get("DRIVEN_CORR_PENALTY"), 0.35),
            step=0.01,
            format="%.2f",
        )
        driven_threshold_floor = st.number_input(
            "DRIVEN_DECISION_THRESHOLD_FLOOR",
            min_value=0.50,
            max_value=0.95,
            value=parse_float(env_vals.get("DRIVEN_DECISION_THRESHOLD_FLOOR"), 0.55),
            step=0.01,
            format="%.2f",
        )
        driven_threshold_cap = st.number_input(
            "DRIVEN_DECISION_THRESHOLD_CAP",
            min_value=float(driven_threshold_floor),
            max_value=0.98,
            value=max(parse_float(env_vals.get("DRIVEN_DECISION_THRESHOLD_CAP"), 0.82), float(driven_threshold_floor)),
            step=0.01,
            format="%.2f",
        )
        driven_signal_cooldown_seconds = st.number_input(
            "DRIVEN_SIGNAL_COOLDOWN_SECONDS",
            min_value=0,
            max_value=7200,
            value=parse_int(env_vals.get("DRIVEN_SIGNAL_COOLDOWN_SECONDS"), 120),
            step=5,
        )
        driven_llm_enabled = st.selectbox(
            "DRIVEN_LLM_ENABLED",
            options=["true", "false"],
            index=0 if parse_bool(env_vals.get("DRIVEN_LLM_ENABLED"), True) else 1,
        )
        driven_llm_mode = st.selectbox(
            "DRIVEN_LLM_MODE",
            options=["confirm", "blend", "off"],
            index=["confirm", "blend", "off"].index(env_vals.get("DRIVEN_LLM_MODE", "confirm")) if env_vals.get("DRIVEN_LLM_MODE", "confirm") in {"confirm", "blend", "off"} else 0,
            help="confirm = valida decisión final; blend = combina señal; off = desactiva LLM en el meta-orquestador.",
        )
        driven_llm_min_confidence = st.number_input(
            "DRIVEN_LLM_MIN_CONFIDENCE",
            min_value=0.50,
            max_value=0.98,
            value=parse_float(env_vals.get("DRIVEN_LLM_MIN_CONFIDENCE"), 0.62),
            step=0.01,
            format="%.2f",
        )
        driven_llm_veto_gap = st.number_input(
            "DRIVEN_LLM_VETO_GAP",
            min_value=0.0,
            max_value=0.50,
            value=parse_float(env_vals.get("DRIVEN_LLM_VETO_GAP"), 0.08),
            step=0.01,
            format="%.2f",
        )
        section_card(
            "Runtime Agents",
            "Activa o desactiva cada agente operativo del orquestador en tiempo real sin editar manualmente el .env.",
        )
        agent_runtime_enabled = st.selectbox(
            "AGENT_RUNTIME_ENABLED",
            options=["true", "false"],
            index=0 if parse_bool(env_vals.get("AGENT_RUNTIME_ENABLED"), True) else 1,
        )
        agent_runtime_strict = st.selectbox(
            "AGENT_RUNTIME_STRICT",
            options=["true", "false"],
            index=0 if parse_bool(env_vals.get("AGENT_RUNTIME_STRICT"), True) else 1,
            help="true = bloquea señales cuando un agente crítico falla validaciones; false = solo registra advertencias.",
        )
        agent_runtime_min_confidence = st.number_input(
            "AGENT_RUNTIME_MIN_CONFIDENCE",
            min_value=0.50,
            max_value=0.98,
            value=parse_float(env_vals.get("AGENT_RUNTIME_MIN_CONFIDENCE"), 0.55),
            step=0.01,
            format="%.2f",
        )
        agent_runtime_max_spread_pips = st.number_input(
            "AGENT_RUNTIME_MAX_SPREAD_PIPS",
            min_value=0.0,
            max_value=20.0,
            value=parse_float(env_vals.get("AGENT_RUNTIME_MAX_SPREAD_PIPS"), 2.2),
            step=0.1,
            format="%.2f",
        )

        st.markdown("**Toggles por agente**")
        col_a1, col_a2, col_a3 = st.columns(3)
        enable_strategy_architect_agent = col_a1.selectbox(
            "ENABLE_STRATEGY_ARCHITECT_AGENT",
            options=["true", "false"],
            index=0 if parse_bool(env_vals.get("ENABLE_STRATEGY_ARCHITECT_AGENT"), True) else 1,
        )
        enable_market_data_agent = col_a2.selectbox(
            "ENABLE_MARKET_DATA_AGENT",
            options=["true", "false"],
            index=0 if parse_bool(env_vals.get("ENABLE_MARKET_DATA_AGENT"), True) else 1,
        )
        enable_backtesting_agent = col_a3.selectbox(
            "ENABLE_BACKTESTING_AGENT",
            options=["true", "false"],
            index=0 if parse_bool(env_vals.get("ENABLE_BACKTESTING_AGENT"), True) else 1,
        )

        col_b1, col_b2, col_b3 = st.columns(3)
        enable_risk_manager_agent = col_b1.selectbox(
            "ENABLE_RISK_MANAGER_AGENT",
            options=["true", "false"],
            index=0 if parse_bool(env_vals.get("ENABLE_RISK_MANAGER_AGENT"), True) else 1,
        )
        enable_optimizer_agent = col_b2.selectbox(
            "ENABLE_OPTIMIZER_AGENT",
            options=["true", "false"],
            index=0 if parse_bool(env_vals.get("ENABLE_OPTIMIZER_AGENT"), True) else 1,
        )
        enable_llm_meta_agent = col_b3.selectbox(
            "ENABLE_LLM_META_AGENT",
            options=["true", "false"],
            index=0 if parse_bool(env_vals.get("ENABLE_LLM_META_AGENT"), True) else 1,
        )

        col_c1, col_c2, col_c3 = st.columns(3)
        enable_execution_agent = col_c1.selectbox(
            "ENABLE_EXECUTION_AGENT",
            options=["true", "false"],
            index=0 if parse_bool(env_vals.get("ENABLE_EXECUTION_AGENT"), True) else 1,
        )
        enable_monitoring_agent = col_c2.selectbox(
            "ENABLE_MONITORING_AGENT",
            options=["true", "false"],
            index=0 if parse_bool(env_vals.get("ENABLE_MONITORING_AGENT"), True) else 1,
        )
        enable_qa_agent = col_c3.selectbox(
            "ENABLE_QA_AGENT",
            options=["true", "false"],
            index=0 if parse_bool(env_vals.get("ENABLE_QA_AGENT"), True) else 1,
        )

        section_card(
            "Presets de Riesgo",
            "Aplica perfiles de un clic para runtime agents y umbrales driven. Guarda en .env y recarga la UI.",
        )
        pcol1, pcol2, pcol3 = st.columns(3)
        if pcol1.button("Aplicar Conservador"):
            env_vals.update(
                {
                    "AGENT_RUNTIME_ENABLED": "true",
                    "AGENT_RUNTIME_STRICT": "true",
                    "AGENT_RUNTIME_MIN_CONFIDENCE": "0.62",
                    "AGENT_RUNTIME_MAX_SPREAD_PIPS": "1.60",
                    "ENABLE_STRATEGY_ARCHITECT_AGENT": "true",
                    "ENABLE_MARKET_DATA_AGENT": "true",
                    "ENABLE_BACKTESTING_AGENT": "true",
                    "ENABLE_RISK_MANAGER_AGENT": "true",
                    "ENABLE_OPTIMIZER_AGENT": "true",
                    "ENABLE_LLM_META_AGENT": "true",
                    "ENABLE_EXECUTION_AGENT": "true",
                    "ENABLE_MONITORING_AGENT": "true",
                    "ENABLE_QA_AGENT": "true",
                    "DRIVEN_DECISION_THRESHOLD_FLOOR": "0.62",
                    "DRIVEN_DECISION_THRESHOLD_CAP": "0.88",
                    "DRIVEN_LLM_MODE": "confirm",
                    "DRIVEN_LLM_MIN_CONFIDENCE": "0.70",
                    "DRIVEN_MAX_SPREAD_PIPS": "1.80",
                    "DRIVEN_EXPLORE_PROB": "0.03",
                }
            )
            save_env(env_vals)
            st.success("Preset Conservador aplicado en .env")
            st.rerun()

        if pcol2.button("Aplicar Balanceado"):
            env_vals.update(
                {
                    "AGENT_RUNTIME_ENABLED": "true",
                    "AGENT_RUNTIME_STRICT": "true",
                    "AGENT_RUNTIME_MIN_CONFIDENCE": "0.55",
                    "AGENT_RUNTIME_MAX_SPREAD_PIPS": "2.20",
                    "ENABLE_STRATEGY_ARCHITECT_AGENT": "true",
                    "ENABLE_MARKET_DATA_AGENT": "true",
                    "ENABLE_BACKTESTING_AGENT": "true",
                    "ENABLE_RISK_MANAGER_AGENT": "true",
                    "ENABLE_OPTIMIZER_AGENT": "true",
                    "ENABLE_LLM_META_AGENT": "true",
                    "ENABLE_EXECUTION_AGENT": "true",
                    "ENABLE_MONITORING_AGENT": "true",
                    "ENABLE_QA_AGENT": "true",
                    "DRIVEN_DECISION_THRESHOLD_FLOOR": "0.55",
                    "DRIVEN_DECISION_THRESHOLD_CAP": "0.82",
                    "DRIVEN_LLM_MODE": "confirm",
                    "DRIVEN_LLM_MIN_CONFIDENCE": "0.62",
                    "DRIVEN_MAX_SPREAD_PIPS": "2.20",
                    "DRIVEN_EXPLORE_PROB": "0.05",
                }
            )
            save_env(env_vals)
            st.success("Preset Balanceado aplicado en .env")
            st.rerun()

        if pcol3.button("Aplicar Agresivo"):
            env_vals.update(
                {
                    "AGENT_RUNTIME_ENABLED": "true",
                    "AGENT_RUNTIME_STRICT": "false",
                    "AGENT_RUNTIME_MIN_CONFIDENCE": "0.52",
                    "AGENT_RUNTIME_MAX_SPREAD_PIPS": "2.80",
                    "ENABLE_STRATEGY_ARCHITECT_AGENT": "true",
                    "ENABLE_MARKET_DATA_AGENT": "true",
                    "ENABLE_BACKTESTING_AGENT": "true",
                    "ENABLE_RISK_MANAGER_AGENT": "true",
                    "ENABLE_OPTIMIZER_AGENT": "true",
                    "ENABLE_LLM_META_AGENT": "true",
                    "ENABLE_EXECUTION_AGENT": "true",
                    "ENABLE_MONITORING_AGENT": "true",
                    "ENABLE_QA_AGENT": "true",
                    "DRIVEN_DECISION_THRESHOLD_FLOOR": "0.54",
                    "DRIVEN_DECISION_THRESHOLD_CAP": "0.78",
                    "DRIVEN_LLM_MODE": "blend",
                    "DRIVEN_LLM_MIN_CONFIDENCE": "0.58",
                    "DRIVEN_MAX_SPREAD_PIPS": "2.80",
                    "DRIVEN_EXPLORE_PROB": "0.12",
                }
            )
            save_env(env_vals)
            st.success("Preset Agresivo aplicado en .env")
            st.rerun()
        section_card(
            "Fundamental + LLM",
            "Analiza titulares macro/economicos de fuentes RSS y consulta un LLM para decidir BUY/SELL/HOLD sobre cualquier simbolo (forex, commodities, indices/futuros, acciones).",
        )
        fundamental_news_sources = st.text_area(
            "FUNDAMENTAL_NEWS_SOURCES",
            value=env_vals.get(
                "FUNDAMENTAL_NEWS_SOURCES",
                "https://www.investing.com/rss/news_25.rss,https://feeds.reuters.com/reuters/businessNews,https://www.fxstreet.com/rss/news,https://feeds.marketwatch.com/marketwatch/topstories/",
            ),
            help="URLs RSS separadas por coma.",
        )
        fundamental_lookback_minutes = st.number_input(
            "FUNDAMENTAL_NEWS_LOOKBACK_MINUTES",
            min_value=30,
            max_value=1440,
            value=parse_int(env_vals.get("FUNDAMENTAL_NEWS_LOOKBACK_MINUTES"), 240),
            step=30,
        )
        fundamental_max_headlines = st.number_input(
            "FUNDAMENTAL_MAX_HEADLINES",
            min_value=5,
            max_value=200,
            value=parse_int(env_vals.get("FUNDAMENTAL_MAX_HEADLINES"), 30),
            step=5,
        )
        fundamental_min_conf = st.number_input(
            "FUNDAMENTAL_MIN_CONFIDENCE",
            min_value=0.50,
            max_value=0.95,
            value=parse_float(env_vals.get("FUNDAMENTAL_MIN_CONFIDENCE"), 0.60),
            step=0.01,
            format="%.2f",
        )
        fundamental_cooldown = st.number_input(
            "FUNDAMENTAL_SIGNAL_COOLDOWN_SECONDS",
            min_value=30,
            max_value=7200,
            value=parse_int(env_vals.get("FUNDAMENTAL_SIGNAL_COOLDOWN_SECONDS"), 300),
            step=30,
        )
        fundamental_fallback = st.selectbox(
            "FUNDAMENTAL_USE_HEURISTIC_FALLBACK",
            options=["true", "false"],
            index=0 if parse_bool(env_vals.get("FUNDAMENTAL_USE_HEURISTIC_FALLBACK"), True) else 1,
            help="Si el LLM no responde, usa fallback de sentimiento por keywords.",
        )
        fundamental_llm_base = st.text_input(
            "FUNDAMENTAL_LLM_API_BASE_URL",
            value=env_vals.get("FUNDAMENTAL_LLM_API_BASE_URL", "https://api.openai.com/v1"),
        )
        fundamental_llm_model = st.text_input(
            "FUNDAMENTAL_LLM_MODEL",
            value=env_vals.get("FUNDAMENTAL_LLM_MODEL", "gpt-4o-mini"),
        )
        fundamental_llm_key = st.text_input(
            "FUNDAMENTAL_LLM_API_KEY",
            value=env_vals.get("FUNDAMENTAL_LLM_API_KEY", ""),
            type="password",
        )
        label_mode = st.selectbox(
            "Modo de etiquetado",
            options=["sign", "quantile", "quantile_monthly"],
            index=["sign", "quantile", "quantile_monthly"].index(env_vals.get("DIRECTION_LABEL_MODE", "quantile_monthly")) if env_vals.get("DIRECTION_LABEL_MODE", "quantile_monthly") in ["sign", "quantile", "quantile_monthly"] else 2,
        )
        section_card(
            "Semaforo de Calidad",
            "Umbrales usados por la UI para marcar oportunidades en verde/amarillo/rojo.",
        )
        sem_min_signals_in = st.number_input("SEM_MIN_SIGNALS", min_value=1, max_value=1000, value=sem_min_signals, step=1)
        sem_min_edge_in = st.number_input("SEM_MIN_EDGE", min_value=0.0, max_value=1.0, value=float(sem_min_edge), step=0.01)
        sem_min_conf_in = st.number_input("SEM_MIN_CONF", min_value=0.0, max_value=1.0, value=float(sem_min_conf), step=0.01)
        section_card(
            "Riesgo y Costos",
            "Parametros para analisis monetario, costos netos y comportamiento por defecto de presets operativos.",
        )
        ny_latam_default_in = st.selectbox(
            "NY_LATAM_PRESET_DEFAULT",
            options=["false", "true"],
            index=1 if ny_latam_preset_default else 0,
            help="Define si el toggle NY/LATAM inicia activo al abrir la UI.",
        )
        risk_usd_in = st.number_input(
            "RISK_USD_PER_TRADE",
            min_value=1.0,
            max_value=100000.0,
            value=parse_float(env_vals.get("RISK_USD_PER_TRADE"), 25.0),
            step=1.0,
            help="Valor usado en la pestaña Histórico Operaciones para estimar balance monetario.",
        )
        comm_usd_in = st.number_input(
            "COMMISSION_USD_PER_TRADE",
            min_value=0.0,
            max_value=10000.0,
            value=parse_float(env_vals.get("COMMISSION_USD_PER_TRADE"), 0.0),
            step=0.1,
            help="Costo fijo por operación para balance neto en la pestaña Histórico Operaciones.",
        )
        spread_bps_in = st.number_input(
            "SPREAD_BPS_PER_TRADE",
            min_value=0.0,
            max_value=500.0,
            value=parse_float(env_vals.get("SPREAD_BPS_PER_TRADE"), 0.0),
            step=0.1,
            help="Costo variable por operación en bps sobre el riesgo por trade.",
        )
        dynamic_spread_in = st.selectbox(
            "DYNAMIC_SPREAD_COST",
            options=["true", "false"],
            index=0 if parse_bool(env_vals.get("DYNAMIC_SPREAD_COST"), True) else 1,
            help="Si true, usa spread real bid/ask cuando esté disponible en el histórico.",
        )

        if st.button("Guardar configuración"):
            env_vals["SYMBOL"] = symbol
            env_vals["EVENT_MIN_IMPORTANCE"] = str(min_imp)
            env_vals["EVENT_INCLUDE_KEYWORDS"] = include_kw
            env_vals["EVENT_EXCLUDE_KEYWORDS"] = exclude_kw
            env_vals["DECISION_THRESHOLD"] = threshold
            env_vals["NO_TRADE_BAND"] = no_trade
            env_vals["PAPER_TRADING"] = paper
            env_vals["STRATEGY"] = strategy
            env_vals["Z_SCORE_LOOKBACK_SECONDS"] = str(int(z_lookback))
            env_vals["Z_SCORE_THRESHOLD"] = f"{float(z_threshold):.4f}"
            env_vals["Z_WEIGHT"] = f"{float(z_weight):.4f}"
            env_vals["Z_COMBINATION_MODE"] = z_mode
            env_vals["MOMENTUM_LOOKBACK_SECONDS"] = str(int(mom_lookback))
            env_vals["MOMENTUM_THRESHOLD"] = f"{float(mom_threshold):.6f}"
            env_vals["MOMENTUM_WEIGHT"] = f"{float(mom_weight):.4f}"
            env_vals["MOMENTUM_MODE"] = mom_mode
            env_vals["DONCHIAN_LOOKBACK_SECONDS"] = str(int(don_lookback))
            env_vals["DONCHIAN_BREAKOUT_BUFFER_PIPS"] = f"{float(don_buffer):.2f}"
            env_vals["DONCHIAN_MIN_CHANNEL_PIPS"] = f"{float(don_channel):.2f}"
            env_vals["DONCHIAN_CONFIRM_TICKS"] = str(int(don_confirm))
            env_vals["DONCHIAN_TRIGGER_QUANTILE"] = f"{float(don_quantile):.2f}"
            env_vals["DONCHIAN_SESSION_FILTER"] = "true" if (strategy == "donchian_nylondon" or don_session_filter == "true") else "false"
            env_vals["DONCHIAN_SESSIONS"] = ",".join(don_sessions) if don_sessions else "london,ny"
            env_vals["EMA_FAST_SPAN"] = str(int(ema_fast_span))
            env_vals["EMA_SLOW_SPAN"] = str(int(ema_slow_span))
            env_vals["EMA_RSI_PERIOD"] = str(int(ema_rsi_period))
            env_vals["EMA_RSI_BUY_LEVEL"] = f"{float(ema_rsi_buy_level):.2f}"
            env_vals["EMA_RSI_SELL_LEVEL"] = f"{float(ema_rsi_sell_level):.2f}"
            env_vals["EMA_MIN_SEPARATION_PIPS"] = f"{float(ema_min_sep):.2f}"
            env_vals["EMA_MOMENTUM_LOOKBACK_TICKS"] = str(int(ema_mom_lb))
            env_vals["EMA_MIN_MOMENTUM_PIPS"] = f"{float(ema_min_mom):.2f}"
            env_vals["EMA_VOL_PERIOD"] = str(int(ema_vol_period))
            env_vals["EMA_MIN_VOL_PIPS"] = f"{float(ema_min_vol):.2f}"
            env_vals["EVENTLESS_EVAL_SECONDS"] = str(int(eventless_eval_seconds))
            env_vals["AGENTIC_LEARNING_RATE"] = f"{float(agentic_learning_rate):.2f}"
            env_vals["AGENTIC_EXPLORE_PROB"] = f"{float(agentic_explore_prob):.2f}"
            env_vals["AGENTIC_MIN_CONFIDENCE"] = f"{float(agentic_min_conf):.2f}"
            env_vals["AGENTIC_REWARD_HORIZON_SECONDS"] = str(int(agentic_horizon))
            env_vals["AGENTIC_REWARD_TARGET_PIPS"] = f"{float(agentic_target_pips):.2f}"
            env_vals["AGENTIC_STATE_PATH"] = str(agentic_state_path).strip() or "models/agentic_state.json"
            env_vals["DRIVEN_STATE_PATH"] = str(driven_state_path).strip() or "models/driven_agentic_state.json"
            env_vals["DRIVEN_LEARNING_RATE"] = f"{float(driven_learning_rate):.2f}"
            env_vals["DRIVEN_EXPLORE_PROB"] = f"{float(driven_explore_prob):.2f}"
            env_vals["DRIVEN_MIN_AGENT_CONFIDENCE"] = f"{float(driven_min_agent_conf):.2f}"
            env_vals["DRIVEN_MIN_SAMPLES_DISABLE"] = str(int(driven_min_samples_disable))
            env_vals["DRIVEN_DISABLE_THRESHOLD"] = f"{float(driven_disable_threshold):.2f}"
            env_vals["DRIVEN_DISABLE_COOLDOWN_MINUTES"] = str(int(driven_disable_cooldown_minutes))
            env_vals["DRIVEN_REWARD_HORIZON_SECONDS"] = str(int(driven_reward_horizon))
            env_vals["DRIVEN_REWARD_TARGET_PIPS"] = f"{float(driven_reward_target_pips):.2f}"
            env_vals["DRIVEN_COST_PER_TRADE_PIPS"] = f"{float(driven_cost_per_trade_pips):.2f}"
            env_vals["DRIVEN_LATENCY_PENALTY_PIPS"] = f"{float(driven_latency_penalty_pips):.2f}"
            env_vals["DRIVEN_MAX_SPREAD_PIPS"] = f"{float(driven_max_spread_pips):.2f}"
            env_vals["DRIVEN_CORR_WINDOW"] = str(int(driven_corr_window))
            env_vals["DRIVEN_CORR_PENALTY"] = f"{float(driven_corr_penalty):.2f}"
            env_vals["DRIVEN_DECISION_THRESHOLD_FLOOR"] = f"{float(driven_threshold_floor):.2f}"
            env_vals["DRIVEN_DECISION_THRESHOLD_CAP"] = f"{float(driven_threshold_cap):.2f}"
            env_vals["DRIVEN_SIGNAL_COOLDOWN_SECONDS"] = str(int(driven_signal_cooldown_seconds))
            env_vals["DRIVEN_LLM_ENABLED"] = driven_llm_enabled
            env_vals["DRIVEN_LLM_MODE"] = str(driven_llm_mode).strip().lower() or "confirm"
            env_vals["DRIVEN_LLM_MIN_CONFIDENCE"] = f"{float(driven_llm_min_confidence):.2f}"
            env_vals["DRIVEN_LLM_VETO_GAP"] = f"{float(driven_llm_veto_gap):.2f}"
            env_vals["AGENT_RUNTIME_ENABLED"] = agent_runtime_enabled
            env_vals["AGENT_RUNTIME_STRICT"] = agent_runtime_strict
            env_vals["AGENT_RUNTIME_MIN_CONFIDENCE"] = f"{float(agent_runtime_min_confidence):.2f}"
            env_vals["AGENT_RUNTIME_MAX_SPREAD_PIPS"] = f"{float(agent_runtime_max_spread_pips):.2f}"
            env_vals["ENABLE_STRATEGY_ARCHITECT_AGENT"] = enable_strategy_architect_agent
            env_vals["ENABLE_MARKET_DATA_AGENT"] = enable_market_data_agent
            env_vals["ENABLE_BACKTESTING_AGENT"] = enable_backtesting_agent
            env_vals["ENABLE_RISK_MANAGER_AGENT"] = enable_risk_manager_agent
            env_vals["ENABLE_OPTIMIZER_AGENT"] = enable_optimizer_agent
            env_vals["ENABLE_LLM_META_AGENT"] = enable_llm_meta_agent
            env_vals["ENABLE_EXECUTION_AGENT"] = enable_execution_agent
            env_vals["ENABLE_MONITORING_AGENT"] = enable_monitoring_agent
            env_vals["ENABLE_QA_AGENT"] = enable_qa_agent
            env_vals["FUNDAMENTAL_NEWS_SOURCES"] = str(fundamental_news_sources).replace("\n", ",").strip()
            env_vals["FUNDAMENTAL_NEWS_LOOKBACK_MINUTES"] = str(int(fundamental_lookback_minutes))
            env_vals["FUNDAMENTAL_MAX_HEADLINES"] = str(int(fundamental_max_headlines))
            env_vals["FUNDAMENTAL_MIN_CONFIDENCE"] = f"{float(fundamental_min_conf):.2f}"
            env_vals["FUNDAMENTAL_SIGNAL_COOLDOWN_SECONDS"] = str(int(fundamental_cooldown))
            env_vals["FUNDAMENTAL_USE_HEURISTIC_FALLBACK"] = fundamental_fallback
            env_vals["FUNDAMENTAL_LLM_API_BASE_URL"] = str(fundamental_llm_base).strip()
            env_vals["FUNDAMENTAL_LLM_MODEL"] = str(fundamental_llm_model).strip() or "gpt-4o-mini"
            env_vals["FUNDAMENTAL_LLM_API_KEY"] = str(fundamental_llm_key).strip()
            env_vals["DIRECTION_LABEL_MODE"] = label_mode
            env_vals["SEM_MIN_SIGNALS"] = str(int(sem_min_signals_in))
            env_vals["SEM_MIN_EDGE"] = f"{float(sem_min_edge_in):.4f}"
            env_vals["SEM_MIN_CONF"] = f"{float(sem_min_conf_in):.4f}"
            env_vals["NY_LATAM_PRESET_DEFAULT"] = ny_latam_default_in
            env_vals["RISK_USD_PER_TRADE"] = f"{float(risk_usd_in):.2f}"
            env_vals["COMMISSION_USD_PER_TRADE"] = f"{float(comm_usd_in):.2f}"
            env_vals["SPREAD_BPS_PER_TRADE"] = f"{float(spread_bps_in):.2f}"
            env_vals["DYNAMIC_SPREAD_COST"] = dynamic_spread_in
            save_env(env_vals)
            st.success("Configuración guardada en .env")

    with tab_advanced:
        st.subheader("Herramientas avanzadas")
        st.caption("Estas acciones se mantienen disponibles, pero se ocultan del flujo principal agentic.")

        with st.expander("Datos", expanded=False):
            st.subheader("Recolección y diagnóstico de datos")

            if st.button("Ejecutar bootstrap"):
                code, out = run_module("src.bootstrap")
                st.code(out)
                st.info(f"Exit code: {code}")

            col_a, col_b = st.columns(2)
            if col_a.button("Recolectar data entrenamiento"):
                code, out = run_module("src.data_collection")
                st.code(out)
                st.info(f"Exit code: {code}")

            if col_b.button("Preparar dataset mensual largo"):
                code, out = run_module("src.prepare_monthly_dataset")
                st.code(out)
                st.info(f"Exit code: {code}")

            if st.button("Diagnóstico por mes"):
                code, out = run_module("src.dataset_diagnostics")
                st.code(out)
                st.info(f"Exit code: {code}")
                diag_path = PROJECT_ROOT / "models/dataset_monthly_diagnostics.csv"
                if diag_path.exists():
                    st.dataframe(pd.read_csv(diag_path).head(100), use_container_width=True)

        with st.expander("Entrenamiento", expanded=False):
            st.subheader("Entrenar, evaluar y gestionar históricos de modelos")

            c1, c2 = st.columns(2)
            if c1.button("Entrenar modelos"):
                code, out = run_module("src.train")
                st.code(out)
                st.info(f"Exit code: {code}")

            if c2.button("Evaluar modelos"):
                code, out = run_module("src.evaluate")
                st.code(out)
                st.info(f"Exit code: {code}")

            st.markdown("### Snapshots de modelos")
            snap_name = st.text_input("Nombre snapshot (opcional)", value="")
            if st.button("Guardar snapshot actual"):
                try:
                    name = snapshot_current_models(name=snap_name if snap_name.strip() else None)
                    st.success(f"Snapshot guardado: {name}")
                except Exception as ex:
                    st.error(str(ex))

            snaps = list_snapshots()
            selected = st.selectbox("Restaurar snapshot", options=[""] + snaps)
            if st.button("Restaurar snapshot seleccionado"):
                if not selected:
                    st.warning("Selecciona un snapshot")
                else:
                    try:
                        restore_snapshot(selected)
                        st.success(f"Snapshot restaurado: {selected}")
                    except Exception as ex:
                        st.error(str(ex))

        with st.expander("Backtest", expanded=False):
            st.subheader("Backtesting")

            strict = st.selectbox("Validación mensual estricta", options=["true", "false"], index=0)
            events_csv = st.text_input("EVENTS_CSV para backtest", value=env_vals.get("EVENTS_CSV", "data/events.csv"))
            market_csv = st.text_input("MARKET_CSV para backtest", value=env_vals.get("MARKET_CSV", "data/market_ticks.csv"))

            if st.button("Ejecutar walk-forward backtest"):
                code, out = run_module(
                    "src.walkforward_backtest",
                    extra_env={
                        "STRICT_MONTHLY_VALIDATION": strict,
                        "EVENTS_CSV": events_csv,
                        "MARKET_CSV": market_csv,
                    },
                )
                st.code(out)
                st.info(f"Exit code: {code}")

                summary_path = PROJECT_ROOT / "models/walkforward_summary.json"
                report_path = PROJECT_ROOT / "models/walkforward_monthly_report.csv"
                if summary_path.exists():
                    st.code(summary_path.read_text(encoding="utf-8", errors="ignore"))
                if report_path.exists():
                    st.dataframe(pd.read_csv(report_path), use_container_width=True)

            st.markdown("### Visuales de backtest")
            render_walkforward_charts(PROJECT_ROOT / "models/walkforward_monthly_report.csv")

            st.markdown("### Optimización Donchian")
            don_events_csv = st.text_input(
                "Events CSV sweep Donchian",
                value=env_vals.get("EVENTS_CSV", "data/events.csv"),
                key="don_sweep_events_csv",
            )
            don_market_csv = st.text_input(
                "Market CSV sweep Donchian",
                value=env_vals.get("MARKET_CSV", "data/market_ticks.csv"),
                key="don_sweep_market_csv",
            )
            don_session_mode = st.selectbox(
                "Sweep sesión",
                options=["both", "on", "off"],
                index=0,
                help="both prueba normal y NY/Londres; on solo NY/Londres; off solo Donchian normal.",
                key="don_sweep_session_mode",
            )
            don_quick = st.checkbox(
                "Sweep rápido",
                value=True,
                help="Reduce combinaciones para terminar más rápido.",
                key="don_sweep_quick",
            )

            result_path = PROJECT_ROOT / "models/donchian_sweep_best.json"
            col_s1, col_s2 = st.columns(2)
            if col_s1.button("Ejecutar sweep Donchian"):
                code, out = run_script(
                    "scripts/sweep_donchian.py",
                    args=[
                        "--events-csv",
                        don_events_csv,
                        "--market-csv",
                        don_market_csv,
                        "--session-filter",
                        don_session_mode,
                        "--output",
                        str(result_path),
                    ] + (["--quick"] if don_quick else []),
                )
                st.code(out)
                st.info(f"Exit code: {code}")

            if result_path.exists():
                try:
                    sweep_data = json.loads(result_path.read_text(encoding="utf-8"))
                    st.json(sweep_data.get("best", {}))
                except Exception as ex:
                    st.warning(f"No se pudo leer resultado sweep: {ex}")
                    sweep_data = {}
            else:
                sweep_data = {}

            if col_s2.button("Aplicar mejor Donchian a .env"):
                if not sweep_data or "best" not in sweep_data:
                    st.warning("No hay resultado de sweep para aplicar. Ejecuta primero el sweep.")
                else:
                    best = sweep_data["best"]
                    env_vals["STRATEGY"] = "donchian_nylondon" if bool(best.get("session_filter", False)) else "donchian"
                    env_vals["DONCHIAN_LOOKBACK_SECONDS"] = str(int(best.get("lookback", 600)))
                    env_vals["DONCHIAN_BREAKOUT_BUFFER_PIPS"] = f"{float(best.get('buffer', 0.2)):.2f}"
                    env_vals["DONCHIAN_MIN_CHANNEL_PIPS"] = f"{float(best.get('min_channel', 1.0)):.2f}"
                    env_vals["DONCHIAN_CONFIRM_TICKS"] = str(int(best.get("confirm_ticks", 1)))
                    env_vals["DONCHIAN_TRIGGER_QUANTILE"] = f"{float(best.get('quantile', 0.80)):.2f}"
                    env_vals["DONCHIAN_SESSION_FILTER"] = "true" if bool(best.get("session_filter", False)) else "false"
                    env_vals["DONCHIAN_SESSIONS"] = str(best.get("sessions", "london,ny"))
                    save_env(env_vals)
                    st.success("Mejor configuración Donchian aplicada en .env")

    with tab_live:
        st.subheader("Operación real")
        st.write("Esta sección está orientada a producción. Verifica Modo de ejecución=LIVE antes de arrancar.")

        live_auto_refresh = st.toggle(
            "Auto-actualizar panel LIVE",
            value=bool(st.session_state.get("live_auto_refresh", False)),
            key="live_auto_refresh",
            help="Si está activo, refresca el panel automáticamente para actualizar countdown y estado.",
        )
        refresh_interval = st.slider(
            "Intervalo auto-refresh (segundos)",
            min_value=5,
            max_value=60,
            value=int(st.session_state.get("live_refresh_seconds", 10)),
            step=1,
            key="live_refresh_seconds",
            help="Frecuencia de actualización automática del panel LIVE.",
        )

        if st.button("Actualizar estado LIVE"):
            st.rerun()
        render_live_status_panel(
            PROJECT_ROOT / settings.live_activity_csv,
            PROJECT_ROOT / settings.model_dir / "daily_live_report.json",
            strategy_mode=strategy_mode,
            events_csv=env_vals.get("EVENTS_CSV", settings.events_csv),
            seconds_before_event=parse_int(env_vals.get("SECONDS_BEFORE_EVENT"), settings.seconds_before_event),
            event_min_importance=parse_int(env_vals.get("EVENT_MIN_IMPORTANCE"), settings.event_min_importance),
            utc_offset_hours=utc_offset_hours,
            donchian_session_filter=parse_bool(env_vals.get("DONCHIAN_SESSION_FILTER"), settings.donchian_session_filter),
            donchian_sessions=env_vals.get("DONCHIAN_SESSIONS", settings.donchian_sessions),
        )

        if live_auto_refresh:
            if paper_mode:
                st.caption("Auto-refresh en pausa: modo PAPER activo.")
            else:
                running_pid = get_live_bot_pid()
                if running_pid:
                    st.caption(f"Auto-refresh activo (PID {running_pid}): próxima actualización en {refresh_interval}s")
                    time.sleep(float(refresh_interval))
                    st.rerun()
                else:
                    st.caption("Auto-refresh en pausa: bot LIVE no está RUNNING.")

        st.caption(f"Modo actual detectado en configuración: {'PAPER' if paper_mode else 'LIVE'}")
        if paper_mode:
            st.warning("La configuración actual está en PAPER. Para habilitar LIVE debes cambiar PAPER_TRADING=false.")
            if st.button("Cambiar a LIVE ahora (guardar en .env)"):
                env_vals["PAPER_TRADING"] = "false"
                save_env(env_vals)
                st.success("Modo cambiado a LIVE en .env. Recargando panel...")
                st.rerun()

        if "live_mt5_last_ok" not in st.session_state:
            st.session_state["live_mt5_last_ok"] = False
        if "live_mt5_last_msg" not in st.session_state:
            st.session_state["live_mt5_last_msg"] = "Sin verificación en esta sesión."

        st.markdown("### Checklist pre-LIVE")
        critical_checks = [
            ("Modo de ejecución LIVE", not paper_mode),
            ("Modelos entrenados", (PROJECT_ROOT / "models/metadata.json").exists()),
        ]
        advisory_checks = [
            ("Calendario de eventos disponible", (PROJECT_ROOT / "data/events.csv").exists()),
            ("Datos de mercado disponibles (analítica UI)", (PROJECT_ROOT / settings.market_csv).exists()),
            ("Credenciales MT5 configuradas en .env", settings.mt5_login > 0 and bool(settings.mt5_server)),
            ("MT5 verificado en esta sesión", bool(st.session_state.get("live_mt5_last_ok", False))),
        ]

        st.markdown("#### Requisitos críticos (bloquean LIVE)")
        for label, ok in critical_checks:
            st.write(f"{'OK' if ok else 'PENDIENTE'} - {label}")

        st.markdown("#### Requisitos recomendados (no bloquean LIVE)")
        for label, ok in advisory_checks:
            st.write(f"{'OK' if ok else 'PENDIENTE'} - {label}")

        if st.button("Probar conexión MT5 ahora"):
            ok_mt5, msg_mt5 = verify_mt5_connection()
            st.session_state["live_mt5_last_ok"] = ok_mt5
            st.session_state["live_mt5_last_msg"] = msg_mt5
            (st.success if ok_mt5 else st.warning)(msg_mt5)

        st.caption(f"Última verificación MT5: {st.session_state.get('live_mt5_last_msg', 'Sin verificación')}")

        all_ready = all(flag for _, flag in critical_checks)
        if all_ready:
            st.success("Requisitos críticos completos. Listo para operación LIVE.")
        else:
            st.warning("Checklist crítico incompleto. Corrige los ítems PENDIENTE para habilitar LIVE.")

        st.markdown("### Armado de seguridad LIVE")
        arm_live = st.checkbox(
            "He verificado el checklist y autorizo operación LIVE",
            value=False,
            key="live_arm_checkbox",
        )
        arm_code = st.text_input(
            "Confirmación manual",
            value="",
            key="live_arm_code",
            placeholder="Escribe ARMAR LIVE para confirmar",
        )
        live_armed = (not paper_mode) and all_ready and arm_live and arm_code.strip().upper() == "ARMAR LIVE"
        st.info("Estado armado: ACTIVO" if live_armed else "Estado armado: INACTIVO")

        st.markdown("### Control del bot LIVE")
        live_pid = get_live_bot_pid()
        st.write(f"Estado del proceso: {'EJECUTANDO' if live_pid else 'DETENIDO'}")
        if live_pid:
            st.caption(f"PID activo: {live_pid}")

        c_start, c_stop = st.columns(2)
        if c_start.button("Iniciar bot LIVE", disabled=not live_armed or bool(live_pid)):
            ok, msg = start_live_bot_process()
            (st.success if ok else st.error)(msg)
            st.rerun()
        if c_stop.button("Detener bot LIVE", disabled=not bool(live_pid)):
            ok, msg = stop_live_bot_process()
            (st.success if ok else st.warning)(msg)
            st.rerun()

        st.markdown("### Comando de arranque LIVE")
        st.code("$env:PAPER_TRADING='false'; .\\.venv\\Scripts\\python.exe -m src.main")
        if not live_armed:
            st.caption("El comando se muestra, pero la ejecución LIVE requiere armado activo y checklist completo.")
        else:
            st.caption("Armado LIVE activo. Puedes ejecutar el comando con seguridad operativa reforzada.")

        if paper_mode:
            if st.button("Ver últimos registros de ejecución"):
                p = PROJECT_ROOT / "data/paper_trades.csv"
                if p.exists():
                    st.dataframe(pd.read_csv(p).tail(100), use_container_width=True)
                else:
                    st.info("Aún no existe data/paper_trades.csv")

            st.markdown("### Gráficos de operación")
            st.caption("Fuente actual del dashboard: data/paper_trades.csv")
            render_paper_trade_charts(
                PROJECT_ROOT / "data/paper_trades.csv",
                widget_prefix="live",
                min_signals_sem=sem_min_signals,
                min_edge_sem=sem_min_edge,
                min_conf_sem=sem_min_conf,
                utc_offset_hours=utc_offset_hours,
                ny_latam_preset_default=ny_latam_preset_default,
            )

            with st.expander("Herramientas de prueba (paper)", expanded=False):
                st.write("Utilidades de prueba para validar pipeline sin enviar órdenes reales.")
                if st.button("Ejecutar smoke test de pipeline"):
                    code, out = run_module("src.bootstrap")
                    st.code(out)
                    st.info(f"Exit code: {code}")
        else:
            st.markdown("### Gráficos de operación")
            st.info(
                "Los gráficos de señales PAPER se ocultan en modo LIVE para evitar confusión. "
                "Usa 'Histórico Operaciones' para ver posiciones y deals reales de MT5."
            )

    with tab_history:
        render_trade_history_tab()


if __name__ == "__main__":
    main()
