from __future__ import annotations

import json
import os
import subprocess
import sys
import signal
import time
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import dotenv_values

from src.model_registry import list_snapshots, restore_snapshot, snapshot_current_models
from src.config import settings
from src.mt5_executor import MT5Executor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
LIVE_PID_PATH = PROJECT_ROOT / "logs/live_bot.pid"


def apply_modern_theme(theme_mode: str = "light") -> None:
    dark = str(theme_mode).strip().lower() == "dark"

    if dark:
        bg_a = "#0e161c"
        bg_b = "#111c24"
        ink = "#e8f1ef"
        muted = "#9db1ac"
        card = "#17242d"
        line = "#29404d"
        shadow = "0 10px 28px rgba(0, 0, 0, 0.35)"
        hero_bg = "linear-gradient(135deg, #18262f 0%, #1a2d38 100%)"
        tab_bg = "#17242d"
        tab_text = "#cfe1dd"
        tab_active_bg = "linear-gradient(135deg, #1f3a47 0%, #3c2e2a 100%)"
        tab_active_text = "#f5fbf9"
        toggle_track_off = "#3b4d58"
        toggle_track_on = "#0ea5a0"
        toggle_knob = "#f8fbfb"
        slider_track = "#324754"
        slider_fill = "#0ea5a0"
        scrollbar_track = "#1b2a34"
        scrollbar_thumb = "#4f6774"
        scrollbar_thumb_hover = "#6e8a98"
        app_bg = (
            "radial-gradient(1200px 450px at 100% -10%, rgba(19, 81, 71, 0.35) 0%, transparent 60%),"
            "radial-gradient(900px 350px at -10% 0%, rgba(154, 52, 18, 0.22) 0%, transparent 55%),"
            "linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 30%)"
        )
    else:
        bg_a = "#f3f7f5"
        bg_b = "#ffffff"
        ink = "#0f2a24"
        muted = "#5f746f"
        card = "#ffffff"
        line = "#d9e6e1"
        shadow = "0 8px 24px rgba(15, 42, 36, 0.08)"
        hero_bg = "linear-gradient(135deg, #ffffff 0%, #f4fbf9 100%)"
        tab_bg = "#ffffff"
        tab_text = "#24433c"
        tab_active_bg = "linear-gradient(135deg, #e0f6f0 0%, #fff3ee 100%)"
        tab_active_text = "#11352f"
        toggle_track_off = "#7f968d"
        toggle_track_on = "#0f766e"
        toggle_knob = "#ffffff"
        slider_track = "#d5e3de"
        slider_fill = "#0f766e"
        scrollbar_track = "#e4ece9"
        scrollbar_thumb = "#88a39a"
        scrollbar_thumb_hover = "#5c7e73"
        app_bg = (
            "radial-gradient(1200px 450px at 100% -10%, #d6efe8 0%, transparent 60%),"
            "radial-gradient(900px 350px at -10% 0%, #fde7df 0%, transparent 55%),"
            "linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 30%)"
        )

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap');

        :root {{
            --bg-a: {bg_a};
            --bg-b: {bg_b};
            --ink: {ink};
            --muted: {muted};
            --card: {card};
            --line: {line};
            --shadow: {shadow};
            --radius: 16px;
            --toggle-track-off: {toggle_track_off};
            --toggle-track-on: {toggle_track_on};
            --toggle-knob: {toggle_knob};
            --slider-track: {slider_track};
            --slider-fill: {slider_fill};
            --scrollbar-track: {scrollbar_track};
            --scrollbar-thumb: {scrollbar_thumb};
            --scrollbar-thumb-hover: {scrollbar_thumb_hover};
        }}

        html, body, [class*="css"], p, span, label, h1, h2, h3, h4, h5 {{
            font-family: 'Manrope', sans-serif;
            color: var(--ink);
        }}

        .stApp {{
            background: {app_bg};
        }}

        div.block-container {{
            padding-top: 4.6rem;
        }}

        #theme-switch-anchor + div div[data-testid="stToggle"] {{
            position: fixed;
            top: 0.75rem;
            right: 1rem;
            width: min(320px, calc(100vw - 1.5rem));
            z-index: 9999;
            margin: 0;
        }}

        @media (max-width: 900px) {{
            #theme-switch-anchor + div div[data-testid="stToggle"] {{
                right: 0.5rem;
                top: 0.5rem;
                width: min(260px, calc(100vw - 1rem));
            }}

            div.block-container {{
                padding-top: 4.9rem;
            }}
        }}

        * {{
            scrollbar-width: thin;
            scrollbar-color: var(--scrollbar-thumb) var(--scrollbar-track);
        }}

        *::-webkit-scrollbar {{
            width: 12px;
            height: 12px;
        }}

        *::-webkit-scrollbar-track {{
            background: var(--scrollbar-track);
            border-radius: 999px;
        }}

        *::-webkit-scrollbar-thumb {{
            background: var(--scrollbar-thumb);
            border-radius: 999px;
            border: 2px solid var(--scrollbar-track);
        }}

        *::-webkit-scrollbar-thumb:hover {{
            background: var(--scrollbar-thumb-hover);
        }}

        .app-hero {{
            border: 1px solid var(--line);
            background: {hero_bg};
            border-radius: 20px;
            padding: 1.1rem 1.2rem;
            margin: 0.25rem 0 0.9rem 0;
            box-shadow: var(--shadow);
        }}

        .app-hero h1 {{
            color: var(--ink);
            font-size: 1.65rem;
            line-height: 1.15;
            margin: 0 0 0.3rem 0;
            letter-spacing: -0.02em;
        }}

        .app-hero p {{
            margin: 0;
            color: var(--muted);
        }}

        .mode-pill {{
            display: inline-block;
            margin-top: 0.65rem;
            border-radius: 999px;
            padding: 0.34rem 0.72rem;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            border: 1px solid transparent;
        }}

        .mode-pill.live {{
            background: #d8f3ef;
            color: #0f766e;
            border-color: #9adbcf;
        }}

        .mode-pill.paper {{
            background: #fde7df;
            color: #9a3412;
            border-color: #f4b7a6;
        }}

        div[data-testid="stMetric"] {{
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 0.55rem 0.7rem;
            box-shadow: var(--shadow);
        }}

        div[data-testid="stMetric"] * {{
            color: var(--ink) !important;
        }}

        div[data-testid="stTabs"] button[role="tab"] {{
            border-radius: 12px;
            border: 1px solid var(--line);
            background: {tab_bg};
            margin-right: 0.28rem;
            color: {tab_text};
            font-weight: 700;
        }}

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
            background: {tab_active_bg};
            border-color: var(--line);
            color: {tab_active_text};
        }}

        div[data-testid="stToggle"] {{
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.35rem 0.55rem;
            margin-bottom: 0.35rem;
            box-shadow: var(--shadow);
        }}

        div[data-testid="stToggle"] label,
        div[data-testid="stToggle"] p,
        div[data-testid="stToggle"] span {{
            color: var(--ink) !important;
            font-weight: 700;
        }}

        div[data-testid="stToggle"] [role="switch"] {{
            background-color: var(--toggle-track-off) !important;
            border: 2px solid var(--line) !important;
        }}

        div[data-testid="stToggle"] [role="switch"][aria-checked="true"] {{
            background-color: var(--toggle-track-on) !important;
            border-color: var(--toggle-track-on) !important;
        }}

        div[data-testid="stToggle"] [role="switch"] > div {{
            background: var(--toggle-knob) !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.28);
        }}

        div[data-testid="stToggle"] [data-baseweb="switch"] > div {{
            background-color: var(--toggle-track-off) !important;
            border: 1px solid var(--line) !important;
        }}

        div[data-testid="stToggle"] [data-baseweb="switch"] input:checked + div {{
            background-color: var(--toggle-track-on) !important;
            border-color: var(--toggle-track-on) !important;
        }}

        div[data-testid="stToggle"] [data-baseweb="switch"] > div > div {{
            background: var(--toggle-knob) !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25);
        }}

        div[data-testid="stSlider"] [role="slider"] {{
            background: var(--slider-fill) !important;
            border: 2px solid var(--toggle-knob) !important;
            box-shadow: 0 0 0 2px color-mix(in srgb, var(--slider-fill) 30%, transparent) !important;
        }}

        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {{
            background: var(--slider-track) !important;
        }}

        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(2) {{
            background: var(--slider-fill) !important;
        }}

        .stButton > button {{
            border-radius: 12px;
            border: 1px solid #a9d3c8;
            background: linear-gradient(135deg, #0f766e 0%, #146a63 100%);
            color: #ffffff;
            font-weight: 700;
        }}

        .stButton > button:hover {{
            border-color: #7fbdae;
            filter: brightness(1.05);
        }}

        .stCodeBlock pre, code, .stTextInput input {{
            font-family: 'IBM Plex Mono', monospace !important;
        }}

        .section-card {{
            border: 1px solid var(--line);
            background: var(--card);
            border-radius: 14px;
            padding: 0.72rem 0.85rem;
            margin: 0.35rem 0 0.65rem 0;
            box-shadow: var(--shadow);
        }}

        .section-card h3 {{
            margin: 0;
            font-size: 1.02rem;
            line-height: 1.25;
            color: var(--ink);
        }}

        .section-card p {{
            margin: 0.24rem 0 0 0;
            color: var(--muted);
            font-size: 0.9rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_card(title: str, subtitle: str = "") -> None:
    st.markdown(
        (
            f'<div class="section-card"><h3>{title}</h3>'
            f"<p>{subtitle}</p></div>"
        ),
        unsafe_allow_html=True,
    )


def load_env() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    data = dotenv_values(str(ENV_PATH))
    return {str(k): str(v) for k, v in data.items() if k is not None and v is not None}


def save_env(values: dict[str, str]) -> None:
    lines = [f"{k}={v}" for k, v in values.items()]
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_module(module: str, extra_env: dict[str, str] | None = None) -> tuple[int, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    cmd = [sys.executable, "-m", module]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, output.strip()


def run_script(script_rel_path: str, args: list[str] | None = None, extra_env: dict[str, str] | None = None) -> tuple[int, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    cmd = [sys.executable, script_rel_path]
    if args:
        cmd.extend(args)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, output.strip()


def read_if_exists(path: Path, n: int = 200) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    return "\n".join(lines[:n])


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def parse_datetime_utc(values: pd.Series | list | np.ndarray) -> pd.Series:
    """Parse datetimes to UTC using explicit formats to avoid inference warnings."""
    try:
        return pd.to_datetime(values, utc=True, errors="coerce", format="ISO8601")
    except Exception:
        return pd.to_datetime(values, utc=True, errors="coerce")


def load_live_mt5_trades(symbol: str, history_days: int) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    executor = MT5Executor()
    try:
        executor.initialize()
        open_positions = executor.get_open_positions(symbol)
        recent_deals = executor.get_recent_deals(symbol, days=history_days)
        return open_positions, recent_deals, None
    except Exception as ex:
        return pd.DataFrame(), pd.DataFrame(), str(ex)
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass


def build_monitor_source(environment: str, history_days: int) -> tuple[pd.DataFrame, str | None, str]:
    env_upper = str(environment).strip().upper()
    if env_upper == "LIVE":
        open_live, deals_live, live_error = load_live_mt5_trades(settings.symbol, history_days)
        if live_error:
            return pd.DataFrame(), live_error, "MT5 LIVE"

        frames: list[pd.DataFrame] = []

        if not deals_live.empty:
            deals = deals_live.copy()
            deals["time_utc"] = parse_datetime_utc(deals.get("time_utc"))
            deals["side"] = deals.get("side", "").astype(str).str.upper()
            deals = deals[deals["side"].isin(["BUY", "SELL"])].copy()
            if "entry_label" not in deals.columns:
                deals["entry_label"] = "LIVE_DEAL"
            if "confidence" not in deals.columns:
                deals["confidence"] = 0.5
            if "proba_buy" not in deals.columns:
                deals["proba_buy"] = np.where(deals["side"] == "BUY", 1.0, 0.0)
            if "event_name" not in deals.columns:
                deals["event_name"] = deals.get("entry_label", "LIVE_DEAL").astype(str)
            if "event_currency" not in deals.columns:
                deals["event_currency"] = str(settings.symbol)[:3]
            if "event_importance" not in deals.columns:
                deals["event_importance"] = np.nan
            frames.append(deals)

        if not open_live.empty:
            opens = open_live.copy()
            opens["time_utc"] = parse_datetime_utc(opens.get("time_utc"))
            opens["side"] = opens.get("side", "").astype(str).str.upper()
            opens = opens[opens["side"].isin(["BUY", "SELL"])].copy()
            opens["entry_label"] = "OPEN_POSITION"
            opens["confidence"] = 0.5
            opens["proba_buy"] = np.where(opens["side"] == "BUY", 1.0, 0.0)
            opens["event_name"] = "LIVE_OPEN_POSITION"
            opens["event_currency"] = str(settings.symbol)[:3]
            opens["event_importance"] = np.nan
            frames.append(opens)

        if not frames:
            return pd.DataFrame(), None, "MT5 LIVE"

        merged = pd.concat(frames, ignore_index=True)
        cols = [
            c
            for c in [
                "time_utc",
                "side",
                "confidence",
                "proba_buy",
                "event_name",
                "entry_label",
                "event_currency",
                "event_importance",
                "symbol",
                "comment",
            ]
            if c in merged.columns
        ]
        return merged[cols], None, "MT5 LIVE"

    paper_path = PROJECT_ROOT / "data/paper_trades.csv"
    return load_csv(paper_path), None, "data/paper_trades.csv"


def parse_int(value: str | None, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def parse_float(value: str | None, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_event_datetime_column(events: pd.DataFrame) -> str | None:
    for col in ["date_utc", "datetime_utc", "time_utc", "event_time_utc"]:
        if col in events.columns:
            return col
    return None


def _is_in_session_window(ts_utc: pd.Timestamp, sessions_raw: str) -> bool:
    sessions = {s.strip().lower() for s in str(sessions_raw or "").split(",") if s.strip()}
    if not sessions:
        sessions = {"london", "ny"}
    if "all" in sessions:
        return True

    hour = int(ts_utc.hour)
    in_london = 6 <= hour <= 11
    in_ny = 12 <= hour <= 17

    london_alias = {"london", "ldn"}
    ny_alias = {"ny", "newyork", "new_york", "new-york", "us"}
    use_london = bool(sessions.intersection(london_alias))
    use_ny = bool(sessions.intersection(ny_alias))
    return (use_london and in_london) or (use_ny and in_ny)


def _format_countdown(seconds_left: float) -> str:
    sec = max(0, int(seconds_left))
    hh = sec // 3600
    mm = (sec % 3600) // 60
    ss = sec % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


EVENTLESS_STRATEGIES = {
    "ema_rsi",
    "ema_rsi_trend",
    "ema_rsi_active",
    "crossover_rsi",
    "turtle_atr",
    "agentic_hybrid",
    "agentic",
    "agentic_ai",
    "multi_agent",
    "fundamental_llm",
    "fundamental",
    "macro_llm",
    "news_llm",
    "driven_trading_agentic_system",
    "driven_agentic",
    "driven",
    "multi_strategy_driven",
}


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
    if strategy_mode in EVENTLESS_STRATEGIES:
        return {"status": "eventless_strategy", "path": str(events_csv_path)}

    if not events_csv_path.exists():
        return {"status": "missing_events", "path": str(events_csv_path)}

    try:
        events = pd.read_csv(events_csv_path)
    except Exception as ex:
        return {"status": "read_error", "path": str(events_csv_path), "error": str(ex)}

    dt_col = _resolve_event_datetime_column(events)
    if not dt_col:
        return {"status": "missing_datetime_col", "path": str(events_csv_path), "columns": list(events.columns)}

    events[dt_col] = parse_datetime_utc(events[dt_col])
    events = events.dropna(subset=[dt_col]).sort_values(dt_col)
    now_utc = pd.Timestamp.now(tz="UTC")
    upcoming = events[events[dt_col] > now_utc].copy()

    if "importance" in upcoming.columns:
        imp = pd.to_numeric(upcoming["importance"], errors="coerce")
        upcoming = upcoming[imp >= float(event_min_importance)]

    session_filter_on = donchian_session_filter or (strategy_mode == "donchian_nylondon")
    if session_filter_on:
        upcoming = upcoming[upcoming[dt_col].apply(lambda ts: _is_in_session_window(ts, donchian_sessions))]

    if upcoming.empty:
        return {
            "status": "no_upcoming",
            "path": str(events_csv_path),
            "session_filter_on": session_filter_on,
        }

    next_event = upcoming.iloc[0]
    event_time_utc = parse_datetime_utc(next_event[dt_col])
    if pd.isna(event_time_utc):
        return {"status": "invalid_event_time", "path": str(events_csv_path)}
    trigger_utc = event_time_utc - pd.Timedelta(seconds=int(seconds_before_event))
    local_tz = timezone(timedelta(hours=float(utc_offset_hours)))
    event_local = event_time_utc.tz_convert(local_tz)
    trigger_local = trigger_utc.tz_convert(local_tz)
    countdown_seconds = float((trigger_utc - now_utc).total_seconds())

    return {
        "status": "ok",
        "path": str(events_csv_path),
        "event_name": str(next_event.get("name", "N/A")),
        "event_currency": str(next_event.get("currency", "N/A")),
        "event_importance": str(next_event.get("importance", "N/A")),
        "event_time_utc": event_time_utc,
        "trigger_utc": trigger_utc,
        "event_time_local": event_local,
        "trigger_local": trigger_local,
        "countdown": _format_countdown(countdown_seconds),
        "session_filter_on": session_filter_on,
        "donchian_sessions": donchian_sessions,
    }


def _is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=False,
            )
            text = (out.stdout or "").strip().lower()
            return ("no tasks are running" not in text) and (str(pid) in text)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, SystemError):
        return False


def get_live_bot_pid() -> int | None:
    if not LIVE_PID_PATH.exists():
        return None
    try:
        pid = int(LIVE_PID_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        return None
    return pid if _is_pid_running(pid) else None


def start_live_bot_process() -> tuple[bool, str]:
    existing_pid = get_live_bot_pid()
    if existing_pid:
        return False, f"Ya existe un bot LIVE ejecutándose (PID {existing_pid})."

    python_path = PROJECT_ROOT / ".venv/Scripts/python.exe"
    if not python_path.exists():
        python_path = Path(sys.executable)
    if not python_path.exists():
        return False, "No se encontró Python para iniciar LIVE (.venv o sys.executable)."

    os.makedirs(LIVE_PID_PATH.parent, exist_ok=True)
    env = os.environ.copy()
    env["PAPER_TRADING"] = "false"

    try:
        creation_flags = 0
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

        proc = subprocess.Popen(
            [str(python_path), "-m", "src.main"],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        LIVE_PID_PATH.write_text(str(proc.pid), encoding="utf-8")
        return True, f"Bot LIVE iniciado (PID {proc.pid})."
    except Exception as ex:
        return False, f"No se pudo iniciar el bot LIVE: {ex}"


def stop_live_bot_process() -> tuple[bool, str]:
    pid = get_live_bot_pid()
    if not pid:
        if LIVE_PID_PATH.exists():
            LIVE_PID_PATH.unlink(missing_ok=True)
        return False, "No hay bot LIVE activo registrado."

    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
        LIVE_PID_PATH.unlink(missing_ok=True)
        return True, f"Bot LIVE detenido (PID {pid})."
    except Exception as ex:
        return False, f"No se pudo detener el bot LIVE: {ex}"


def verify_mt5_connection() -> tuple[bool, str]:
    executor = MT5Executor()
    try:
        executor.initialize()
        return True, "Conexión MT5 verificada correctamente."
    except Exception as ex:
        return False, f"MT5 no disponible: {ex}"
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass


def render_walkforward_charts(report_path: Path) -> None:
    report = load_csv(report_path)
    if report.empty:
        st.info("No hay reporte de walk-forward para graficar.")
        return

    period_col = "month" if "month" in report.columns else ("week" if "week" in report.columns else "split")
    plot_df = report[[period_col, "hit_rate", "avg_r", "max_drawdown_r", "num_trades"]].copy()
    plot_df = plot_df.set_index(period_col)

    st.markdown("#### Rendimiento por periodo")
    st.line_chart(plot_df[["hit_rate", "avg_r"]])
    st.bar_chart(plot_df[["num_trades", "max_drawdown_r"]])


def render_paper_trade_charts(
    paper_source: Path | pd.DataFrame,
    widget_prefix: str,
    min_signals_sem: int,
    min_edge_sem: float,
    min_conf_sem: float,
    utc_offset_hours: float,
    ny_latam_preset_default: bool,
) -> None:
    if isinstance(paper_source, pd.DataFrame):
        paper = paper_source.copy()
    else:
        paper = load_csv(paper_source)
    if paper.empty:
        st.info("No hay registros de ejecución para graficar.")
        return

    required_cols = {"time_utc", "side", "confidence"}
    if not required_cols.issubset(set(paper.columns)):
        st.warning("El archivo de registros no tiene todas las columnas requeridas: time_utc, side, confidence")
        return

    paper["time_utc"] = parse_datetime_utc(paper["time_utc"])
    paper = paper.dropna(subset=["time_utc"]).sort_values("time_utc")
    if paper.empty:
        st.info("Los registros no tienen timestamps válidos.")
        return

    st.markdown("#### Filtros")
    min_date = paper["time_utc"].dt.date.min()
    max_date = paper["time_utc"].dt.date.max()
    date_range = st.date_input(
        "Rango de fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key=f"{widget_prefix}_date_range",
    )

    side_options = sorted(paper["side"].astype(str).str.upper().dropna().unique().tolist())
    selected_sides = st.multiselect(
        "Sides",
        options=side_options,
        default=side_options,
        key=f"{widget_prefix}_sides",
    )

    if "event_currency" in paper.columns:
        cur_options = sorted(paper["event_currency"].astype(str).dropna().unique().tolist())
        selected_currencies = st.multiselect(
            "Monedas de evento",
            options=cur_options,
            default=cur_options,
            key=f"{widget_prefix}_currencies",
        )
    else:
        selected_currencies = []

    if "event_importance" in paper.columns:
        imp_options = sorted(paper["event_importance"].astype(str).dropna().unique().tolist())
        selected_importance = st.multiselect(
            "Importancia",
            options=imp_options,
            default=imp_options,
            key=f"{widget_prefix}_importance",
        )
    else:
        selected_importance = []

    event_query = st.text_input("Buscar evento", value="", key=f"{widget_prefix}_event_query")
    use_ny_latam_preset = st.toggle(
        "Aplicar ventana operativa NY/LATAM",
        value=ny_latam_preset_default,
        key=f"{widget_prefix}_ny_latam_preset",
        help="Filtra automáticamente horas líquidas locales, eventos de mayor relevancia y monedas objetivo.",
    )

    filtered = paper.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["time_utc"].dt.date >= start_date)
            & (filtered["time_utc"].dt.date <= end_date)
        ]
    if selected_sides:
        filtered = filtered[filtered["side"].astype(str).str.upper().isin(selected_sides)]
    if selected_currencies and "event_currency" in filtered.columns:
        filtered = filtered[filtered["event_currency"].astype(str).isin(selected_currencies)]
    if selected_importance and "event_importance" in filtered.columns:
        filtered = filtered[filtered["event_importance"].astype(str).isin(selected_importance)]
    if event_query.strip() and "event_name" in filtered.columns:
        filtered = filtered[
            filtered["event_name"].astype(str).str.contains(event_query.strip(), case=False, na=False)
        ]

    if use_ny_latam_preset:
        offset = pd.Timedelta(hours=float(utc_offset_hours))
        filtered["local_hour"] = (filtered["time_utc"] + offset).dt.hour
        filtered = filtered[(filtered["local_hour"] >= 7) & (filtered["local_hour"] <= 17)]

        if "event_importance" in filtered.columns:
            imp_numeric = pd.to_numeric(filtered["event_importance"], errors="coerce")
            filtered = filtered[imp_numeric.fillna(0) >= 2]

        if "event_currency" in filtered.columns:
            target_ccy = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF", "MXN", "BRL", "CLP"}
            filtered = filtered[filtered["event_currency"].astype(str).str.upper().isin(target_ccy)]

        st.caption(
            f"Preset NY/LATAM activo: hora local UTC{utc_offset_hours:+g} entre 07:00-17:59, importancia >=2 y monedas objetivo."
        )

    if filtered.empty:
        st.info("No hay señales con los filtros seleccionados.")
        return

    filtered["side_upper"] = filtered["side"].astype(str).str.upper()
    filtered["signal"] = filtered["side_upper"].map({"BUY": 1, "SELL": -1}).fillna(0)
    filtered["signal_weighted"] = filtered["signal"] * filtered["confidence"].fillna(0.0)
    filtered["signal_cum"] = filtered["signal_weighted"].cumsum()
    filtered["hour"] = filtered["time_utc"].dt.hour
    if "proba_buy" in filtered.columns:
        proba_buy = pd.to_numeric(filtered["proba_buy"], errors="coerce").fillna(0.5)
        filtered["edge_proxy"] = np.where(filtered["side_upper"] == "BUY", proba_buy, 1.0 - proba_buy)
    else:
        filtered["edge_proxy"] = filtered["confidence"].fillna(0.0)

    st.markdown("#### KPIs")
    c1, c2, c3, c4 = st.columns(4)
    total_signals = int(len(filtered))
    net_bias = float(filtered["signal"].sum())
    avg_conf = float(filtered["confidence"].fillna(0.0).mean())
    top_hour = int(filtered["hour"].mode().iloc[0]) if not filtered["hour"].mode().empty else -1
    c1.metric("Total señales", total_signals)
    c2.metric("Sesgo neto (BUY-SELL)", f"{net_bias:.0f}")
    c3.metric("Confianza media", f"{avg_conf:.3f}")
    c4.metric("Hora pico (UTC)", "N/A" if top_hour < 0 else str(top_hour))

    st.markdown("#### Semáforo de recomendación")
    st.caption(
        f"Umbrales activos: min_signals={min_signals_sem}, min_edge={min_edge_sem:.2f}, min_conf={min_conf_sem:.2f}"
    )
    time_focus = st.selectbox(
        "Ventana recomendación",
        options=["Todo historial", "Solo hoy", "Próximas 24h"],
        index=0,
        key=f"{widget_prefix}_time_focus",
    )
    use_local_day = st.checkbox(
        f"Usar día local (UTC{utc_offset_hours:+g}) para 'Solo hoy'",
        value=True,
        key=f"{widget_prefix}_use_local_day",
    )

    rec_df = filtered.copy()
    ref_col = "event_time_utc" if "event_time_utc" in rec_df.columns else "time_utc"
    rec_df[ref_col] = parse_datetime_utc(rec_df[ref_col])
    rec_df = rec_df.dropna(subset=[ref_col])
    now_utc = pd.Timestamp.now(tz="UTC")
    if time_focus == "Solo hoy":
        if use_local_day:
            offset = pd.Timedelta(hours=float(utc_offset_hours))
            rec_local_date = (rec_df[ref_col] + offset).dt.date
            now_local_date = (now_utc + offset).date()
            rec_df = rec_df[rec_local_date == now_local_date]
        else:
            rec_df = rec_df[rec_df[ref_col].dt.date == now_utc.date()]
    elif time_focus == "Próximas 24h":
        rec_df = rec_df[(rec_df[ref_col] >= now_utc) & (rec_df[ref_col] <= now_utc + pd.Timedelta(hours=24))]

    if rec_df.empty:
        st.info("La ventana temporal seleccionada no contiene datos para recomendación.")
        rec_df = filtered.copy()

    def classify_row(row: pd.Series) -> str:
        hard_fail = (
            row.get("signals", 0) < max(1, int(min_signals_sem * 0.6))
            or row.get("edge_proxy_mean", 0.0) < min_edge_sem - 0.05
            or row.get("confidence_mean", 0.0) < min_conf_sem - 0.05
        )
        if hard_fail:
            return "ROJO"

        strong_pass = (
            row.get("signals", 0) >= min_signals_sem
            and row.get("edge_proxy_mean", 0.0) >= min_edge_sem
            and row.get("confidence_mean", 0.0) >= min_conf_sem
        )
        if strong_pass:
            return "VERDE"
        return "AMARILLO"

    st.markdown("#### Curva acumulada de señales")
    curve = filtered[["time_utc", "signal_cum"]].set_index("time_utc")
    st.line_chart(curve)

    st.markdown("#### Distribución de señales por hora")
    by_hour = filtered.groupby("hour", as_index=True)["signal"].count().to_frame("signals")
    st.bar_chart(by_hour)

    st.markdown("#### Distribución BUY/SELL")
    by_side = filtered.groupby("side_upper", as_index=True)["signal"].count().to_frame("count")
    st.bar_chart(by_side)

    if "event_name" in filtered.columns:
        st.markdown("#### Top eventos por frecuencia")
        top_events = (
            filtered["event_name"].astype(str).value_counts().head(10).rename_axis("event_name").to_frame("count")
        )
        st.dataframe(top_events, use_container_width=True)

    st.markdown("#### Últimas señales filtradas")
    st.dataframe(filtered.tail(100), use_container_width=True)

    st.markdown("#### Rendimiento proxy por moneda")
    if "event_currency" in rec_df.columns:
        by_currency = (
            rec_df.groupby(rec_df["event_currency"].astype(str), as_index=True)
            .agg(
                signals=("signal", "count"),
                confidence_mean=("confidence", "mean"),
                edge_proxy_mean=("edge_proxy", "mean"),
                net_bias=("signal", "sum"),
            )
            .sort_values(["edge_proxy_mean", "signals"], ascending=[False, False])
        )
        by_currency["semaforo"] = by_currency.apply(classify_row, axis=1)
        st.bar_chart(by_currency[["signals", "edge_proxy_mean"]])
        st.dataframe(by_currency.head(20), use_container_width=True)

        st.markdown("##### Monedas recomendadas (VERDE)")
        greens_currency = by_currency[by_currency["semaforo"] == "VERDE"].head(10)
        if greens_currency.empty:
            st.info("No hay monedas en VERDE con los umbrales actuales.")
        else:
            st.dataframe(greens_currency, use_container_width=True)
            st.download_button(
                "Exportar monedas VERDE (CSV)",
                data=greens_currency.reset_index().to_csv(index=False),
                file_name="recommended_currencies_green.csv",
                mime="text/csv",
                key=f"{widget_prefix}_export_green_currency",
            )
    else:
        st.info("No hay columna event_currency para análisis por moneda.")

    st.markdown("#### Rendimiento proxy por evento")
    if "event_name" in rec_df.columns:
        by_event = (
            rec_df.groupby(rec_df["event_name"].astype(str), as_index=True)
            .agg(
                signals=("signal", "count"),
                confidence_mean=("confidence", "mean"),
                edge_proxy_mean=("edge_proxy", "mean"),
                net_bias=("signal", "sum"),
            )
            .sort_values(["signals", "edge_proxy_mean"], ascending=[False, False])
        )
        by_event["semaforo"] = by_event.apply(classify_row, axis=1)
        st.dataframe(by_event.head(25), use_container_width=True)

        st.markdown("##### Eventos recomendados (VERDE)")
        greens_event = by_event[by_event["semaforo"] == "VERDE"].head(15)
        if greens_event.empty:
            st.info("No hay eventos en VERDE con los umbrales actuales.")
        else:
            st.dataframe(greens_event, use_container_width=True)
            st.download_button(
                "Exportar eventos VERDE (CSV)",
                data=greens_event.reset_index().to_csv(index=False),
                file_name="recommended_events_green.csv",
                mime="text/csv",
                key=f"{widget_prefix}_export_green_events",
            )

        st.markdown("##### Top 5 eventos a operar")
        score_df = by_event.copy()
        score_df["signals_score"] = (score_df["signals"] / max(float(min_signals_sem), 1.0)).clip(upper=1.0)
        score_df["operability_score"] = (
            0.45 * score_df["edge_proxy_mean"]
            + 0.35 * score_df["confidence_mean"]
            + 0.20 * score_df["signals_score"]
        )
        score_df = score_df.sort_values(["semaforo", "operability_score", "signals"], ascending=[True, False, False])
        top5 = score_df.head(5)
        st.dataframe(top5[["semaforo", "signals", "confidence_mean", "edge_proxy_mean", "operability_score"]], use_container_width=True)
        st.download_button(
            "Exportar Top 5 eventos (CSV)",
            data=top5.reset_index().to_csv(index=False),
            file_name="top5_events_operability.csv",
            mime="text/csv",
            key=f"{widget_prefix}_export_top5_events",
        )
    else:
        st.info("No hay columna event_name para análisis por evento.")


def _get_mid_column(market: pd.DataFrame) -> pd.Series:
    if {"bid", "ask"}.issubset(set(market.columns)):
        return (pd.to_numeric(market["bid"], errors="coerce") + pd.to_numeric(market["ask"], errors="coerce")) / 2.0
    if "close" in market.columns:
        return pd.to_numeric(market["close"], errors="coerce")
    return pd.Series(dtype=float)


def enrich_trade_history_with_results(trades: pd.DataFrame, market_path: Path) -> pd.DataFrame:
    if trades.empty:
        return trades

    out = trades.copy()
    out["time_utc"] = parse_datetime_utc(out.get("time_utc"))

    event_col = "event_time_utc" if "event_time_utc" in out.columns else "time_utc"
    out[event_col] = parse_datetime_utc(out.get(event_col))

    out["side_upper"] = out.get("side", "").astype(str).str.upper()
    out["signal"] = out["side_upper"].map({"BUY": 1, "SELL": -1}).fillna(0).astype(int)
    out["confidence"] = pd.to_numeric(out.get("confidence"), errors="coerce").fillna(0.0)

    market = load_csv(market_path)
    if market.empty or "time_utc" not in market.columns:
        out["ret_post"] = np.nan
        out["result_r"] = np.nan
        out["result_label"] = "SIN_MARKET_DATA"
        out["balance_r"] = np.nan
        return out

    market = market.copy()
    market["time_utc"] = parse_datetime_utc(market["time_utc"])
    market = market.dropna(subset=["time_utc"]).sort_values("time_utc")
    market["mid"] = _get_mid_column(market)
    if {"bid", "ask"}.issubset(set(market.columns)):
        market["bid"] = pd.to_numeric(market["bid"], errors="coerce")
        market["ask"] = pd.to_numeric(market["ask"], errors="coerce")
        market["spread_abs"] = market["ask"] - market["bid"]
        market["spread_bps"] = np.where(
            market["mid"] > 0,
            (market["spread_abs"] / market["mid"]) * 10000.0,
            np.nan,
        )
    else:
        market["spread_bps"] = np.nan
    market = market.dropna(subset=["mid"])

    if market.empty:
        out["ret_post"] = np.nan
        out["result_r"] = np.nan
        out["result_label"] = "SIN_MID_PRICE"
        out["balance_r"] = np.nan
        return out

    market_idx = market.set_index("time_utc")
    market_ts = market_idx["mid"]

    ret_post = []
    result_r = []
    spread_bps_real = []
    for _, row in out.iterrows():
        event_time = row.get(event_col)
        signal = int(row.get("signal", 0))

        if pd.isna(event_time) or signal == 0:
            ret_post.append(np.nan)
            result_r.append(np.nan)
            spread_bps_real.append(np.nan)
            continue

        t0 = event_time + pd.Timedelta(seconds=5)
        t1 = event_time + pd.Timedelta(seconds=60)

        try:
            p0_idx = market_ts.index.get_indexer([t0], method="nearest")[0]
            p1_idx = market_ts.index.get_indexer([t1], method="nearest")[0]
            p0 = float(market_ts.iloc[p0_idx])
            p1 = float(market_ts.iloc[p1_idx])
            spread_entry_bps = pd.to_numeric(market_idx["spread_bps"].iloc[p0_idx], errors="coerce")
            if p0 <= 0:
                ret_post.append(np.nan)
                result_r.append(np.nan)
                spread_bps_real.append(np.nan)
                continue
            realized_ret = (p1 - p0) / p0
            trade_ret = realized_ret * signal
            ret_post.append(trade_ret)
            result_r.append(1.0 if trade_ret > 0 else -1.0)
            spread_bps_real.append(float(spread_entry_bps) if pd.notna(spread_entry_bps) else np.nan)
        except Exception:
            ret_post.append(np.nan)
            result_r.append(np.nan)
            spread_bps_real.append(np.nan)

    out["ret_post"] = ret_post
    out["result_r"] = result_r
    out["spread_bps_real"] = spread_bps_real
    out["result_label"] = np.where(
        out["result_r"].isna(),
        "SIN_RESULTADO",
        np.where(out["result_r"] > 0, "WIN", "LOSS"),
    )
    out = out.sort_values("time_utc")
    out["balance_r"] = out["result_r"].fillna(0.0).cumsum()

    return out

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
    st.markdown("### Estado LIVE en tiempo real")
    live_pid = get_live_bot_pid()

    report_obj: dict[str, object] = {}
    if daily_report_path.exists():
        try:
            report_obj = json.loads(daily_report_path.read_text(encoding="utf-8"))
        except Exception:
            report_obj = {}

    activity = load_csv(live_activity_path)
    if not activity.empty and "time_utc" in activity.columns:
        activity["time_utc"] = parse_datetime_utc(activity["time_utc"])
        activity = activity.dropna(subset=["time_utc"]).sort_values("time_utc")

    activity_live = activity.copy()
    if (not activity_live.empty) and ("mode" in activity_live.columns):
        activity_live = activity_live[activity_live["mode"].astype(str).str.upper() == "LIVE"].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Archivo actividad", "OK" if live_activity_path.exists() else "Missing")
    c2.metric("Reporte diario", "OK" if daily_report_path.exists() else "Missing")
    c3.metric("Proceso bot", "RUNNING" if live_pid else "STOPPED")
    c4.metric("PID bot", str(live_pid) if live_pid else "N/A")

    events_path = Path(events_csv)
    if not events_path.is_absolute():
        events_path = PROJECT_ROOT / events_path
    next_trigger = get_next_trigger_info(
        events_csv_path=events_path,
        strategy_mode=strategy_mode,
        seconds_before_event=seconds_before_event,
        event_min_importance=event_min_importance,
        utc_offset_hours=utc_offset_hours,
        donchian_session_filter=donchian_session_filter,
        donchian_sessions=donchian_sessions,
    )

    st.markdown("#### Proximo trigger")
    next_status = str(next_trigger.get("status", "unknown"))
    if next_status == "ok":
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Countdown", str(next_trigger.get("countdown", "N/A")))
        t2.metric("Trigger UTC", str(next_trigger.get("trigger_utc", "N/A")))
        t3.metric("Trigger local", str(next_trigger.get("trigger_local", "N/A")))
        t4.metric("Evento", f"{next_trigger.get('event_currency', 'N/A')} | imp {next_trigger.get('event_importance', 'N/A')}")
        st.caption(
            f"{next_trigger.get('event_name', 'N/A')} | event_utc={next_trigger.get('event_time_utc', 'N/A')} | "
            f"event_local={next_trigger.get('event_time_local', 'N/A')}"
        )
    elif next_status == "no_upcoming":
        st.info(
            "No hay eventos próximos que cumplan filtros actuales "
            f"(importance>={event_min_importance}, sesiones={donchian_sessions if (donchian_session_filter or strategy_mode == 'donchian_nylondon') else 'todas'})."
        )
    elif next_status == "eventless_strategy":
        st.info("La estrategia activa opera en modo continuo (eventless): no depende de calendario para abrir operaciones.")
    elif next_status == "missing_events":
        st.warning(f"No se encontró archivo de eventos: {next_trigger.get('path', 'N/A')}")
    elif next_status == "missing_datetime_col":
        st.warning("El archivo de eventos no tiene columna datetime compatible (date_utc/datetime_utc/time_utc/event_time_utc).")
    else:
        st.warning(f"No se pudo calcular el próximo trigger: {next_trigger.get('error', 'error desconocido')}")

    if not activity_live.empty:
        last_row = activity_live.iloc[-1]
        st.caption(
            f"Última acción: {str(last_row.get('action', 'N/A'))} | "
            f"Último evento UTC: {str(last_row.get('time_utc', 'N/A'))}"
        )
    elif not activity.empty:
        st.caption("No hay actividad LIVE reciente en el archivo (solo registros PAPER).")
    else:
        st.caption("Última acción: N/A | Último evento UTC: N/A")

    if report_obj:
        st.caption(
            f"Reporte 24h: generado {report_obj.get('generated_at_utc', 'N/A')} | "
            f"actividad={report_obj.get('activity', {}).get('rows', 0)}"
        )

        agents_obj = report_obj.get("agents", {}) if isinstance(report_obj, dict) else {}
        by_agent = agents_obj.get("by_agent", {}) if isinstance(agents_obj, dict) else {}
        if isinstance(by_agent, dict) and by_agent:
            st.markdown("#### Resumen por agente (24h)")
            rows = []
            for agent_name, obj in by_agent.items():
                if not isinstance(obj, dict):
                    continue
                rows.append(
                    {
                        "agent": str(agent_name),
                        "strategy_class": str(obj.get("strategy_class", "")),
                        "rows": int(obj.get("rows", 0)),
                        "signal_rows": int(obj.get("signal_rows", 0)),
                        "signal_rate": float(obj.get("signal_rate", 0.0)),
                        "max_calls": int(obj.get("max_calls", 0)),
                        "max_decisions": int(obj.get("max_decisions", 0)),
                        "last_side": str(obj.get("last_side", "")),
                        "last_seen_utc": str(obj.get("last_seen_utc", "")),
                    }
                )

            if rows:
                df_agents = pd.DataFrame(rows).sort_values(["signal_rows", "rows"], ascending=False)
                a1, a2, a3 = st.columns(3)
                a1.metric("Agentes activos 24h", int(len(df_agents)))
                a2.metric("Señales agente 24h", int(df_agents["signal_rows"].sum()))
                a3.metric("Tasa señal prom.", f"{float(df_agents['signal_rate'].mean()):.2%}")
                st.dataframe(df_agents, use_container_width=True)

    if activity_live.empty:
        st.info("No hay actividad LIVE registrada todavía.")
        return

    now_utc = pd.Timestamp.now(tz="UTC")
    recent = activity_live[activity_live["time_utc"] >= (now_utc - pd.Timedelta(hours=24))].copy()
    env_local = load_env()
    no_money_alert_minutes = parse_int(env_local.get("UI_NO_MONEY_ALERT_MINUTES"), 60)
    heartbeat_minutes = parse_float(env_local.get("UI_LIVE_HEARTBEAT_MINUTES"), 15.0)

    no_money_actions = {
        "order_error_no_money",
        "order_error_no_money_eventless",
    }
    success_actions = {
        "order_sent",
        "order_sent_eventless",
    }
    recent_no_money = (
        recent[recent["action"].astype(str).isin(no_money_actions)].copy()
        if (not recent.empty and "action" in recent.columns)
        else pd.DataFrame()
    )
    mins_since_no_money: float | None = None
    if not recent_no_money.empty:
        last_nm = recent_no_money.iloc[-1]
        parsed_nm_ts = parse_datetime_utc([last_nm.get("time_utc")])
        last_nm_ts = parsed_nm_ts[0] if len(parsed_nm_ts) > 0 else pd.NaT
        has_activity_after_no_money = False
        if pd.notna(last_nm_ts):
            mins_since_no_money = float((now_utc - last_nm_ts).total_seconds() / 60.0)
            has_activity_after_no_money = bool((recent["time_utc"] > last_nm_ts).any()) if not recent.empty else False

        success_after_no_money = False
        if pd.notna(last_nm_ts) and (not recent.empty):
            recent_success = recent[recent["action"].astype(str).isin(success_actions)].copy()
            if not recent_success.empty:
                success_after_no_money = bool((recent_success["time_utc"] > last_nm_ts).any())

        alert_msg = (
            "Alerta de riesgo/fondos: se detectaron rechazos por balance insuficiente o NO_MONEY en MT5. "
            "El bot sigue activo y no se detiene, pero no puede abrir nuevas operaciones hasta corregir margen/fondos."
        )
        if success_after_no_money:
            st.success("NO_MONEY resuelto: se detectó al menos una orden enviada con éxito después del último rechazo.")
        elif has_activity_after_no_money and (mins_since_no_money is not None) and (mins_since_no_money > float(heartbeat_minutes)):
            st.warning(
                "Último NO_MONEY detectado, pero hubo actividad posterior sin repetir rechazo inmediato. "
                "Verifica próximos intentos de orden para confirmar resolución completa."
            )
        elif (mins_since_no_money is None) or (mins_since_no_money <= float(no_money_alert_minutes)):
            st.error(alert_msg)
        else:
            st.warning("Hubo rechazos NO_MONEY anteriormente, pero no son recientes.")

        nm_age_txt = "N/A" if mins_since_no_money is None else f"{mins_since_no_money:.1f}m"
        st.caption(
            f"Último rechazo NO_MONEY: {str(last_nm.get('time_utc', 'N/A'))} | "
            f"edad={nm_age_txt} | action={str(last_nm.get('action', 'N/A'))} | "
            f"detail={str(last_nm.get('detail', ''))[:220]}"
        )

    # Semáforo operativo basado en señal de vida reciente y errores de calendario.
    last_ts = activity_live["time_utc"].iloc[-1]
    mins_since_last = float((now_utc - last_ts).total_seconds() / 60.0)
    has_recent_heartbeat = mins_since_last <= float(heartbeat_minutes)
    has_recent_no_money = (mins_since_no_money is not None) and (mins_since_no_money <= float(heartbeat_minutes) * 2.0)
    has_calendar_error = bool(recent["action"].astype(str).eq("calendar_refresh_error").any()) if not recent.empty else False
    has_recent_refresh = bool(recent["action"].astype(str).eq("calendar_refresh").any()) if not recent.empty else False
    only_no_events = bool(
        (not recent.empty)
        and recent["action"].astype(str).isin(["calendar_refresh", "no_upcoming_events"]).all()
    )

    if (live_pid is not None) and has_recent_heartbeat and has_recent_refresh and (not has_calendar_error):
        health_state = "VERDE"
        health_msg = "Bot activo y refrescando calendario con normalidad."
        st.success(f"Semáforo LIVE: {health_state} | {health_msg}")
    elif (live_pid is not None) and has_recent_no_money:
        health_state = "AMARILLO"
        health_msg = "Bot activo, pero bloqueado por margen/fondos (NO_MONEY reciente)."
        st.warning(f"Semáforo LIVE: {health_state} | {health_msg}")
    elif (live_pid is not None) and has_recent_heartbeat and (only_no_events or not has_recent_refresh):
        health_state = "AMARILLO"
        health_msg = "Bot activo, pero sin eventos operables recientes."
        st.warning(f"Semáforo LIVE: {health_state} | {health_msg}")
    else:
        health_state = "ROJO"
        if live_pid is None:
            health_msg = "Bot detenido o sin PID válido."
        else:
            health_msg = "Actividad estancada o error de calendario. Revisar conectividad/fuente de eventos."
        st.error(f"Semáforo LIVE: {health_state} | {health_msg}")

    s1, s2, s3 = st.columns(3)
    s1.metric("Estado", health_state)
    s2.metric("Min desde última actividad", f"{mins_since_last:.1f}")
    s3.metric("Errores calendar 24h", int(recent["action"].astype(str).eq("calendar_refresh_error").sum()) if not recent.empty else 0)

    st.markdown("#### Acciones últimas 24h")
    if recent.empty:
        st.info("Sin acciones en las últimas 24h.")
    else:
        counts = recent["action"].astype(str).value_counts().rename_axis("action").to_frame("count")
        st.bar_chart(counts)

    st.markdown("#### Últimos eventos LIVE")
    cols = [c for c in ["time_utc", "mode", "strategy", "action", "event_id", "detail"] if c in activity_live.columns]
    st.dataframe(activity_live[cols].tail(120).sort_values("time_utc", ascending=False), use_container_width=True)


def render_trade_history_tab() -> None:
    st.subheader("Histórico de operaciones")

    st.markdown("### LIVE MT5 (real)")
    history_days = st.slider(
        "Ventana de historial LIVE (días)",
        min_value=1,
        max_value=30,
        value=7,
        step=1,
        key="live_history_days",
    )
    open_live, deals_live, live_error = load_live_mt5_trades(settings.symbol, history_days)

    if live_error:
        st.warning(f"No se pudo leer LIVE MT5: {live_error}")
    else:
        l1, l2 = st.columns(2)
        l1.metric("Posiciones abiertas LIVE", int(len(open_live)))
        l2.metric("Deals LIVE recientes", int(len(deals_live)))

        st.markdown("#### Posiciones abiertas (LIVE)")
        if open_live.empty:
            st.info("No hay posiciones abiertas en MT5 para el símbolo actual.")
        else:
            open_cols = [
                c
                for c in ["time_utc", "ticket", "symbol", "side", "volume", "price_open", "sl", "tp", "profit", "comment"]
                if c in open_live.columns
            ]
            st.dataframe(open_live[open_cols].sort_values("time_utc", ascending=False), use_container_width=True)

        st.markdown("#### Deals recientes (LIVE)")
        if deals_live.empty:
            st.info("No hay deals LIVE en la ventana seleccionada.")
        else:
            deal_cols = [
                c
                for c in ["time_utc", "ticket", "position_id", "symbol", "entry_label", "side", "volume", "price", "profit", "commission", "swap", "comment"]
                if c in deals_live.columns
            ]
            st.dataframe(deals_live[deal_cols], use_container_width=True)

    st.markdown("---")
    st.markdown("### PAPER / Simulado")

    paper_path = PROJECT_ROOT / "data/paper_trades.csv"
    trades = load_csv(paper_path)
    if trades.empty:
        st.info(
            "No hay histórico aún. El registro disponible en la UI se construye con data/paper_trades.csv "
            "(pipeline de observabilidad)."
        )
        return

    market_path = PROJECT_ROOT / settings.market_csv
    enriched = enrich_trade_history_with_results(trades, market_path=market_path)

    env_local = load_env()
    risk_usd_default = parse_float(env_local.get("RISK_USD_PER_TRADE"), 25.0)
    comm_usd_default = parse_float(env_local.get("COMMISSION_USD_PER_TRADE"), 0.0)
    spread_bps_default = parse_float(env_local.get("SPREAD_BPS_PER_TRADE"), 0.0)
    dynamic_spread_default = parse_bool(env_local.get("DYNAMIC_SPREAD_COST"), True)

    risk_usd = st.number_input(
        "Riesgo estimado por operación (USD)",
        min_value=1.0,
        max_value=100000.0,
        value=float(risk_usd_default),
        step=1.0,
        key="history_risk_usd",
        help="Convierte el balance en R a balance monetario estimado: USD = R * riesgo_por_operacion.",
    )
    comm_usd = st.number_input(
        "Comisión estimada por operación (USD)",
        min_value=0.0,
        max_value=10000.0,
        value=float(comm_usd_default),
        step=0.1,
        key="history_comm_usd",
        help="Costo fijo por operación (ida y vuelta).",
    )
    spread_bps = st.number_input(
        "Spread/costo variable (bps por operación)",
        min_value=0.0,
        max_value=500.0,
        value=float(spread_bps_default),
        step=0.1,
        key="history_spread_bps",
        help="Costo variable sobre riesgo: costo_spread = riesgo * (bps / 10000).",
    )
    use_dynamic_spread = st.toggle(
        "Usar spread real por operación (si hay bid/ask)",
        value=bool(dynamic_spread_default),
        key="history_dynamic_spread",
        help="Si está activo, usa spread real en bps al momento de entrada. Si falta, usa el bps fijo.",
    )

    st.caption(f"Archivo de operaciones: {paper_path}")
    st.caption(f"Archivo de mercado para resultados: {market_path}")

    if "time_utc" in enriched.columns:
        enriched["time_utc"] = parse_datetime_utc(enriched["time_utc"])
        min_date = enriched["time_utc"].dt.date.min()
        max_date = enriched["time_utc"].dt.date.max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.date_input(
                "Rango de fechas",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="history_date_range",
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                enriched = enriched[
                    (enriched["time_utc"].dt.date >= start_date)
                    & (enriched["time_utc"].dt.date <= end_date)
                ]

    if enriched.empty:
        st.info("No hay operaciones en el rango seleccionado.")
        return

    valid = enriched.dropna(subset=["result_r"])
    total_ops = int(len(enriched))
    ops_with_result = int(len(valid))
    wins = int((valid["result_r"] > 0).sum()) if not valid.empty else 0
    losses = int((valid["result_r"] < 0).sum()) if not valid.empty else 0
    hit_rate = float(wins / ops_with_result) if ops_with_result > 0 else 0.0
    balance_r = float(valid["result_r"].sum()) if ops_with_result > 0 else 0.0
    avg_r = float(valid["result_r"].mean()) if ops_with_result > 0 else 0.0
    balance_usd = balance_r * float(risk_usd)
    avg_usd = avg_r * float(risk_usd)

    enriched["result_usd"] = enriched["result_r"] * float(risk_usd)
    enriched["balance_usd"] = enriched["balance_r"] * float(risk_usd)

    spread_cost_usd_per_trade = float(risk_usd) * (float(spread_bps) / 10000.0)
    dynamic_spread_cost = float(risk_usd) * (pd.to_numeric(enriched.get("spread_bps_real"), errors="coerce") / 10000.0)
    dynamic_available = dynamic_spread_cost.notna()
    effective_spread_cost = np.where(
        use_dynamic_spread,
        np.where(dynamic_available, dynamic_spread_cost, spread_cost_usd_per_trade),
        spread_cost_usd_per_trade,
    )
    enriched["spread_cost_usd"] = np.where(enriched["result_r"].isna(), 0.0, effective_spread_cost)
    enriched["cost_usd"] = np.where(
        enriched["result_r"].isna(),
        0.0,
        float(comm_usd) + enriched["spread_cost_usd"],
    )
    enriched["result_usd_net"] = np.where(
        enriched["result_r"].isna(),
        np.nan,
        enriched["result_usd"] - enriched["cost_usd"],
    )
    enriched["balance_usd_net"] = enriched["result_usd_net"].fillna(0.0).cumsum()

    total_cost_usd = float(enriched["cost_usd"].sum())
    balance_usd_net = float(enriched["result_usd_net"].dropna().sum()) if ops_with_result > 0 else 0.0
    avg_usd_net = float(enriched["result_usd_net"].dropna().mean()) if ops_with_result > 0 else 0.0
    dynamic_coverage = float(dynamic_available.mean()) if len(dynamic_available) > 0 else 0.0

    k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
    k1.metric("Operaciones", total_ops)
    k2.metric("Con resultado", ops_with_result)
    k3.metric("Wins", wins)
    k4.metric("Losses", losses)
    k5.metric("Hit Rate", f"{hit_rate:.1%}")
    k6.metric("Balance general (R)", f"{balance_r:+.2f}")
    k7.metric("Balance general (USD)", f"${balance_usd:+,.2f}")
    k8.metric("Balance neto (USD)", f"${balance_usd_net:+,.2f}")
    st.caption(
        f"Promedio por operación: {avg_r:+.3f} R | bruto ${avg_usd:+,.2f} | neto ${avg_usd_net:+,.2f}"
    )
    st.caption(
        f"Costos aplicados: comisión ${float(comm_usd):,.2f} + spread base {float(spread_bps):.2f} bps "
        f"(=${spread_cost_usd_per_trade:,.2f}) por operación. Total costos: ${total_cost_usd:,.2f}"
    )
    if use_dynamic_spread:
        st.caption(f"Spread dinámico activo. Cobertura con bid/ask real: {dynamic_coverage:.1%} de operaciones.")

    st.markdown("#### Curva de balance acumulado")
    if "time_utc" in enriched.columns:
        curve = enriched[["time_utc", "balance_r"]].dropna(subset=["time_utc"]).set_index("time_utc")
        if not curve.empty:
            st.line_chart(curve)
        else:
            st.info("No hay timestamps válidos para graficar balance.")

    st.markdown("#### Curva de balance acumulado (USD estimado)")
    if "time_utc" in enriched.columns:
        curve_usd = enriched[["time_utc", "balance_usd"]].dropna(subset=["time_utc"]).set_index("time_utc")
        if not curve_usd.empty:
            st.line_chart(curve_usd)
        else:
            st.info("No hay timestamps válidos para graficar balance USD.")

    st.markdown("#### Curva de balance acumulado neto (USD)")
    if "time_utc" in enriched.columns:
        curve_usd_net = enriched[["time_utc", "balance_usd_net"]].dropna(subset=["time_utc"]).set_index("time_utc")
        if not curve_usd_net.empty:
            st.line_chart(curve_usd_net)
        else:
            st.info("No hay timestamps válidos para graficar balance USD neto.")

    st.markdown("#### Resumen por side")
    if "side_upper" in enriched.columns:
        side_summary = (
            enriched.groupby("side_upper", as_index=False)
            .agg(
                operaciones=("side_upper", "count"),
                wins=("result_label", lambda s: int((s == "WIN").sum())),
                losses=("result_label", lambda s: int((s == "LOSS").sum())),
                balance_r=("result_r", "sum"),
                balance_usd=("result_usd", "sum"),
                balance_usd_net=("result_usd_net", "sum"),
            )
        )
        side_summary["hit_rate"] = np.where(
            side_summary["operaciones"] > 0,
            side_summary["wins"] / side_summary["operaciones"],
            0.0,
        )
        st.dataframe(side_summary, use_container_width=True)

    st.markdown("#### Detalle de operaciones")
    cols_preferred = [
        "time_utc",
        "event_time_utc",
        "event_id",
        "event_name",
        "event_currency",
        "symbol",
        "side",
        "confidence",
        "proba_buy",
        "ret_post",
        "result_label",
        "result_r",
        "result_usd",
        "spread_bps_real",
        "spread_cost_usd",
        "cost_usd",
        "result_usd_net",
        "balance_r",
        "balance_usd",
        "balance_usd_net",
        "mode",
    ]
    cols_present = [c for c in cols_preferred if c in enriched.columns]
    history_view = enriched[cols_present].copy()
    st.dataframe(history_view.sort_values("time_utc", ascending=False).head(500), use_container_width=True)

    st.download_button(
        "Exportar histórico enriquecido (CSV)",
        data=history_view.to_csv(index=False),
        file_name="trade_history_enriched.csv",
        mime="text/csv",
        key="history_export_csv",
    )


def main() -> None:
    st.set_page_config(page_title="Economic AE Control Center", layout="wide")
    env_vals = load_env()
    default_dark = parse_bool(env_vals.get("UI_DARK_MODE"), False)
    if "ui_dark_mode" not in st.session_state:
        st.session_state["ui_dark_mode"] = default_dark

    apply_modern_theme("dark" if st.session_state.get("ui_dark_mode") else "light")

    st.markdown('<div id="theme-switch-anchor"></div>', unsafe_allow_html=True)
    theme_col, _ = st.columns([2, 4])
    with theme_col:
        theme_label = "Tema actual: Oscuro (Luna)" if st.session_state.get("ui_dark_mode") else "Tema actual: Claro (Sol)"
        st.toggle(theme_label, key="ui_dark_mode", help="Activa o desactiva el modo oscuro.")

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
    if paper_mode:
        st.warning("Actualmente estás en PAPER mode. Cambia a LIVE en Configuración para operar real.")
    else:
        st.success("Actualmente estás en LIVE mode (producción real).")

    tab_overview, tab_config, tab_data, tab_train, tab_backtest, tab_live, tab_history = st.tabs(
        ["Resumen", "Configuración", "Datos", "Entrenamiento", "Backtest", "Operación Real", "Histórico Operaciones"]
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

    with tab_config:
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

    with tab_data:
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

    with tab_train:
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

    with tab_backtest:
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
