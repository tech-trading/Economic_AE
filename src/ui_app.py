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
    if strategy_mode in {"ema_rsi", "ema_rsi_trend", "ema_rsi_active", "crossover_rsi", "agentic_hybrid", "agentic", "agentic_ai", "multi_agent"}:
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

    events[dt_col] = pd.to_datetime(events[dt_col], utc=True, errors="coerce")
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
    event_time_utc = pd.to_datetime(next_event[dt_col], utc=True, errors="coerce")
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
    paper_path: Path,
    widget_prefix: str,
    min_signals_sem: int,
    min_edge_sem: float,
    min_conf_sem: float,
    utc_offset_hours: float,
    ny_latam_preset_default: bool,
) -> None:
    paper = load_csv(paper_path)
    if paper.empty:
        st.info("No hay registros de ejecución para graficar.")
        return

    required_cols = {"time_utc", "side", "confidence"}
    if not required_cols.issubset(set(paper.columns)):
        st.warning("El archivo de registros no tiene todas las columnas requeridas: time_utc, side, confidence")
        return

    paper["time_utc"] = pd.to_datetime(paper["time_utc"], utc=True, errors="coerce")
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
    rec_df[ref_col] = pd.to_datetime(rec_df[ref_col], utc=True, errors="coerce")
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
    out["time_utc"] = pd.to_datetime(out.get("time_utc"), utc=True, errors="coerce")

    event_col = "event_time_utc" if "event_time_utc" in out.columns else "time_utc"
    out[event_col] = pd.to_datetime(out.get(event_col), utc=True, errors="coerce")

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
    market["time_utc"] = pd.to_datetime(market["time_utc"], utc=True, errors="coerce")
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
        activity["time_utc"] = pd.to_datetime(activity["time_utc"], utc=True, errors="coerce")
        activity = activity.dropna(subset=["time_utc"]).sort_values("time_utc")

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

    if not activity.empty:
        last_row = activity.iloc[-1]
        st.caption(
            f"Última acción: {str(last_row.get('action', 'N/A'))} | "
            f"Último evento UTC: {str(last_row.get('time_utc', 'N/A'))}"
        )
    else:
        st.caption("Última acción: N/A | Último evento UTC: N/A")

    if report_obj:
        st.caption(
            f"Reporte 24h: generado {report_obj.get('generated_at_utc', 'N/A')} | "
            f"actividad={report_obj.get('activity', {}).get('rows', 0)}"
        )

    if activity.empty:
        st.info("No hay actividad LIVE registrada todavía.")
        return

    now_utc = pd.Timestamp.now(tz="UTC")
    recent = activity[activity["time_utc"] >= (now_utc - pd.Timedelta(hours=24))].copy()

    # Semáforo operativo basado en señal de vida reciente y errores de calendario.
    last_ts = activity["time_utc"].iloc[-1]
    mins_since_last = float((now_utc - last_ts).total_seconds() / 60.0)
    has_recent_heartbeat = mins_since_last <= 5.0
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
    cols = [c for c in ["time_utc", "mode", "strategy", "action", "event_id", "detail"] if c in activity.columns]
    st.dataframe(activity[cols].tail(120).sort_values("time_utc", ascending=False), use_container_width=True)


def render_trade_history_tab() -> None:
    st.subheader("Histórico de operaciones")

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
        enriched["time_utc"] = pd.to_datetime(enriched["time_utc"], utc=True, errors="coerce")
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

    st.title("Economic AE Control Center")
    st.caption("Panel orientado a operación real: configuración LIVE, monitoreo, entrenamiento y validación")

    env_vals = load_env()
    sem_min_signals = parse_int(env_vals.get("SEM_MIN_SIGNALS"), 8)
    sem_min_edge = parse_float(env_vals.get("SEM_MIN_EDGE"), 0.58)
    sem_min_conf = parse_float(env_vals.get("SEM_MIN_CONF"), 0.60)
    utc_offset_hours = parse_float(env_vals.get("UTC_OFFSET_HOURS"), -5.0)
    ny_latam_preset_default = parse_bool(env_vals.get("NY_LATAM_PRESET_DEFAULT"), False)
    paper_mode = parse_bool(env_vals.get("PAPER_TRADING"), settings.paper_trading)
    strategy_mode = (env_vals.get("STRATEGY") or getattr(settings, "strategy", "default") or "default").strip().lower()

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
        c1, c2, c3 = st.columns(3)
        c1.metric("Events CSV", "OK" if (PROJECT_ROOT / "data/events.csv").exists() else "Missing")
        c2.metric("Market CSV", "OK" if (PROJECT_ROOT / "data/market_ticks.csv").exists() else "Missing")
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
        else:
            st.caption("Usando estrategia base de ensemble (tabular + LSTM cuando esté disponible).")

        st.subheader("Último summary de walk-forward")
        summary = read_if_exists(PROJECT_ROOT / "models/walkforward_summary.json")
        st.code(summary or "No disponible")

        render_walkforward_charts(PROJECT_ROOT / "models/walkforward_monthly_report.csv")

        st.subheader("Monitoreo de ejecución")
        st.caption("Vista analítica sobre los registros disponibles en data/paper_trades.csv")
        render_paper_trade_charts(
            PROJECT_ROOT / "data/paper_trades.csv",
            widget_prefix="overview",
            min_signals_sem=sem_min_signals,
            min_edge_sem=sem_min_edge,
            min_conf_sem=sem_min_conf,
            utc_offset_hours=utc_offset_hours,
            ny_latam_preset_default=ny_latam_preset_default,
        )

    with tab_config:
        st.subheader("Parámetros de trading y filtros")

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
        strategy = st.selectbox(
            "Estrategia de decisión",
            options=["default", "zscore", "momentum", "donchian", "donchian_nylondon", "ema_rsi_trend", "agentic_hybrid"],
            index=["default", "zscore", "momentum", "donchian", "donchian_nylondon", "ema_rsi_trend", "agentic_hybrid"].index(strategy_mode) if strategy_mode in ["default", "zscore", "momentum", "donchian", "donchian_nylondon", "ema_rsi_trend", "agentic_hybrid"] else 0,
            help="Selecciona la lógica para generar señal de entrada antes de enviar órdenes.",
        )

        st.markdown("### Parámetros de estrategia")
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
        st.markdown("### Parámetros Agentic IA")
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
        label_mode = st.selectbox(
            "Modo de etiquetado",
            options=["sign", "quantile", "quantile_monthly"],
            index=["sign", "quantile", "quantile_monthly"].index(env_vals.get("DIRECTION_LABEL_MODE", "quantile_monthly")) if env_vals.get("DIRECTION_LABEL_MODE", "quantile_monthly") in ["sign", "quantile", "quantile_monthly"] else 2,
        )
        st.markdown("### Preset semáforo")
        sem_min_signals_in = st.number_input("SEM_MIN_SIGNALS", min_value=1, max_value=1000, value=sem_min_signals, step=1)
        sem_min_edge_in = st.number_input("SEM_MIN_EDGE", min_value=0.0, max_value=1.0, value=float(sem_min_edge), step=0.01)
        sem_min_conf_in = st.number_input("SEM_MIN_CONF", min_value=0.0, max_value=1.0, value=float(sem_min_conf), step=0.01)
        st.markdown("### Preset operativo")
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
            if not paper_mode:
                st.warning("Herramientas de prueba bloqueadas porque el sistema está configurado en LIVE.")
            if st.button("Ejecutar smoke test de pipeline", disabled=not paper_mode):
                code, out = run_module("src.bootstrap")
                st.code(out)
                st.info(f"Exit code: {code}")

    with tab_history:
        render_trade_history_tab()


if __name__ == "__main__":
    main()
