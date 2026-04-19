from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from src.mt5_executor import MT5Executor
from src.ui.common import load_csv, parse_datetime_utc
from src.ui.env import load_env, parse_float, parse_int


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


def get_live_bot_pid(live_pid_path: Path) -> int | None:
    if not live_pid_path.exists():
        return None
    try:
        pid = int(live_pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        return None
    return pid if _is_pid_running(pid) else None


def start_live_bot_process(project_root: Path, live_pid_path: Path) -> tuple[bool, str]:
    existing_pid = get_live_bot_pid(live_pid_path)
    if existing_pid:
        return False, f"Ya existe un bot LIVE ejecutándose (PID {existing_pid})."

    python_path = project_root / ".venv/Scripts/python.exe"
    if not python_path.exists():
        python_path = Path(sys.executable)
    if not python_path.exists():
        return False, "No se encontró Python para iniciar LIVE (.venv o sys.executable)."

    os.makedirs(live_pid_path.parent, exist_ok=True)
    env = os.environ.copy()
    env["PAPER_TRADING"] = "false"

    try:
        creation_flags = 0
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

        proc = subprocess.Popen(
            [str(python_path), "-m", "src.main"],
            cwd=str(project_root),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        live_pid_path.write_text(str(proc.pid), encoding="utf-8")
        return True, f"Bot LIVE iniciado (PID {proc.pid})."
    except Exception as ex:
        return False, f"No se pudo iniciar el bot LIVE: {ex}"


def stop_live_bot_process(live_pid_path: Path) -> tuple[bool, str]:
    pid = get_live_bot_pid(live_pid_path)
    if not pid:
        if live_pid_path.exists():
            live_pid_path.unlink(missing_ok=True)
        return False, "No hay bot LIVE activo registrado."

    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
        live_pid_path.unlink(missing_ok=True)
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


def render_live_status_panel(
    project_root: Path,
    env_path: Path,
    live_pid_path: Path,
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
    live_pid = get_live_bot_pid(live_pid_path)

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
        events_path = project_root / events_path
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
    env_local = load_env(env_path)
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
