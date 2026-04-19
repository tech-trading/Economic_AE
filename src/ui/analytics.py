from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config import settings
from src.ui.common import load_csv, parse_datetime_utc
from src.ui.live_ops import load_live_mt5_trades


def build_monitor_source(
    project_root: Path,
    environment: str,
    history_days: int,
) -> tuple[pd.DataFrame, str | None, str]:
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

    paper_path = project_root / "data/paper_trades.csv"
    return load_csv(paper_path), None, "data/paper_trades.csv"


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
