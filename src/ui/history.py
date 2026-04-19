from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config import settings
from src.ui.common import load_csv, parse_datetime_utc
from src.ui.env import load_env, parse_bool, parse_float
from src.ui.live_ops import load_live_mt5_trades


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


def render_trade_history_tab(project_root: Path, env_path: Path) -> None:
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

    paper_path = project_root / "data/paper_trades.csv"
    trades = load_csv(paper_path)
    if trades.empty:
        st.info(
            "No hay histórico aún. El registro disponible en la UI se construye con data/paper_trades.csv "
            "(pipeline de observabilidad)."
        )
        return

    market_path = project_root / settings.market_csv
    enriched = enrich_trade_history_with_results(trades, market_path=market_path)

    env_local = load_env(env_path)
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
