from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
LIVE_ACTIVITY_PATH = PROJECT_ROOT / "data" / "live_activity.csv"
DAILY_REPORT_PATH = PROJECT_ROOT / "models" / "daily_live_report.json"

TUNABLE_KEYS = [
    "AGENTIC_DECISION_THRESHOLD",
    "AGENTIC_MIN_CONFIDENCE",
    "AGENTIC_MIN_FALLBACK_CONFIDENCE",
    "AGENTIC_DYNAMIC_THRESHOLD_FLOOR",
    "AGENTIC_DYNAMIC_THRESHOLD_CAP",
    "AGENTIC_MAX_SPREAD_PIPS",
    "AGENTIC_SIGNAL_COOLDOWN_SECONDS",
    "MIN_SECONDS_BETWEEN_TRADES",
    "SAME_SIDE_COOLDOWN_SECONDS",
    "AGENTIC_REQUIRE_AGENT_AGREEMENT",
    "AGENTIC_USE_FUNDAMENTAL_FALLBACK",
]


def _read_env_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontro {path}")
    return path.read_text(encoding="utf-8").splitlines()


def _parse_env(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in lines:
        txt = line.strip()
        if not txt or txt.startswith("#") or "=" not in txt:
            continue
        k, v = txt.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _to_float(env: dict[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, default))
    except Exception:
        return float(default)


def _to_int(env: dict[str, str], key: str, default: int) -> int:
    try:
        return int(float(env.get(key, default)))
    except Exception:
        return int(default)


def _to_bool(env: dict[str, str], key: str, default: bool) -> bool:
    val = str(env.get(key, str(default))).strip().lower()
    return val in {"1", "true", "yes", "y"}


def _load_activity(hours: int) -> pd.DataFrame:
    if not LIVE_ACTIVITY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(LIVE_ACTIVITY_PATH)
    if df.empty or "time_utc" not in df.columns:
        return pd.DataFrame()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc"]).sort_values("time_utc")
    since = datetime.now(timezone.utc) - timedelta(hours=max(1, int(hours)))
    return df[df["time_utc"] >= pd.Timestamp(since)].copy()


def _load_report() -> dict[str, Any]:
    if not DAILY_REPORT_PATH.exists():
        return {}
    try:
        return json.loads(DAILY_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _extract_max_decisions_from_detail(activity: pd.DataFrame) -> int:
    if activity.empty or "detail" not in activity.columns:
        return 0
    max_decisions = 0
    for txt in activity["detail"].astype(str).tolist():
        m = re.search(r"decisions=(\d+)", txt)
        if not m:
            continue
        max_decisions = max(max_decisions, int(m.group(1)))
    return int(max_decisions)


def _build_metrics(hours: int) -> dict[str, float]:
    activity = _load_activity(hours)
    report = _load_report()

    rows = int(len(activity))
    action_counts: dict[str, int] = {}
    if rows > 0 and "action" in activity.columns:
        action_counts = {str(k): int(v) for k, v in activity["action"].value_counts().to_dict().items()}

    no_decision = int(action_counts.get("eventless_no_decision", 0))
    sent = int(action_counts.get("order_sent", 0) + action_counts.get("order_sent_eventless", 0) + action_counts.get("paper_signal", 0) + action_counts.get("paper_signal_eventless", 0))
    max_open_skips = int(action_counts.get("skip_max_open_positions", 0) + action_counts.get("skip_eval_max_open_positions_eventless", 0))

    mt5 = report.get("mt5_performance", {}) if isinstance(report, dict) else {}
    pf = float(mt5.get("profit_factor", 0.0)) if isinstance(mt5, dict) else 0.0
    win_rate = float(mt5.get("win_rate", 0.0)) if isinstance(mt5, dict) else 0.0
    net_profit = float(mt5.get("net_profit", 0.0)) if isinstance(mt5, dict) else 0.0
    max_dd = float(mt5.get("max_drawdown_profit", 0.0)) if isinstance(mt5, dict) else 0.0
    closed_deals = int(mt5.get("closed_deals", 0)) if isinstance(mt5, dict) else 0

    max_decisions = _extract_max_decisions_from_detail(activity)

    return {
        "rows": float(rows),
        "no_decision_rate": _safe_ratio(no_decision, rows),
        "signal_rate": _safe_ratio(sent, rows),
        "max_open_skip_rate": _safe_ratio(max_open_skips, rows),
        "profit_factor": pf,
        "win_rate": win_rate,
        "net_profit": net_profit,
        "max_drawdown_profit": max_dd,
        "closed_deals": float(closed_deals),
        "max_agent_decisions": float(max_decisions),
    }


def _recommend(env: dict[str, str], m: dict[str, float]) -> dict[str, str]:
    decision_thr = _to_float(env, "AGENTIC_DECISION_THRESHOLD", 0.58)
    min_conf = _to_float(env, "AGENTIC_MIN_CONFIDENCE", 0.60)
    min_fallback = _to_float(env, "AGENTIC_MIN_FALLBACK_CONFIDENCE", 0.55)
    floor_thr = _to_float(env, "AGENTIC_DYNAMIC_THRESHOLD_FLOOR", 0.54)
    cap_thr = _to_float(env, "AGENTIC_DYNAMIC_THRESHOLD_CAP", 0.72)
    max_spread = _to_float(env, "AGENTIC_MAX_SPREAD_PIPS", 2.2)
    sig_cd = _to_int(env, "AGENTIC_SIGNAL_COOLDOWN_SECONDS", 180)
    min_gap = _to_int(env, "MIN_SECONDS_BETWEEN_TRADES", 120)
    same_side_cd = _to_int(env, "SAME_SIDE_COOLDOWN_SECONDS", 240)
    require_agreement = _to_bool(env, "AGENTIC_REQUIRE_AGENT_AGREEMENT", False)
    use_fund = _to_bool(env, "AGENTIC_USE_FUNDAMENTAL_FALLBACK", True)

    no_decision_rate = float(m.get("no_decision_rate", 0.0))
    pf = float(m.get("profit_factor", 0.0))
    wr = float(m.get("win_rate", 0.0))
    net = float(m.get("net_profit", 0.0))
    dd = float(m.get("max_drawdown_profit", 0.0))
    max_decisions = int(m.get("max_agent_decisions", 0.0))
    closed_deals = int(m.get("closed_deals", 0.0))

    # Regimen A: casi no hay decisiones -> abrir ligeramente la puerta.
    if no_decision_rate > 0.80 or max_decisions <= 2:
        decision_thr = max(0.54, decision_thr - 0.02)
        min_conf = max(0.56, min_conf - 0.02)
        min_fallback = max(0.52, min_fallback - 0.01)
        floor_thr = max(0.52, floor_thr - 0.01)
        sig_cd = max(90, int(round(sig_cd * 0.85)))
        min_gap = max(90, int(round(min_gap * 0.85)))
        same_side_cd = max(180, int(round(same_side_cd * 0.85)))
        require_agreement = False
        use_fund = True

    # Regimen B: hay operaciones suficientes y rendimiento flojo -> endurecer calidad.
    if closed_deals >= 8 and (pf < 1.0 or net <= 0):
        decision_thr = min(0.72, decision_thr + 0.02)
        min_conf = min(0.70, min_conf + 0.02)
        min_fallback = min(0.60, min_fallback + 0.01)
        floor_thr = min(0.60, floor_thr + 0.01)
        max_spread = max(1.4, max_spread - 0.2)

    # Regimen C: drawdown desproporcionado -> más selectivo.
    if dd > 0 and abs(net) > 0 and dd > abs(net) * 2.0:
        decision_thr = min(0.74, decision_thr + 0.02)
        min_conf = min(0.72, min_conf + 0.02)
        max_spread = max(1.3, max_spread - 0.2)

    # Regimen D: rendimiento sano -> mantener y estabilizar.
    if closed_deals >= 6 and pf >= 1.25 and wr >= 0.50 and net > 0:
        decision_thr = max(0.56, decision_thr)
        min_conf = max(0.58, min_conf)
        min_fallback = max(0.53, min_fallback)

    cap_thr = max(decision_thr + 0.08, cap_thr)
    cap_thr = min(0.80, cap_thr)

    rec = {
        "AGENTIC_DECISION_THRESHOLD": f"{decision_thr:.2f}",
        "AGENTIC_MIN_CONFIDENCE": f"{min_conf:.2f}",
        "AGENTIC_MIN_FALLBACK_CONFIDENCE": f"{min_fallback:.2f}",
        "AGENTIC_DYNAMIC_THRESHOLD_FLOOR": f"{floor_thr:.2f}",
        "AGENTIC_DYNAMIC_THRESHOLD_CAP": f"{cap_thr:.2f}",
        "AGENTIC_MAX_SPREAD_PIPS": f"{max_spread:.1f}",
        "AGENTIC_SIGNAL_COOLDOWN_SECONDS": str(sig_cd),
        "MIN_SECONDS_BETWEEN_TRADES": str(min_gap),
        "SAME_SIDE_COOLDOWN_SECONDS": str(same_side_cd),
        "AGENTIC_REQUIRE_AGENT_AGREEMENT": "true" if require_agreement else "false",
        "AGENTIC_USE_FUNDAMENTAL_FALLBACK": "true" if use_fund else "false",
    }
    return rec


def _apply_updates(lines: list[str], updates: dict[str, str]) -> list[str]:
    used = set()
    out: list[str] = []
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in line:
            out.append(line)
            continue

        k = line.split("=", 1)[0].strip()
        if k in updates:
            out.append(f"{k}={updates[k]}")
            used.add(k)
        else:
            out.append(line)

    for k in TUNABLE_KEYS:
        if k in updates and k not in used:
            out.append(f"{k}={updates[k]}")
    return out


def _print_block(title: str, data: dict[str, Any]) -> None:
    print(f"\n[{title}]")
    for k, v in data.items():
        print(f"- {k}: {v}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-optimiza parametros de agentic en .env usando metricas recientes.")
    parser.add_argument("--hours", type=int, default=24, help="Ventana de analisis en horas.")
    parser.add_argument("--apply", action="store_true", help="Aplica cambios a .env.")
    args = parser.parse_args()

    lines = _read_env_lines(ENV_PATH)
    env = _parse_env(lines)
    metrics = _build_metrics(hours=args.hours)
    rec = _recommend(env, metrics)

    before = {k: env.get(k, "") for k in TUNABLE_KEYS}
    after = {k: rec.get(k, env.get(k, "")) for k in TUNABLE_KEYS}

    _print_block("METRICAS", metrics)
    _print_block("ANTES", before)
    _print_block("RECOMENDADO", after)

    if not args.apply:
        print("\nModo simulacion: no se aplicaron cambios. Usa --apply para escribir en .env")
        return 0

    backup_name = f".env.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_path = PROJECT_ROOT / "models" / backup_name
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    updated_lines = _apply_updates(lines, after)
    ENV_PATH.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

    print(f"\nCambios aplicados en {ENV_PATH}")
    print(f"Backup creado en {backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
