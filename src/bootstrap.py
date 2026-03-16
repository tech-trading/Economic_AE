from __future__ import annotations

import os
from pathlib import Path

import MetaTrader5 as mt5

from src.calendar_sources import fetch_and_store_events
from src.config import settings


def check_env() -> list[str]:
    issues: list[str] = []

    te_key = (settings.te_api_key or "").strip()
    if (not te_key) or te_key.upper().startswith("YOUR_"):
        issues.append("TE_API_KEY no configurada: se usara scraping web como fallback.")

    mt5_password = (settings.mt5_password or "").strip()
    mt5_server = (settings.mt5_server or "").strip()
    if settings.mt5_login <= 0 or (not mt5_password) or mt5_password.startswith("your_") or (not mt5_server):
        issues.append(
            "Faltan credenciales completas de MT5 en .env (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)."
        )

    return issues


def check_mt5_connection() -> str:
    ok = False
    try:
        if settings.mt5_login > 0 and settings.mt5_password and settings.mt5_server:
            ok = mt5.initialize(
                login=settings.mt5_login,
                password=settings.mt5_password,
                server=settings.mt5_server,
            )
        else:
            ok = mt5.initialize()

        if not ok:
            return f"MT5 no conectado: {mt5.last_error()}"

        account = mt5.account_info()
        if account is None:
            return "MT5 conectado, pero no hay cuenta activa."

        return f"MT5 conectado. Login={account.login}, Server={account.server}"
    finally:
        if ok:
            mt5.shutdown()


def ensure_data_dirs() -> None:
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.model_dir).mkdir(parents=True, exist_ok=True)


def check_data_files() -> list[str]:
    missing = []
    if not Path(settings.events_csv).exists():
        missing.append(settings.events_csv)
    if not Path(settings.market_csv).exists():
        missing.append(settings.market_csv)
    return missing


def main() -> None:
    print("=== Diagnostico del Bot de Trading ===")
    ensure_data_dirs()

    issues = check_env()
    if issues:
        print("Configuracion pendiente:")
        for item in issues:
            print(f"- {item}")
    else:
        print("Configuracion .env completa.")

    print("\nChequeando MT5...")
    print(check_mt5_connection())

    te_key = (settings.te_api_key or "").strip()
    print("\nIntentando descarga de eventos (API o scraping fallback)...")
    try:
        events = fetch_and_store_events(days_ahead=14)
        print(f"Eventos relevantes guardados: {len(events)} en {settings.events_csv}")
    except Exception as ex:
        print(f"No se pudo descargar calendario: {ex}")

    missing = check_data_files()
    print("\nChequeo de archivos:")
    if not missing:
        print("- OK: existen events.csv y market_ticks.csv")
    else:
        for path in missing:
            print(f"- Falta: {path}")

    print("\nOrden recomendado de ejecucion:")
    print("1) python -m src.bootstrap")
    print("2) python -m src.data_collection")
    print("3) python -m src.train")
    print("4) python -m src.evaluate")
    print("5) python -m src.main")


if __name__ == "__main__":
    main()
