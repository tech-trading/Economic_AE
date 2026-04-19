from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values


def load_env(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}
    data = dotenv_values(str(env_path))
    return {str(k): str(v) for k, v in data.items() if k is not None and v is not None}


def save_env(env_path: Path, values: dict[str, str]) -> None:
    lines = [f"{k}={v}" for k, v in values.items()]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
