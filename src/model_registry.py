from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


def list_snapshots(registry_dir: str = "models_registry") -> list[str]:
    p = Path(registry_dir)
    if not p.exists():
        return []
    return sorted([x.name for x in p.iterdir() if x.is_dir()], reverse=True)


def snapshot_current_models(model_dir: str = "models", registry_dir: str = "models_registry", name: str | None = None) -> str:
    src = Path(model_dir)
    if not src.exists():
        raise RuntimeError(f"Model dir not found: {model_dir}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snap_name = name or f"snapshot_{timestamp}"

    dst = Path(registry_dir) / snap_name
    dst.mkdir(parents=True, exist_ok=False)

    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)

    return snap_name


def restore_snapshot(snapshot_name: str, model_dir: str = "models", registry_dir: str = "models_registry") -> None:
    src = Path(registry_dir) / snapshot_name
    if not src.exists() or not src.is_dir():
        raise RuntimeError(f"Snapshot not found: {snapshot_name}")

    dst = Path(model_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)
