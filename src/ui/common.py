from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run_module(project_root: Path, module: str, extra_env: dict[str, str] | None = None) -> tuple[int, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    cmd = [sys.executable, "-m", module]
    proc = subprocess.run(cmd, cwd=str(project_root), env=env, capture_output=True, text=True)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, output.strip()


def run_script(
    project_root: Path,
    script_rel_path: str,
    args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    cmd = [sys.executable, script_rel_path]
    if args:
        cmd.extend(args)
    proc = subprocess.run(cmd, cwd=str(project_root), env=env, capture_output=True, text=True)
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
