from __future__ import annotations

import os
import subprocess
from pathlib import Path

from src.live_trader import LiveTrader


PID_PATH = Path("logs/live_bot.pid")


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
    except OSError:
        return False


def _acquire_singleton() -> bool:
    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PID_PATH.exists():
        try:
            old_pid = int(PID_PATH.read_text(encoding="utf-8").strip())
        except Exception:
            old_pid = -1
        if _is_pid_running(old_pid):
            print(f"Live trader already running (PID {old_pid}). Exiting.")
            return False

    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _release_singleton() -> None:
    try:
        if PID_PATH.exists():
            PID_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def main() -> None:
    if not _acquire_singleton():
        return

    trader = LiveTrader()
    try:
        trader.run()
    finally:
        _release_singleton()


if __name__ == "__main__":
    main()
