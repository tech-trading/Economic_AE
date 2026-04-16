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

    # Use atomic create to avoid race conditions when two launches happen at once.
    def _try_create_pid_file() -> bool:
        try:
            fd = os.open(str(PID_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        return True

    if _try_create_pid_file():
        return True

    try:
        old_pid = int(PID_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        # Another process may be creating/writing the lock right now.
        print("Live trader start already in progress. Exiting.")
        return False

    if _is_pid_running(old_pid):
        print(f"Live trader already running (PID {old_pid}). Exiting.")
        return False

    # Stale lock: remove and retry a final atomic create.
    try:
        PID_PATH.unlink(missing_ok=True)
    except Exception:
        pass
    if _try_create_pid_file():
        return True

    print("Live trader lock is busy. Exiting.")
    return False


def _release_singleton() -> None:
    try:
        if PID_PATH.exists():
            try:
                owner_pid = int(PID_PATH.read_text(encoding="utf-8").strip())
            except Exception:
                owner_pid = -1
            if owner_pid in {-1, os.getpid()}:
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
