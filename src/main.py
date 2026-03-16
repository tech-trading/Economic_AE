from __future__ import annotations

from src.live_trader import LiveTrader


def main() -> None:
    trader = LiveTrader()
    trader.run()


if __name__ == "__main__":
    main()
