# AGENTS

## Repo Reality Checks
- This is a single Python package (no monorepo): main code in `src/`, automation scripts in `scripts/`, one unit test in `tests/`.
- There is no CI, lint, formatter, typecheck, or pre-commit config in the repo; do not assume `ruff`, `mypy`, or workflow gates exist.
- Git LFS tracks model/binary artifacts (`*.h5`, `*.npz`, `*.joblib`, etc.) via `.gitattributes`; avoid rewriting large tracked artifacts unless explicitly requested.

## Environment + Interpreter Gotchas
- `src/config.py` calls `load_dotenv(override=True)`: values from `.env` override already-exported env vars.
- Virtualenv usage is inconsistent across scripts:
  - `scripts/create_unified_venv.ps1`, `scripts/start_production_ui.ps1`, `scripts/auto_optimize_cycle.ps1` expect `.venv`.
  - `scripts/run_daily_snapshot.ps1` hardcodes `.venv311`.
- Before running automation scripts, verify which interpreter path that script expects; this repo does not have one fully unified venv convention yet.

## High-Value Commands (verified from code)
- Bootstrap diagnostics: `python -m src.bootstrap`
- Collect events/ticks for training: `python -m src.data_collection`
- Train + save models/policy: `python -m src.train`
- Evaluate saved models: `python -m src.evaluate`
- Walk-forward backtest: `python -m src.walkforward_backtest`
- Prepare strict monthly dataset (MT5 M1-derived): `python -m src.prepare_monthly_dataset`
- Live trader entrypoint: `python -m src.main`
- Streamlit UI: `python -m streamlit run src/ui_app.py`
- Single test: `python -m unittest tests.test_driven_trading_agentic_system`

## Execution Order That Matters
- Offline pipeline order is enforced by dependencies: `bootstrap -> data_collection -> train -> evaluate`.
- `src.evaluate` and `src.live_trader` require artifacts already saved in `models/` by training.
- `src.walkforward_backtest` retrains per split internally; it does not use saved model artifacts.

## Live Trading Behavior You Must Not Miss
- `src.main` enforces singleton execution with `logs/live_bot.pid`; a stale PID file can block startup.
- Default mode is paper trading (`PAPER_TRADING=true` by default in `Settings`); live orders require explicitly disabling paper mode.
- `scripts/start_production_ui.ps1` forces live mode by setting `PAPER_TRADING=false` before launching `src.main`.
- `LiveTrader` logs runtime activity to `data/live_activity.csv`; paper signals append to `data/paper_trades.csv`.

## Strategy Wiring / Naming
- Strategy selection is string-based in `src.strategies.get_strategy`; aliases map to concrete implementations.
- `AGENT_MANAGE_ALL_STRATEGIES=true` (default) wraps any selected strategy in `AgentManagedStrategy` for telemetry/status; do not assume raw strategy instances.
- Eventless strategies (e.g., `ema_rsi`, `turtle_atr`, `fundamental_llm`, `driven_trading_agentic_system`) run continuously without waiting for calendar trigger windows.

## Data + State Artifacts
- Primary inputs default to `data/events.csv` and `data/market_ticks.csv` (overridable in `.env`).
- Strict monthly validation requires `STRICT_MONTHLY_VALIDATION=true` and monthly files (`data/events_monthly.csv`, `data/market_ticks_monthly.csv`), otherwise walk-forward may fallback to weekly/sequential splits.
- Adaptive strategies persist state under `models/` (for example `models/agentic_state.json`, `models/driven_agentic_state.json`); keep these when behavior continuity matters.

## MT5 and External Dependencies
- Many core flows require a working local MetaTrader 5 terminal/API (`MetaTrader5` Python package and terminal access).
- If `TE_API_KEY` is missing, calendar ingestion falls back to web scraping in `src/calendar_sources.py`; expect fragility and slower behavior.
