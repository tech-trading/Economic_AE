# Economic Event Trading Bot for MT5

This project provides a production-ready starting point for an economic-news trading robot connected to MetaTrader 5 (MT5). It includes:

- Economic calendar ingestion
- Relevance filtering by impact, region, and symbol mapping
- Deep learning and classical model training
- Time-series-safe model evaluation
- Pre-event inference and order execution 10 seconds before the release
- Stop loss and trailing stop logic

## Important Risk and Reality Notes

- No model can know the true event result before release. The model can only estimate directional probability or volatility regime from market microstructure and historical event behavior.
- Trading high-impact events has slippage, spread widening, and execution risk.
- Backtest and paper-trade before real capital.
- Web scraping can break if website HTML changes and may be subject to website terms. Verify legal and compliance requirements.

## Architecture

1. Calendar Layer
- Pull events from Trading Economics API (or your preferred provider).
- Keep fields: event time, country, impact, previous, forecast, actual, and event id.

2. Market Data Layer
- Pull second-level data from MT5 around each event.
- Build pre-event windows (for example, last 300 seconds).

3. Feature Engineering
- Price/return momentum features
- Volatility and spread features
- Pre-event drift and microtrend features
- Time-to-event and event metadata embeddings

4. Modeling
- Baselines: Logistic Regression, RandomForest, GradientBoosting
- Deep model: LSTM classifier (direction), optional LSTM regressor (volatility)
- Ensemble voting and confidence score

5. Evaluation
- Walk-forward split and TimeSeriesSplit
- Classification metrics: F1, ROC-AUC, precision at high confidence
- Trading metrics: hit-rate, average R multiple, max drawdown, Sharpe proxy

6. Live Execution
- Every second check next high-impact event.
- Build features from current pre-event window.
- If now is event_time - 10 seconds and confidence threshold is met: place BUY/SELL.
- Attach stop loss and optional take profit.
- Manage trailing stop while position is open.

## Model Targets (suggested)

Use at least two targets in experiments:

1. Directional target
- Label 1 if post-event return in [t+5s, t+60s] is positive, else 0.

2. Volatility target
- Label 1 if realized volatility in [t+5s, t+60s] exceeds threshold, else 0.

Then combine:

- Direction model predicts BUY/SELL probability.
- Volatility model acts as gate (trade only if expected volatility is high enough).

## Evaluation Methods to Compare

1. TimeSeriesSplit cross-validation
2. Walk-forward validation by month
3. Rolling retrain (simulates production)
4. Regime-based holdout (high volatility periods)

## Quick Start

1. Create stable Python 3.11 environment and install deps:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
pip install tensorflow==2.16.1
```

This project defaults to `.venv311` to keep deep learning dependencies stable.

2. Fill credentials and API keys in `.env`.

3. Run diagnostics (checks .env, MT5 connection, and base files):

```powershell
python -m src.bootstrap
```

4. Train models:

```powershell
python -m src.data_collection
python -m src.train
```

5. Evaluate models:

```powershell
python -m src.evaluate
```

7. Run monthly walk-forward backtest:

```powershell
python -m src.walkforward_backtest
```

Generated reports:

- `models/walkforward_monthly_report.csv`
- `models/walkforward_summary.json`

Strict monthly mode (no weekly/sequential fallback):

1. Prepare long dataset (3-12 months):

```powershell
python -m src.prepare_monthly_dataset
```

2. Set in `.env`:

- `EVENTS_CSV=data/events_monthly.csv`
- `MARKET_CSV=data/market_ticks_monthly.csv`
- `STRICT_MONTHLY_VALIDATION=true`
- `DIRECTION_LABEL_MODE=quantile_monthly` (recommended for monthly synthetic anchors)

3. Run backtest again:

```powershell
python -m src.walkforward_backtest
```

4. Inspect class balance by month:

```powershell
python -m src.dataset_diagnostics
```

Output: `models/dataset_monthly_diagnostics.csv`

8. Run live trader:

```powershell
python -m src.main
```

## Windows Desktop Shortcut with Custom Icon

Create a desktop shortcut that launches the production app and applies a custom icon:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\create_desktop_shortcut.ps1
```

Optional parameters:

- `-ShortcutName "Economic AE Live"`: custom shortcut name on desktop.
- `-Force`: regenerate the icon file if you replaced the source image.

## Without Trading Economics API Key

If `TE_API_KEY` is not configured, the bot uses a scraping fallback from Trading Economics calendar page.

Daily snapshot at 00:00 UTC (events + MT5 historical bars):

```powershell
python -m src.daily_jobs
```

Timezone behavior:

- Set `UTC_OFFSET_HOURS` in `.env` (example: `-5`).
- The daily scheduler runs at 00:00 of that timezone.
- Scraped calendar times are interpreted in that timezone and converted to UTC internally.

This process creates/updates:

- `data/events.csv` (calendar events)
- `data/mt5_history_m1.csv` (1-minute MT5 historical bars)

## Anti-Overfitting and Risk Controls

- Balanced class weighting is applied during training.
- Decision policy is optimized after training (`models/trading_policy.json`) by tuning threshold and no-trade band.
- Evaluation and live trading both consume the saved decision policy.

## Paper Trading Mode

- Set `PAPER_TRADING=true` in `.env` to avoid real orders.
- Signals are logged into `data/paper_trades.csv`.

## Fundamental LLM Strategy (News-Driven)

- Set `STRATEGY=fundamental_llm` to enable a fundamental agent that reads recent economic headlines (RSS sources) and asks an LLM to decide `BUY`, `SELL`, or `HOLD`.
- Works with the configured `SYMBOL` across asset classes (forex, commodities, indices/futures, equities), as long as the symbol exists in your broker/MT5.
- Main env vars:
	- `FUNDAMENTAL_NEWS_SOURCES`
	- `FUNDAMENTAL_NEWS_LOOKBACK_MINUTES`
	- `FUNDAMENTAL_NEWS_POLL_SECONDS` (cada cuánto refrescar RSS)
	- `FUNDAMENTAL_MIN_CONFIDENCE`
	- `FUNDAMENTAL_DECISION_THRESHOLD` (override específico para fundamental)
	- `FUNDAMENTAL_REANALYZE_SECONDS` (cache corto para evitar re-llamadas iguales)
	- `FUNDAMENTAL_ALLOW_SAME_SIDE_ON_NEWS_CHANGE` (permite repetir lado si cambió la noticia)
	- `FUNDAMENTAL_LLM_API_BASE_URL`
	- `FUNDAMENTAL_LLM_API_KEY`
	- `FUNDAMENTAL_LLM_MODEL`

## Dynamic Symbol Routing by Impact News

- The live trader can route orders to different symbols based on impact event currency and event-name keywords.
- This allows trading beyond one fixed forex pair (for example, indices, commodities, or other instruments available in MT5).
- Main env vars:
	- `IMPACT_SYMBOL_MAP` (currency to candidate symbols)
	- `IMPACT_KEYWORD_SYMBOL_MAP` (news keyword to candidate symbols)
	- `IMPACT_SYMBOL_FALLBACK_TO_DEFAULT` (fallback to `SYMBOL` when no route matches)

Example:

```env
IMPACT_SYMBOL_MAP=USD=EURUSD|US500|XAUUSD;EUR=EURUSD|GER40;GBP=GBPUSD|UK100;JPY=USDJPY|JP225
IMPACT_KEYWORD_SYMBOL_MAP=oil=XTIUSD|WTI;gold=XAUUSD;nasdaq=NAS100|USTEC;sp500=US500|SPX500
IMPACT_SYMBOL_FALLBACK_TO_DEFAULT=true
```

## Project Layout

- `src/config.py`: environment and runtime settings
- `src/calendar_sources.py`: calendar provider adapter
- `src/feature_engineering.py`: dataset and feature generation
- `src/models.py`: baseline + deep models + persistence
- `src/train.py`: train pipeline
- `src/evaluate.py`: offline evaluation and plots
- `src/mt5_executor.py`: MT5 connectivity and order management
- `src/live_trader.py`: event scheduler + prediction + execution
- `src/main.py`: application entrypoint

## Next Improvements

- Add transformer-based sequence model
- Add Bayesian uncertainty and no-trade band
- Add latency-aware execution simulation
- Add tick-level slippage model
- Add broker-specific news spread filter
