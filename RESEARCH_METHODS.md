# Research Guide: Prediction and Evaluation Methods

This document proposes a robust research framework for economic-event trading.

## 1) Prediction Problem Definitions

Use multiple targets in parallel:

1. Direction classification
- Predict sign of return in post-event window [5s, 60s].

2. Volatility classification
- Predict if realized volatility exceeds threshold.

3. Magnitude regression
- Predict absolute return magnitude in [5s, 60s].

4. Meta-labeling
- Predict if a base signal should be traded (confidence gating).

## 2) Feature Sets

1. Price microstructure
- return moments, realized volatility, spread moments, bid/ask imbalance.

2. Event context
- importance, country/currency, event type, time-of-day, day-of-week.

3. Forecast surprise (when available historically)
- normalized surprise: (actual - forecast) / rolling sigma.

4. Regime indicators
- ATR regime, pre-event volatility percentile, spread percentile.

## 3) Model Families to Compare

1. Classical baselines
- Logistic Regression, RandomForest, GradientBoosting.

2. Sequence deep learning
- LSTM/GRU with second-level sequence windows.

3. Temporal CNN
- 1D Conv model for local pre-event patterns.

4. Transformer-lite
- Small attention model for long lookback windows.

5. Stacked ensemble
- Blend baseline + deep model probabilities.

## 4) Validation Protocol

Use only time-respecting validation.

1. TimeSeriesSplit CV
2. Monthly walk-forward backtest
3. Rolling retraining every N events
4. Out-of-regime holdout periods

## 5) Metrics

1. Statistical metrics
- ROC-AUC, F1, precision@confidence.

2. Trading metrics
- hit rate, expectancy in R, max drawdown, average trade duration.

3. Execution-aware metrics
- slippage-adjusted expectancy, spread-adjusted net PnL.

## 6) Decision Policy

Trade only when all gates pass:

1. event importance >= threshold
2. predicted volatility high
3. confidence >= threshold (for example 0.60)
4. spread <= max_spread_points

Direction:
- BUY when p_buy >= 0.5
- SELL when p_buy < 0.5

## 7) Ablation Studies

Run and compare:

1. no event metadata
2. no spread features
3. no volatility gate
4. no trailing stop
5. 10s vs 5s vs 15s pre-event trigger

## 8) Deployment Hardening

1. Add circuit breaker: max daily loss and max trades/day.
2. Add fallback no-trade mode when data missing.
3. Log every prediction and order request/response.
4. Add watchdog and process restart strategy.
5. Add paper-trading mode before production.
