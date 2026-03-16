from src.config import settings
import pandas as pd
from src.feature_engineering import build_event_dataset

print('signals file:', settings.data_dir+'/backtest_signals_default.csv')
try:
    df = pd.read_csv(settings.data_dir+'/backtest_signals_default.csv')
    print('signals head:\n', df.head())
except Exception as e:
    print('no default file:', e)

events = pd.read_csv(settings.events_csv)
try:
    ticks = pd.read_csv(settings.market_csv, parse_dates=['time_utc'])
    ticks['time_utc'] = pd.to_datetime(ticks['time_utc'], utc=True)
except Exception as e:
    print('load ticks error', e)
    ticks = pd.DataFrame()

bundle = build_event_dataset(events, ticks, lookback_seconds=settings.lookback_seconds)
print('bundle sizes', getattr(bundle.X_tabular, 'shape', None))
print('ret_post sample (first 20):', bundle.ret_post[:20])
print('event_ids sample (first 20):', bundle.event_ids[:20])
if not ticks.empty:
    first_et = bundle.event_times.iloc[0]
    print('first event_time:', first_et)
    t1 = first_et + pd.Timedelta(seconds=5)
    t2 = first_et + pd.Timedelta(seconds=300)
    times = ticks['time_utc']
    idx1 = int(times.searchsorted(t1, side='left'))
    idx2 = int(times.searchsorted(t2, side='right')) - 1
    print('t1,t2:', t1, t2)
    print('idx1, idx2:', idx1, idx2, 'len ticks', len(ticks))
    lo = max(0, idx1-3)
    hi = min(len(ticks), idx1+4)
    print('ticks around idx1:')
    print(ticks.iloc[lo:hi][['time_utc','bid','ask']])
    lo2 = max(0, idx2-3)
    hi2 = min(len(ticks), idx2+4)
    print('ticks around idx2:')
    print(ticks.iloc[lo2:hi2][['time_utc','bid','ask']])
