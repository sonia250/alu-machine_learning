# Bitcoin Time-Series Forecasting

This project preprocesses Coinbase and Bitstamp Bitcoin market data and trains an RNN to predict the next hourly closing price from the previous 24 hours.

## Workflow

1. Run `preprocess_data.py` on one or more raw CSV files.
2. Run `forecast_btc.py` on the generated `.npz` file.
3. The forecast script reports test MSE and saves training and prediction plots.

The preprocessing step removes malformed rows, combines duplicate timestamps, resamples 60-second records to hourly OHLCV-style features, interpolates short gaps, and scales features using training data statistics. The final feature sequence contains the previous 24 hourly rows; its target is the following hour's close price.

Example:

```bash
python preprocess_data.py coinbase.csv bitstamp.csv --output btc_hourly.npz
python forecast_btc.py btc_hourly.npz --epochs 20
```
