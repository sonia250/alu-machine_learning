# Forecasting Bitcoin One Hour Ahead with an RNN

![Bitcoin time-series forecasting](https://upload.wikimedia.org/wikipedia/commons/4/46/Bitcoin.svg)

## Introduction

Time-series forecasting uses historical observations ordered in time to estimate future values. Bitcoin is a challenging forecasting target because its market is noisy, non-stationary, and affected by news, liquidity, and changing market structure.

A stationary series has stable statistical properties such as its mean and variance over time. Raw Bitcoin prices are generally not stationary, so this project works with regularly sampled market features and scales them using statistics from the training period. Bitcoin also contains daily activity patterns and longer market cycles, which are forms of seasonality or cyclicity that an RNN can partially model.

## Preprocessing

The raw Coinbase and Bitstamp files contain one row per 60-second window. I removed malformed rows, converted numeric columns explicitly, combined duplicate timestamps, and resampled the observations from 60 seconds to one hour. Hourly aggregation uses the first open, maximum high, minimum low, last close, summed volumes, and mean VWAP.

Short missing intervals are interpolated. I retained open, high, low, close, BTC volume, USD volume, and VWAP. Each feature is standardized using only the first 80 percent of the timeline, preventing future information from leaking into training. The processed data is saved as a compressed NumPy archive.

## Dataset Setup

The model receives a rolling window of 24 hourly observations. For every window, the target is the close value in the following hour. The training and validation data are created with `tf.data.Dataset.from_tensor_slices`, shuffled only for training, batched, and prefetched.

## Model

The model uses two GRU layers followed by a dense ReLU layer and a single linear output. GRUs are useful here because their update and reset gates can retain longer-term information while remaining lighter than a large LSTM. The model is trained with Adam and mean-squared error, as required by the task.

## Results

Run the training script to generate `btc_forecast.png` and record the printed test MSE here:

```text
Test MSE: <record the value from your run>
```

The first graph shows training and validation MSE across epochs. The second graph compares the predicted and actual hourly close values on the held-out period. These plots should be included with the published version of this post.

## Conclusion

Bitcoin forecasting is useful as an experiment in feature engineering and sequence modeling, but a low historical MSE does not guarantee profitable trading. Market regimes change, transaction costs matter, and a model trained on past prices cannot reliably predict unexpected events. This project was valuable because it made the full workflow concrete: clean raw data, resample it to a meaningful time window, prevent leakage, feed sequences through `tf.data`, and evaluate on unseen time-ordered data.

Source code: https://github.com/sonia250/alu-machine_learning/tree/main/supervised_learning/time_series

To complete publication requirements, publish this draft on Medium or LinkedIn, add the generated graph, and paste both the article URL and LinkedIn share URL into the project submission.
