#!/usr/bin/env python3
"""Prepare Bitcoin exchange data for hourly RNN forecasting."""

import argparse
import os

import numpy as np
import pandas as pd


COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "btc_volume",
    "usd_volume",
    "vwap",
]
FEATURES = ["open", "high", "low", "close", "btc_volume", "usd_volume", "vwap"]


def load_exchange(path):
    """Load and clean one raw exchange CSV file."""
    frame = pd.read_csv(path, header=None, names=COLUMNS)
    for column in COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "close"])
    frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    return frame.set_index("datetime")[FEATURES]


def preprocess(paths, output):
    """Aggregate exchange data hourly and save a compressed dataset."""
    frames = [load_exchange(path) for path in paths]
    data = pd.concat(frames).sort_index()
    data = data[~data.index.duplicated(keep="last")]
    hourly = data.resample("1h").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "btc_volume": "sum",
            "usd_volume": "sum",
            "vwap": "mean",
        }
    )
    hourly = hourly.interpolate(limit_direction="both").dropna()
    values = hourly.to_numpy(dtype=np.float32)
    split = max(1, int(values.shape[0] * 0.8))
    mean = values[:split].mean(axis=0).astype(np.float32)
    scale = values[:split].std(axis=0).astype(np.float32)
    scale[scale == 0] = 1
    scaled = (values - mean) / scale
    np.savez_compressed(
        output,
        values=scaled,
        close_scale=scale[3],
        close_mean=mean[3],
        timestamps=hourly.index.view("int64"),
    )
    return output


def main():
    """Parse arguments and preprocess the requested files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Raw exchange CSV files")
    parser.add_argument("--output", default="btc_hourly.npz")
    args = parser.parse_args()
    for path in args.inputs:
        if not os.path.isfile(path):
            parser.error(f"input file does not exist: {path}")
    preprocess(args.inputs, args.output)
    print(f"Saved preprocessed data to {args.output}")


if __name__ == "__main__":
    main()
