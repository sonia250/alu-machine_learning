#!/usr/bin/env python3
"""Train an RNN that forecasts the next hourly Bitcoin close."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


WINDOW = 24
CLOSE_COLUMN = 3


def make_sequences(values):
    """Build 24-hour inputs and next-hour close targets."""
    inputs = np.array(
        [values[index - WINDOW:index] for index in range(WINDOW, len(values))],
        dtype=np.float32,
    )
    targets = values[WINDOW:, CLOSE_COLUMN].astype(np.float32)
    return inputs, targets


def build_dataset(inputs, targets, batch_size, shuffle):
    """Create a tf.data.Dataset for model training or evaluation."""
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(min(len(inputs), 10000), seed=7)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(features):
    """Create a compact GRU regression model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(WINDOW, features)),
            tf.keras.layers.GRU(64, return_sequences=True),
            tf.keras.layers.GRU(32),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model


def forecast(path, epochs, batch_size, plot_path):
    """Train, evaluate, and plot forecasts from a preprocessed file."""
    data = np.load(path)
    inputs, targets = make_sequences(data["values"])
    split = max(1, int(len(inputs) * 0.8))
    train_x, test_x = inputs[:split], inputs[split:]
    train_y, test_y = targets[:split], targets[split:]
    train = build_dataset(train_x, train_y, batch_size, True)
    test = build_dataset(test_x, test_y, batch_size, False)
    model = build_model(inputs.shape[2])
    history = model.fit(train, validation_data=test, epochs=epochs, verbose=1)
    predictions = model.predict(test, verbose=0).reshape(-1)
    mse = float(np.mean((predictions - test_y) ** 2))
    print(f"Test MSE: {mse:.6f}")
    figure, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[1].plot(test_y, label="actual")
    axes[1].plot(predictions, label="predicted")
    axes[1].set_title("Hourly close forecast")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(plot_path)
    return model, history, mse


def main():
    """Parse arguments and train the forecasting model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Preprocessed .npz file")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--plot", default="btc_forecast.png")
    args = parser.parse_args()
    forecast(args.data, args.epochs, args.batch_size, args.plot)


if __name__ == "__main__":
    main()
