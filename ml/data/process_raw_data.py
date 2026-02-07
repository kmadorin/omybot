"""
Process raw Uniswap V4 swap event data into training-ready features.

Takes raw swap CSV (from collect_raw_data.py) and produces 1-minute OHLCV bars
with rolling volatility and volume features.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Add agent directory to path for config import
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "agent")
)
import config


def load_raw_data(input_path):
    """Load and validate raw swap CSV."""
    df = pd.read_csv(input_path)

    required = [
        "timestamp", "sqrtPriceX96", "tick", "liquidity",
        "amount0", "amount1", "fee",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in raw data: {missing}")

    # Convert numeric columns from strings
    for col in ["sqrtPriceX96", "liquidity", "amount0", "amount1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["tick"] = pd.to_numeric(df["tick"], errors="coerce").astype("Int64")
    df["fee"] = pd.to_numeric(df["fee"], errors="coerce").astype("Int64")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["timestamp", "sqrtPriceX96"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Loaded {len(df)} raw swap events")
    return df


def compute_price(sqrt_price_x96):
    """
    Convert sqrtPriceX96 to human-readable ETH price in USDC.

    price = (sqrtPriceX96 / 2^96)^2 * 10^12
    The 10^12 factor adjusts for decimal difference: ETH(18) - USDC(6) = 12
    """
    ratio = sqrt_price_x96 / (2 ** 96)
    price = ratio ** 2 * (10 ** (config.ETH_DECIMALS - config.USDC_DECIMALS))
    return price


def compute_volume(amount0):
    """Volume in ETH terms: abs(amount0) / 10^18."""
    return np.abs(amount0) / (10 ** config.ETH_DECIMALS)


def process_raw_to_features(df):
    """
    Process raw swap events into 1-minute OHLCV bars with rolling features.

    Output columns:
        timestamp, price, tick, volume, liquidity, sqrtPriceX96,
        volatility_5m, volatility_1h, volume_ma_5m, price_returns
    """
    # Compute price and volume per swap
    df["price"] = compute_price(df["sqrtPriceX96"].values)
    df["volume"] = compute_volume(df["amount0"].values)

    # Convert timestamp to datetime index for resampling
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("datetime")

    # Resample to 1-minute OHLCV bars
    ohlcv = pd.DataFrame()
    ohlcv["price"] = df["price"].resample("1min").last()
    ohlcv["tick"] = df["tick"].resample("1min").last()
    ohlcv["volume"] = df["volume"].resample("1min").sum()
    ohlcv["liquidity"] = df["liquidity"].resample("1min").last()
    ohlcv["sqrtPriceX96"] = df["sqrtPriceX96"].resample("1min").last()

    # Forward fill gaps (minutes with no swaps)
    ohlcv = ohlcv.ffill()

    # Drop rows that are still NaN (leading edge before first data)
    ohlcv = ohlcv.dropna(subset=["price"])

    # Fill volume with 0 for minutes with no swaps (after ffill of price)
    # Volume should be 0 if no swaps happened, not forward-filled
    volume_raw = df["volume"].resample("1min").sum()
    ohlcv["volume"] = volume_raw.reindex(ohlcv.index).fillna(0)

    # Compute log returns
    ohlcv["price_returns"] = np.log(ohlcv["price"] / ohlcv["price"].shift(1))

    # Rolling volatility: std of log returns
    # 5-minute window = 5 periods at 1-min bars
    ohlcv["volatility_5m"] = ohlcv["price_returns"].rolling(
        window=5, min_periods=2
    ).std()

    # 1-hour window = 60 periods at 1-min bars
    ohlcv["volatility_1h"] = ohlcv["price_returns"].rolling(
        window=60, min_periods=5
    ).std()

    # 5-minute moving average of volume
    ohlcv["volume_ma_5m"] = ohlcv["volume"].rolling(
        window=5, min_periods=1
    ).mean()

    # Reset index to get timestamp column
    ohlcv = ohlcv.reset_index()
    # Use datetime.timestamp() for pandas-version-agnostic unix seconds conversion.
    ohlcv["timestamp"] = ohlcv["datetime"].map(lambda x: int(x.timestamp()))
    ohlcv = ohlcv.drop(columns=["datetime"])

    # Forward fill remaining NaN then drop any leftover
    ohlcv = ohlcv.ffill().dropna()

    # Reorder columns
    ohlcv = ohlcv[
        [
            "timestamp", "price", "tick", "volume", "liquidity",
            "sqrtPriceX96", "volatility_5m", "volatility_1h",
            "volume_ma_5m", "price_returns",
        ]
    ]

    print(f"Produced {len(ohlcv)} 1-minute bars")
    return ohlcv


def main():
    parser = argparse.ArgumentParser(
        description="Process raw swap data into training features"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw swap events CSV",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for processed CSV (default: processed/training_data.csv)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(__file__), "processed", "training_data.csv"
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load and process
    raw_df = load_raw_data(args.input)
    features_df = process_raw_to_features(raw_df)

    # Save
    features_df.to_csv(args.output, index=False)
    print(f"Saved processed data to {args.output}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Time range: {features_df['timestamp'].min()} - {features_df['timestamp'].max()}")
    print(f"  Price range: ${features_df['price'].min():.2f} - ${features_df['price'].max():.2f}")
    print(f"  Total volume: {features_df['volume'].sum():.4f} ETH")
    print(f"  Avg volatility (5m): {features_df['volatility_5m'].mean():.6f}")
    print(f"  Avg volatility (1h): {features_df['volatility_1h'].mean():.6f}")


if __name__ == "__main__":
    main()
