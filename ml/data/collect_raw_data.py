"""
Collect raw Swap event data from Uniswap V4 PoolManager on Base.

Fetches Swap events filtered by pool ID, extracts key fields,
and saves to CSV with resume support and parallel block processing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, cast

import pandas as pd
from web3 import Web3

# Ensure repo root is on sys.path for agent config import (works with type checkers).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent import config


# Retry settings
MAX_RETRIES = 5
BASE_RETRY_DELAY = 5
MAX_RETRY_DELAY = 60

# Swap event signature
SWAP_EVENT_SIG = "Swap(bytes32,address,int128,int128,uint160,uint128,int24,uint24)"


class SwapDataCollector:
    def __init__(self, rpc_url=None):
        self.rpc_url = rpc_url or config.BASE_RPC_URL
        self._connect()
        self._setup_contract()

        # Pool ID for topic filtering (padded to 32 bytes)
        pool_id_hex = config.POOL_ID.replace("0x", "")
        self.pool_id_topic = "0x" + pool_id_hex.zfill(64)

        # Event signature topic
        self.swap_topic0 = "0x" + Web3.keccak(text=SWAP_EVENT_SIG).hex()

        # Block timestamp cache to reduce RPC calls
        self._block_ts_cache = {}

    def _connect(self):
        """Initialize Web3 connection with retries."""
        for attempt in range(MAX_RETRIES):
            try:
                self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
                block = self.w3.eth.block_number
                print(f"Connected to {self.rpc_url} at block {block}")
                return
            except Exception as e:
                delay = min(BASE_RETRY_DELAY * (2**attempt), MAX_RETRY_DELAY)
                print(f"Connection failed (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"Retrying in {delay}s...")
                    time.sleep(delay)
        raise ConnectionError(
            f"Failed to connect to {self.rpc_url} after {MAX_RETRIES} attempts"
        )

    def _setup_contract(self):
        """Create PoolManager contract instance from ABI."""
        abi_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "agent", "abi", "pool_manager.json"
        )
        with open(abi_path) as f:
            abi = json.load(f)

        self.pool_manager_address = Web3.to_checksum_address(config.POOL_MANAGER)
        self.contract = self.w3.eth.contract(address=self.pool_manager_address, abi=abi)

    def _get_block_timestamp(self, block_number):
        """Get block timestamp with caching."""
        if block_number not in self._block_ts_cache:
            for attempt in range(MAX_RETRIES):
                try:
                    block = self.w3.eth.get_block(block_number)
                    block_data = block  # web3 returns a dict-like mapping
                    if not isinstance(block_data, Mapping):
                        raise TypeError("Unexpected block response type")
                    timestamp = block_data.get("timestamp")
                    if timestamp is None:
                        raise KeyError("timestamp missing in block data")
                    self._block_ts_cache[block_number] = int(timestamp)
                    break
                except Exception as e:
                    delay = min(BASE_RETRY_DELAY * (2**attempt), MAX_RETRY_DELAY)
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(delay)
                    else:
                        print(f"Failed to get timestamp for block {block_number}: {e}")
                        return 0
        return self._block_ts_cache[block_number]

    def _process_log(self, log):
        """Decode a raw Swap event log into a dict."""
        try:
            event_builder = cast(Any, self.contract.events.Swap())
            event = event_builder.process_log(log)
            block_num = event.blockNumber
            timestamp = self._get_block_timestamp(block_num)
            return {
                "block_number": block_num,
                "timestamp": timestamp,
                "transaction_hash": event.transactionHash.hex(),
                "amount0": str(event.args.amount0),
                "amount1": str(event.args.amount1),
                "sqrtPriceX96": str(event.args.sqrtPriceX96),
                "liquidity": str(event.args.liquidity),
                "tick": event.args.tick,
                "fee": event.args.fee,
            }
        except Exception as e:
            print(f"Error processing log: {e}")
            return None

    def collect_block_range(self, start_block, end_block):
        """Collect Swap events from a range of blocks with retries."""
        events = []
        current = start_block
        batch_size = config.BATCH_SIZE

        while current <= end_block:
            batch_end = min(current + batch_size - 1, end_block)

            for attempt in range(MAX_RETRIES):
                try:
                    logs = self.w3.eth.get_logs(
                        {
                            "address": self.pool_manager_address,
                            "fromBlock": current,
                            "toBlock": batch_end,
                            "topics": [self.swap_topic0, self.pool_id_topic],
                        }
                    )
                    for log in logs:
                        row = self._process_log(log)
                        if row is not None:
                            events.append(row)

                    print(f"  Blocks {current}-{batch_end}: {len(logs)} swap events")
                    break
                except Exception as e:
                    delay = min(BASE_RETRY_DELAY * (2**attempt), MAX_RETRY_DELAY)
                    print(
                        f"  Error blocks {current}-{batch_end} "
                        f"(attempt {attempt + 1}): {e}"
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(delay)
                        # Reconnect on failure
                        try:
                            self._connect()
                            self._setup_contract()
                        except Exception:
                            pass
                    else:
                        print(f"  Skipping blocks {current}-{batch_end}")

            current = batch_end + 1

        return events

    def get_last_processed_block(self, output_file):
        """Check CSV for the last processed block to support resume."""
        try:
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                df = pd.read_csv(output_file)
                if not df.empty and "block_number" in df.columns:
                    max_block = df["block_number"].max()
                    if isinstance(max_block, pd.Series):
                        max_block = max_block.max()
                    if max_block is None or pd.isna(cast(Any, max_block)):
                        return None
                    return int(cast(float, max_block))
        except Exception as e:
            print(f"Error reading last processed block: {e}")
        return None

    def collect(self, output_file, start_block=None):
        """Main collection loop with parallel workers and resume support."""
        if start_block is None:
            start_block = config.START_BLOCK

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Check for resume
        last_block = self.get_last_processed_block(output_file)
        if last_block is not None:
            start_block = last_block + 1
            print(f"Resuming from block {start_block}")

        num_workers = 4
        # Each worker chunk spans this many blocks
        blocks_per_chunk = config.BATCH_SIZE * 10  # 2000 blocks per chunk

        try:
            while True:
                latest_block = self.w3.eth.block_number

                if start_block >= latest_block:
                    print("Caught up to chain head. Waiting for new blocks...")
                    time.sleep(30)
                    continue

                # Divide work across workers
                total_blocks = min(
                    blocks_per_chunk * num_workers, latest_block - start_block
                )
                chunk_size = max(total_blocks // num_workers, config.BATCH_SIZE)

                ranges = []
                cur = start_block
                for _ in range(num_workers):
                    if cur > start_block + total_blocks:
                        break
                    rng_end = min(cur + chunk_size - 1, latest_block)
                    ranges.append((cur, rng_end))
                    cur = rng_end + 1

                if not ranges:
                    time.sleep(30)
                    continue

                print(
                    f"\nProcessing blocks {ranges[0][0]}-{ranges[-1][1]} "
                    f"with {len(ranges)} workers"
                )

                all_events = []
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(self.collect_block_range, s, e): (s, e)
                        for s, e in ranges
                    }
                    for future in as_completed(futures):
                        rng = futures[future]
                        try:
                            events = future.result()
                            all_events.extend(events)
                            print(
                                f"Worker {rng[0]}-{rng[1]} done: {len(events)} events"
                            )
                        except Exception as e:
                            print(f"Worker {rng[0]}-{rng[1]} failed: {e}")

                # Save chunk
                if all_events:
                    df = pd.DataFrame(all_events)
                    df = df.sort_values("block_number").reset_index(drop=True)
                    file_exists = os.path.isfile(output_file)
                    df.to_csv(
                        output_file,
                        mode="a",
                        header=not file_exists,
                        index=False,
                    )
                    print(f"Saved {len(df)} events to {output_file}")

                # Advance
                start_block = ranges[-1][1] + 1

        except KeyboardInterrupt:
            print("\nCollection stopped by user.")
            if os.path.exists(output_file):
                total = len(pd.read_csv(output_file))
                print(f"Total events collected: {total}")


def main():
    parser = argparse.ArgumentParser(description="Collect Uniswap V4 swap events")
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__),
            "raw",
            f"swap_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        ),
        help="Output CSV path",
    )
    parser.add_argument(
        "--start-block",
        type=int,
        default=None,
        help=f"Starting block (default: config.START_BLOCK={config.START_BLOCK})",
    )
    parser.add_argument(
        "--rpc-url",
        default=None,
        help=f"RPC URL (default: {config.BASE_RPC_URL})",
    )
    args = parser.parse_args()

    collector = SwapDataCollector(rpc_url=args.rpc_url)
    collector.collect(output_file=args.output, start_block=args.start_block)


if __name__ == "__main__":
    main()
