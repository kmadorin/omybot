#!/usr/bin/env python3
"""Simulate alternating large swaps on an Anvil Base fork using Uniswap Universal Router v4."""

from __future__ import annotations

import argparse
import os
import sys
import time
from decimal import Decimal, getcontext

from eth_abi import encode as abi_encode
from eth_utils import to_checksum_address
from web3 import Web3

import config

getcontext().prec = 40

ANVIL_ACCOUNT_1_PRIVATE_KEY = (
    "0x59c6995e998f97a5a0044966f094538e5b0d2cefe4f7fcca7e5fca9c7e5e5fbb"
)

SWAP_EXACT_IN_SINGLE = 0x06
SETTLE_ALL = 0x0C
TAKE_ALL = 0x0F
V4_SWAP_COMMAND = 0x10

MAX_UINT256 = 2**256 - 1
MAX_UINT160 = 2**160 - 1
MAX_UINT48 = 2**48 - 1

ABI_DIR = os.path.join(os.path.dirname(__file__), "abi")


def _load_abi(filename: str) -> list:
    import json

    with open(os.path.join(ABI_DIR, filename)) as f:
        return json.load(f)


UNIVERSAL_ROUTER_ABI = [
    {
        "type": "function",
        "name": "execute",
        "inputs": [
            {"name": "commands", "type": "bytes"},
            {"name": "inputs", "type": "bytes[]"},
            {"name": "deadline", "type": "uint256"},
        ],
        "outputs": [],
        "stateMutability": "payable",
    }
]


def sqrt_price_x96_to_price(sqrt_price_x96: int) -> float:
    sqrt_price = Decimal(sqrt_price_x96)
    q96 = Decimal(2**96)
    return float((sqrt_price / q96) ** 2 * Decimal(10**12))


class SwapSimulator:
    def __init__(self, rpc_url: str, private_key: str):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to RPC at {rpc_url}")

        self.account = self.w3.eth.account.from_key(private_key)
        self.pool_id = bytes.fromhex(config.POOL_ID[2:])

        self.usdc = self.w3.eth.contract(
            address=to_checksum_address(config.USDC_ADDRESS),
            abi=_load_abi("erc20.json"),
        )
        self.permit2 = self.w3.eth.contract(
            address=to_checksum_address(config.PERMIT2),
            abi=_load_abi("permit2.json"),
        )
        self.state_view = self.w3.eth.contract(
            address=to_checksum_address(config.STATE_VIEW),
            abi=_load_abi("state_view.json"),
        )
        self.router = self.w3.eth.contract(
            address=to_checksum_address(config.UNIVERSAL_ROUTER),
            abi=UNIVERSAL_ROUTER_ABI,
        )

        self.pool_key = (
            to_checksum_address(config.ETH_ADDRESS),
            to_checksum_address(config.USDC_ADDRESS),
            config.POOL_FEE,
            config.TICK_SPACING,
            to_checksum_address(config.HOOKS_ADDRESS),
        )

    def _send_tx(self, tx: dict) -> str:
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt["status"] != 1:
            raise RuntimeError(f"Transaction reverted: {tx_hash.hex()}")
        return tx_hash.hex()

    def _build_tx(self, to: str, data: bytes, value: int = 0, gas: int = 400_000) -> dict:
        return {
            "chainId": self.w3.eth.chain_id,
            "from": self.account.address,
            "to": to,
            "nonce": self.w3.eth.get_transaction_count(self.account.address),
            "gas": gas,
            "gasPrice": self.w3.eth.gas_price,
            "value": value,
            "data": data,
        }

    def get_pool_snapshot(self) -> tuple[int, float]:
        slot0 = self.state_view.functions.getSlot0(self.pool_id).call()
        tick = int(slot0[1])
        price = sqrt_price_x96_to_price(int(slot0[0]))
        return tick, price

    def setup_usdc_allowances(self) -> None:
        current_erc20_allowance = self.usdc.functions.allowance(
            self.account.address,
            to_checksum_address(config.PERMIT2),
        ).call()

        if current_erc20_allowance < 10**6:
            tx = self.usdc.functions.approve(
                to_checksum_address(config.PERMIT2), MAX_UINT256
            ).build_transaction(
                {
                    "chainId": self.w3.eth.chain_id,
                    "from": self.account.address,
                    "nonce": self.w3.eth.get_transaction_count(self.account.address),
                    "gas": 120_000,
                    "gasPrice": self.w3.eth.gas_price,
                }
            )
            tx_hash = self._send_tx(tx)
            print(f"[swap-sim] USDC.approve(Permit2) tx={tx_hash}", flush=True)

        permit2_allowance = self.permit2.functions.allowance(
            self.account.address,
            to_checksum_address(config.USDC_ADDRESS),
            to_checksum_address(config.UNIVERSAL_ROUTER),
        ).call()
        current_amount = int(permit2_allowance[0])
        current_expiration = int(permit2_allowance[1])

        if current_amount < 10**6 or current_expiration <= int(time.time()) + 60:
            tx = self.permit2.functions.approve(
                to_checksum_address(config.USDC_ADDRESS),
                to_checksum_address(config.UNIVERSAL_ROUTER),
                MAX_UINT160,
                MAX_UINT48,
            ).build_transaction(
                {
                    "chainId": self.w3.eth.chain_id,
                    "from": self.account.address,
                    "nonce": self.w3.eth.get_transaction_count(self.account.address),
                    "gas": 120_000,
                    "gasPrice": self.w3.eth.gas_price,
                }
            )
            tx_hash = self._send_tx(tx)
            print(f"[swap-sim] Permit2.approve(USDC, UniversalRouter) tx={tx_hash}", flush=True)

    def _build_v4_swap_input(self, *, zero_for_one: bool, amount_in: int, amount_out_min: int) -> tuple[bytes, list[bytes], int]:
        input_currency = (
            to_checksum_address(config.ETH_ADDRESS)
            if zero_for_one
            else to_checksum_address(config.USDC_ADDRESS)
        )
        output_currency = (
            to_checksum_address(config.USDC_ADDRESS)
            if zero_for_one
            else to_checksum_address(config.ETH_ADDRESS)
        )

        swap_params = abi_encode(
            [
                "(address,address,uint24,int24,address)",
                "bool",
                "uint128",
                "uint128",
                "bytes",
            ],
            [
                self.pool_key,
                zero_for_one,
                amount_in,
                amount_out_min,
                b"",
            ],
        )

        settle_params = abi_encode(["address", "uint256"], [input_currency, amount_in])
        take_params = abi_encode(["address", "uint256"], [output_currency, amount_out_min])

        actions = bytes([SWAP_EXACT_IN_SINGLE, SETTLE_ALL, TAKE_ALL])
        inputs0 = abi_encode(["bytes", "bytes[]"], [actions, [swap_params, settle_params, take_params]])
        commands = bytes([V4_SWAP_COMMAND])
        return commands, [inputs0], int(time.time()) + 120

    def execute_eth_to_usdc(self, amount_in_wei: int) -> str:
        commands, inputs, deadline = self._build_v4_swap_input(
            zero_for_one=True,
            amount_in=amount_in_wei,
            amount_out_min=0,
        )
        tx = self.router.functions.execute(commands, inputs, deadline).build_transaction(
            {
                "chainId": self.w3.eth.chain_id,
                "from": self.account.address,
                "nonce": self.w3.eth.get_transaction_count(self.account.address),
                "gas": 1_500_000,
                "gasPrice": self.w3.eth.gas_price,
                "value": amount_in_wei,
            }
        )
        return self._send_tx(tx)

    def execute_usdc_to_eth(self, amount_in_usdc6: int) -> str:
        commands, inputs, deadline = self._build_v4_swap_input(
            zero_for_one=False,
            amount_in=amount_in_usdc6,
            amount_out_min=0,
        )
        tx = self.router.functions.execute(commands, inputs, deadline).build_transaction(
            {
                "chainId": self.w3.eth.chain_id,
                "from": self.account.address,
                "nonce": self.w3.eth.get_transaction_count(self.account.address),
                "gas": 1_500_000,
                "gasPrice": self.w3.eth.gas_price,
                "value": 0,
            }
        )
        return self._send_tx(tx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate alternating ETH/USDC swaps on Anvil fork")
    parser.add_argument("--rpc-url", default=config.LOCAL_RPC_URL, help="RPC URL (default: localhost anvil)")
    parser.add_argument("--private-key", default=ANVIL_ACCOUNT_1_PRIVATE_KEY, help="Private key for swap sender")
    parser.add_argument("--cycles", type=int, default=6, help="Number of alternating swap cycles")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between swaps")
    parser.add_argument("--eth-in", type=float, default=3.0, help="ETH amount per ETH->USDC swap")
    parser.add_argument("--usdc-in", type=float, default=6000.0, help="USDC amount per USDC->ETH swap")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    simulator = SwapSimulator(args.rpc_url, args.private_key)
    print(
        f"[swap-sim] connected chain_id={simulator.w3.eth.chain_id} "
        f"account={simulator.account.address}",
        flush=True,
    )

    simulator.setup_usdc_allowances()

    eth_in_wei = int(args.eth_in * 10**18)
    usdc_in_6 = int(args.usdc_in * 10**6)

    for i in range(args.cycles):
        direction = "ETH->USDC" if i % 2 == 0 else "USDC->ETH"
        tick_before, price_before = simulator.get_pool_snapshot()

        try:
            if i % 2 == 0:
                tx_hash = simulator.execute_eth_to_usdc(eth_in_wei)
            else:
                tx_hash = simulator.execute_usdc_to_eth(usdc_in_6)
        except Exception as exc:
            print(f"[swap-sim] swap failed on cycle={i+1} direction={direction}: {exc}", flush=True)
            return 1

        tick_after, price_after = simulator.get_pool_snapshot()
        print(
            f"[swap-sim] cycle={i+1}/{args.cycles} direction={direction} tx={tx_hash} "
            f"tick: {tick_before} -> {tick_after} price: {price_before:.2f} -> {price_after:.2f}",
            flush=True,
        )

        if i < args.cycles - 1:
            time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())
