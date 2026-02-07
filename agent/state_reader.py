"""
StateReader — reads Uniswap V4 pool and position state from StateView on Base.
"""

import json
import logging
import os
from decimal import Decimal, getcontext

from web3 import Web3

getcontext().prec = 40

logger = logging.getLogger(__name__)

ABI_DIR = os.path.join(os.path.dirname(__file__), "abi")


class StateReader:
    """Reads Uniswap V4 pool and position state via the StateView contract."""

    def __init__(self, w3: Web3, config):
        self.w3 = w3
        self.config = config

        # Pool ID as bytes32
        self.pool_id = bytes.fromhex(config.POOL_ID[2:])

        # Load StateView ABI and create contract
        with open(os.path.join(ABI_DIR, "state_view.json")) as f:
            state_view_abi = json.load(f)
        self.state_view = w3.eth.contract(
            address=Web3.to_checksum_address(config.STATE_VIEW),
            abi=state_view_abi,
        )

        # Load PoolManager ABI for events
        with open(os.path.join(ABI_DIR, "pool_manager.json")) as f:
            pool_manager_abi = json.load(f)
        self.pool_manager = w3.eth.contract(
            address=Web3.to_checksum_address(config.POOL_MANAGER),
            abi=pool_manager_abi,
        )

    def get_slot0(self) -> dict:
        """Get pool slot0 data: sqrtPriceX96, tick, protocolFee, lpFee."""
        try:
            result = self.state_view.functions.getSlot0(self.pool_id).call()
            return {
                "sqrtPriceX96": result[0],
                "tick": result[1],
                "protocolFee": result[2],
                "lpFee": result[3],
            }
        except Exception as e:
            logger.error("Failed to get slot0: %s", e)
            raise

    def get_pool_liquidity(self) -> int:
        """Get current in-range liquidity for the pool."""
        try:
            return self.state_view.functions.getLiquidity(self.pool_id).call()
        except Exception as e:
            logger.error("Failed to get pool liquidity: %s", e)
            raise

    def get_position_info(
        self, owner: str, tick_lower: int, tick_upper: int, salt: bytes = b""
    ) -> dict:
        """Get position info for a specific owner and tick range.

        Returns liquidity, feeGrowthInside0LastX128, feeGrowthInside1LastX128.
        """
        try:
            # salt must be bytes32 — pad with zeros if shorter
            salt_bytes32 = salt.ljust(32, b"\x00") if len(salt) < 32 else salt[:32]
            result = self.state_view.functions.getPositionInfo(
                self.pool_id,
                Web3.to_checksum_address(owner),
                tick_lower,
                tick_upper,
                salt_bytes32,
            ).call()
            return {
                "liquidity": result[0],
                "feeGrowthInside0LastX128": result[1],
                "feeGrowthInside1LastX128": result[2],
            }
        except Exception as e:
            logger.error("Failed to get position info: %s", e)
            raise

    def get_current_price(self) -> float:
        """Derive ETH price in USDC from sqrtPriceX96.

        Formula: price = (sqrtPriceX96 / 2^96)^2 * 10^12
        The 10^12 adjusts for decimal difference: ETH(18) - USDC(6) = 12.
        """
        try:
            slot0 = self.get_slot0()
            sqrt_price = Decimal(slot0["sqrtPriceX96"])
            q96 = Decimal(2**96)
            price = (sqrt_price / q96) ** 2 * Decimal(10**12)
            return float(price)
        except Exception as e:
            logger.error("Failed to get current price: %s", e)
            raise

    def tick_to_price(self, tick: int) -> float:
        """Convert a tick value to a human-readable price.

        Formula: price = 1.0001^tick * 10^12
        """
        base = Decimal("1.0001")
        price = base ** tick * Decimal(10**12)
        return float(price)

    def get_fee_growth_inside(
        self, tick_lower: int, tick_upper: int
    ) -> tuple[int, int]:
        """Get fee growth inside a tick range.

        Returns (feeGrowthInside0X128, feeGrowthInside1X128).
        """
        try:
            result = self.state_view.functions.getFeeGrowthInside(
                self.pool_id, tick_lower, tick_upper
            ).call()
            return (result[0], result[1])
        except Exception as e:
            logger.error("Failed to get fee growth inside: %s", e)
            raise

    def get_pool_state(self) -> dict:
        """Convenience method combining slot0 + liquidity + price."""
        try:
            slot0 = self.get_slot0()
            liquidity = self.get_pool_liquidity()

            sqrt_price = Decimal(slot0["sqrtPriceX96"])
            q96 = Decimal(2**96)
            price = float((sqrt_price / q96) ** 2 * Decimal(10**12))

            return {
                "sqrtPriceX96": slot0["sqrtPriceX96"],
                "tick": slot0["tick"],
                "protocolFee": slot0["protocolFee"],
                "lpFee": slot0["lpFee"],
                "liquidity": liquidity,
                "price": price,
            }
        except Exception as e:
            logger.error("Failed to get pool state: %s", e)
            raise
