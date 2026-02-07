"""
Uniswap V4 PositionManager LP operations: mint, collect fees, rebalance.
Uses raw eth_abi encoding for action commands sent via modifyLiquidities().
"""

import json
import logging
import os
from datetime import datetime, timezone

from eth_abi import encode as abi_encode
from eth_utils import to_checksum_address
from web3 import Web3

logger = logging.getLogger(__name__)

ABI_DIR = os.path.join(os.path.dirname(__file__), "abi")


def _load_abi(filename: str) -> list:
    with open(os.path.join(ABI_DIR, filename)) as f:
        return json.load(f)


POSITIONS_FILE = os.path.join(os.path.dirname(__file__), "positions.json")

# Max uint values used in approvals
MAX_UINT256 = 2**256 - 1
MAX_UINT160 = 2**160 - 1
MAX_UINT48 = 2**48 - 1


class LPManager:
    """Encodes and sends Uniswap V4 PositionManager commands."""

    def __init__(self, w3: Web3, account, config):
        self.w3 = w3
        self.account = account
        self.config = config

        pm_abi = _load_abi("position_manager.json")
        erc20_abi = _load_abi("erc20.json")
        permit2_abi = _load_abi("permit2.json")

        self.position_manager = w3.eth.contract(
            address=to_checksum_address(config.POSITION_MANAGER),
            abi=pm_abi,
        )
        self.usdc = w3.eth.contract(
            address=to_checksum_address(config.USDC_ADDRESS),
            abi=erc20_abi,
        )
        self.permit2 = w3.eth.contract(
            address=to_checksum_address(config.PERMIT2),
            abi=permit2_abi,
        )

    # ------------------------------------------------------------------
    # Pool key
    # ------------------------------------------------------------------

    def build_pool_key(self) -> tuple:
        """Returns (currency0, currency1, fee, tickSpacing, hooks) as a tuple.

        currency0 = ETH (0x0) is always sorted before USDC (0x833...).
        """
        currency0 = to_checksum_address(self.config.ETH_ADDRESS)
        currency1 = to_checksum_address(self.config.USDC_ADDRESS)
        return (
            currency0,
            currency1,
            self.config.POOL_FEE,
            self.config.TICK_SPACING,
            to_checksum_address(self.config.HOOKS_ADDRESS),
        )

    # ------------------------------------------------------------------
    # Approvals (one-time setup)
    # ------------------------------------------------------------------

    def setup_approvals(self):
        """Approve USDC -> Permit2 -> PositionManager chain.

        Native ETH needs no approval (sent as msg.value).
        """
        sender = self.account.address

        # Step 1: USDC.approve(Permit2, MAX_UINT256)
        tx = self.usdc.functions.approve(
            to_checksum_address(self.config.PERMIT2), MAX_UINT256
        ).build_transaction(
            {
                "from": sender,
                "nonce": self.w3.eth.get_transaction_count(sender),
                "gas": 100_000,
                "gasPrice": self.w3.eth.gas_price,
            }
        )
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        logger.info("USDC.approve(Permit2) tx=%s status=%s", tx_hash.hex(), receipt["status"])

        # Step 2: Permit2.approve(USDC, PositionManager, MAX_UINT160, MAX_UINT48)
        tx2 = self.permit2.functions.approve(
            to_checksum_address(self.config.USDC_ADDRESS),
            to_checksum_address(self.config.POSITION_MANAGER),
            MAX_UINT160,
            MAX_UINT48,
        ).build_transaction(
            {
                "from": sender,
                "nonce": self.w3.eth.get_transaction_count(sender),
                "gas": 100_000,
                "gasPrice": self.w3.eth.gas_price,
            }
        )
        signed2 = self.account.sign_transaction(tx2)
        tx_hash2 = self.w3.eth.send_raw_transaction(signed2.raw_transaction)
        receipt2 = self.w3.eth.wait_for_transaction_receipt(tx_hash2)
        logger.info(
            "Permit2.approve(USDC, PosMgr) tx=%s status=%s",
            tx_hash2.hex(),
            receipt2["status"],
        )

    # ------------------------------------------------------------------
    # Mint position
    # ------------------------------------------------------------------

    def mint_position(
        self,
        tick_lower: int,
        tick_upper: int,
        liquidity: int,
        amount0_max: int,
        amount1_max: int,
        deadline: int,
    ) -> dict:
        """Mint a new LP position.

        Actions: [MINT_POSITION, CLOSE_CURRENCY, CLOSE_CURRENCY, SWEEP]

        Returns {"tx_hash": str, "token_id": int}.
        """
        pool_key = self.build_pool_key()
        cfg = self.config

        # MINT_POSITION params:
        # (PoolKey, int24 tickLower, int24 tickUpper, uint256 liquidity,
        #  uint128 amount0Max, uint128 amount1Max, address owner, bytes hookData)
        mint_params = abi_encode(
            [
                "(address,address,uint24,int24,address)",
                "int24",
                "int24",
                "uint256",
                "uint128",
                "uint128",
                "address",
                "bytes",
            ],
            [
                pool_key,
                tick_lower,
                tick_upper,
                liquidity,
                amount0_max,
                amount1_max,
                self.account.address,
                b"",
            ],
        )

        # CLOSE_CURRENCY params: (address currency)
        close_c0 = abi_encode(["address"], [to_checksum_address(cfg.ETH_ADDRESS)])
        close_c1 = abi_encode(["address"], [to_checksum_address(cfg.USDC_ADDRESS)])

        # SWEEP params: (address currency, address to) — recover excess ETH
        sweep = abi_encode(
            ["address", "address"],
            [to_checksum_address(cfg.ETH_ADDRESS), self.account.address],
        )

        actions = bytes(
            [cfg.MINT_POSITION, cfg.CLOSE_CURRENCY, cfg.CLOSE_CURRENCY, cfg.SWEEP]
        )
        params = [mint_params, close_c0, close_c1, sweep]

        receipt = self._send_modify_liquidities(
            actions, params, deadline, value=amount0_max
        )

        token_id = self._parse_token_id_from_receipt(receipt)
        logger.info(
            "Minted position token_id=%s ticks=[%d, %d] liq=%d tx=%s",
            token_id,
            tick_lower,
            tick_upper,
            liquidity,
            receipt["transactionHash"].hex(),
        )
        return {
            "tx_hash": receipt["transactionHash"].hex(),
            "token_id": token_id,
        }

    # ------------------------------------------------------------------
    # Collect fees
    # ------------------------------------------------------------------

    def collect_fees(self, token_id: int, deadline: int) -> dict:
        """Collect accrued fees without removing liquidity.

        Actions: [DECREASE_LIQUIDITY (liquidity=0), TAKE_PAIR]

        Returns {"tx_hash": str}.
        """
        cfg = self.config

        # DECREASE_LIQUIDITY with liquidity=0 => collect fees only
        # (uint256 tokenId, uint256 liquidity, uint128 amount0Min, uint128 amount1Min, bytes hookData)
        decrease_params = abi_encode(
            ["uint256", "uint256", "uint128", "uint128", "bytes"],
            [token_id, 0, 0, 0, b""],
        )

        # TAKE_PAIR params: (address currency0, address currency1)
        take_params = abi_encode(
            ["address", "address"],
            [
                to_checksum_address(cfg.ETH_ADDRESS),
                to_checksum_address(cfg.USDC_ADDRESS),
            ],
        )

        actions = bytes([cfg.DECREASE_LIQUIDITY, cfg.TAKE_PAIR])
        params = [decrease_params, take_params]

        receipt = self._send_modify_liquidities(actions, params, deadline)
        logger.info(
            "Collected fees for token_id=%d tx=%s",
            token_id,
            receipt["transactionHash"].hex(),
        )
        return {"tx_hash": receipt["transactionHash"].hex()}

    # ------------------------------------------------------------------
    # Decrease liquidity (withdraw to wallet)
    # ------------------------------------------------------------------

    def decrease_liquidity(self, token_id: int, liquidity: int, deadline: int) -> dict:
        """Remove liquidity from a position and withdraw tokens to wallet.

        Actions: [DECREASE_LIQUIDITY, CLOSE_CURRENCY, CLOSE_CURRENCY, SWEEP]

        Returns {"tx_hash": str}.
        """
        cfg = self.config

        decrease_params = abi_encode(
            ["uint256", "uint256", "uint128", "uint128", "bytes"],
            [token_id, liquidity, 0, 0, b""],
        )

        close_c0 = abi_encode(["address"], [to_checksum_address(cfg.ETH_ADDRESS)])
        close_c1 = abi_encode(["address"], [to_checksum_address(cfg.USDC_ADDRESS)])

        sweep = abi_encode(
            ["address", "address"],
            [to_checksum_address(cfg.ETH_ADDRESS), self.account.address],
        )

        actions = bytes(
            [cfg.DECREASE_LIQUIDITY, cfg.CLOSE_CURRENCY, cfg.CLOSE_CURRENCY, cfg.SWEEP]
        )
        params = [decrease_params, close_c0, close_c1, sweep]

        receipt = self._send_modify_liquidities(actions, params, deadline)
        logger.info(
            "Decreased liquidity for token_id=%d tx=%s",
            token_id,
            receipt["transactionHash"].hex(),
        )
        return {"tx_hash": receipt["transactionHash"].hex()}

    # ------------------------------------------------------------------
    # Rebalance (atomic: remove old + mint new)
    # ------------------------------------------------------------------

    def rebalance(
        self,
        token_id: int,
        old_liquidity: int,
        new_tick_lower: int,
        new_tick_upper: int,
        new_liquidity: int,
        amount0_max: int,
        amount1_max: int,
        deadline: int,
    ) -> dict:
        """Atomic rebalance: remove all liquidity from old position, mint new one.

        Actions: [DECREASE_LIQUIDITY, MINT_POSITION, CLOSE_CURRENCY, CLOSE_CURRENCY, SWEEP]

        Returns {"tx_hash": str, "new_token_id": int}.
        """
        pool_key = self.build_pool_key()
        cfg = self.config

        # Remove all liquidity from old position
        decrease_params = abi_encode(
            ["uint256", "uint256", "uint128", "uint128", "bytes"],
            [token_id, old_liquidity, 0, 0, b""],
        )

        # Mint new position at new range
        mint_params = abi_encode(
            [
                "(address,address,uint24,int24,address)",
                "int24",
                "int24",
                "uint256",
                "uint128",
                "uint128",
                "address",
                "bytes",
            ],
            [
                pool_key,
                new_tick_lower,
                new_tick_upper,
                new_liquidity,
                amount0_max,
                amount1_max,
                self.account.address,
                b"",
            ],
        )

        # CLOSE_CURRENCY for each token (handles both settle and take directions)
        close_c0 = abi_encode(["address"], [to_checksum_address(cfg.ETH_ADDRESS)])
        close_c1 = abi_encode(["address"], [to_checksum_address(cfg.USDC_ADDRESS)])

        # SWEEP to recover any excess native ETH
        sweep = abi_encode(
            ["address", "address"],
            [to_checksum_address(cfg.ETH_ADDRESS), self.account.address],
        )

        actions = bytes(
            [
                cfg.DECREASE_LIQUIDITY,
                cfg.MINT_POSITION,
                cfg.CLOSE_CURRENCY,
                cfg.CLOSE_CURRENCY,
                cfg.SWEEP,
            ]
        )
        params = [decrease_params, mint_params, close_c0, close_c1, sweep]

        receipt = self._send_modify_liquidities(
            actions, params, deadline, value=amount0_max
        )

        new_token_id = self._parse_token_id_from_receipt(receipt)
        logger.info(
            "Rebalanced: old_token=%d -> new_token=%s ticks=[%d, %d] tx=%s",
            token_id,
            new_token_id,
            new_tick_lower,
            new_tick_upper,
            receipt["transactionHash"].hex(),
        )
        return {
            "tx_hash": receipt["transactionHash"].hex(),
            "new_token_id": new_token_id,
        }

    # ------------------------------------------------------------------
    # Internal: send modifyLiquidities transaction
    # ------------------------------------------------------------------

    def _send_modify_liquidities(
        self,
        actions: bytes,
        params: list,
        deadline: int,
        value: int = 0,
    ) -> dict:
        """Encode unlockData, build tx, sign, send, and wait for receipt."""
        unlock_data = abi_encode(["bytes", "bytes[]"], [actions, params])

        tx = self.position_manager.functions.modifyLiquidities(
            unlock_data, deadline
        ).build_transaction(
            {
                "from": self.account.address,
                "nonce": self.w3.eth.get_transaction_count(self.account.address),
                "value": value,
                "gas": 1_000_000,
                "gasPrice": self.w3.eth.gas_price,
            }
        )

        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        if receipt["status"] != 1:
            logger.error("Transaction reverted: tx=%s", tx_hash.hex())
            raise RuntimeError(f"modifyLiquidities reverted: {tx_hash.hex()}")

        return receipt

    # ------------------------------------------------------------------
    # Parse token ID from receipt (ERC721 Transfer event)
    # ------------------------------------------------------------------

    def _parse_token_id_from_receipt(self, receipt) -> int | None:
        """Extract minted tokenId from ERC721 Transfer(from=0x0, to, id) event."""
        transfer_topic = self.w3.keccak(text="Transfer(address,address,uint256)")
        zero_address_topic = "0x" + "0" * 64
        pm_address = to_checksum_address(self.config.POSITION_MANAGER).lower()

        for log in receipt.get("logs", []):
            if log["address"].lower() != pm_address:
                continue
            if len(log["topics"]) < 4:
                continue
            if log["topics"][0] != transfer_topic:
                continue
            # Transfer from 0x0 means a mint
            topic1_hex = log["topics"][1].hex()
            if not topic1_hex.startswith("0x"):
                topic1_hex = "0x" + topic1_hex
            if topic1_hex == zero_address_topic:
                topic3_hex = log["topics"][3].hex()
                if not topic3_hex.startswith("0x"):
                    topic3_hex = "0x" + topic3_hex
                token_id = int(topic3_hex, 16)
                return token_id

        logger.warning("Could not parse token_id from receipt logs")
        return None

    # ------------------------------------------------------------------
    # Position persistence (JSON file)
    # ------------------------------------------------------------------

    def save_position(
        self,
        token_id: int,
        tick_lower: int,
        tick_upper: int,
        entry_price: float | None = None,
    ):
        """Append a position record to positions.json."""
        positions = self.load_positions()

        record = {
            "token_id": token_id,
            "tick_lower": tick_lower,
            "tick_upper": tick_upper,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if entry_price is not None:
            record["entry_price"] = float(entry_price)

        positions.append(record)

        with open(POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=2)

        logger.info("Saved position token_id=%d to %s", token_id, POSITIONS_FILE)

    def load_positions(self) -> list[dict]:
        """Load all saved positions from positions.json."""
        if not os.path.exists(POSITIONS_FILE):
            return []
        try:
            with open(POSITIONS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Could not load positions from %s, returning empty", POSITIONS_FILE)
            return []
