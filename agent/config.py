import os
from pathlib import Path

from dotenv import load_dotenv

# Load agent/.env into os.environ BEFORE reading any env-backed settings.
# override=False means Docker/shell env vars take precedence over .env.
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

# Base chain (chain ID 8453)
BASE_RPC_URL = os.environ.get("BASE_RPC_URL", "https://mainnet.base.org")
LOCAL_RPC_URL = os.environ.get("LOCAL_RPC_URL", "http://localhost:8545")
USE_FORK = os.environ.get("USE_FORK", "true").lower() in ("true", "1", "yes")

EXPECTED_CHAIN_ID = int(os.environ.get("EXPECTED_CHAIN_ID", "8453"))

# Uniswap V4 contracts on Base
POOL_MANAGER = "0x498581fF718922c3f8e6A244956aF099B2652b2b"
POSITION_MANAGER = "0x7C5f5A4bBd8fD63184577525326123B519429bDc"
STATE_VIEW = "0xA3c0c9b65baD0b08107Aa264b0f3dB444b867A71"
UNIVERSAL_ROUTER = "0x6ff5693b99212da76ad316178a184ab56d299b43"
PERMIT2 = "0x000000000022D473030F116dDEE9F6B43aC78BA3"

# Tokens
ETH_ADDRESS = "0x0000000000000000000000000000000000000000"  # Native ETH (currency0)
USDC_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"  # 6 decimals (currency1)
WETH_ADDRESS = "0x4200000000000000000000000000000000000006"  # 18 decimals

# Pool params — ETH/USDC 0.05% pool on Base
# currency0=ETH (0x0), currency1=USDC, fee=500, tickSpacing=10, hooks=0x0
POOL_FEE = 500
TICK_SPACING = 10
HOOKS_ADDRESS = "0x0000000000000000000000000000000000000000"

# Known pool ID (keccak256 of PoolKey encoding)
# This is the ETH/USDC 0.05% pool: https://app.uniswap.org/explore/pools/base/0x96d4b53a...
POOL_ID = "0x96d4b53a38337a5733179751781178a2613306063c511b78cd02684739288c0a"

# Agent params
TRADE_INTERVAL = 15  # seconds between decisions
REBALANCE_DRIFT_THRESHOLD = 0.75  # rebalance if price drifts >75% toward edge
FEE_COLLECT_THRESHOLD = 0.001  # collect fees if >0.1% of position value

# Token decimals
ETH_DECIMALS = 18
USDC_DECIMALS = 6

# V4 PositionManager Action codes
INCREASE_LIQUIDITY = 0x00
DECREASE_LIQUIDITY = 0x01
MINT_POSITION = 0x02
BURN_POSITION = 0x03
SETTLE_PAIR = 0x0d
TAKE_PAIR = 0x11
CLOSE_CURRENCY = 0x12
SWEEP = 0x14

# Data collection
START_BLOCK = 25350999
BATCH_SIZE = 200
BASE_BLOCK_TIME = 2


def build_pool_key():
    from eth_utils.address import to_checksum_address

    currency0 = to_checksum_address(ETH_ADDRESS)
    currency1 = to_checksum_address(USDC_ADDRESS)
    if int(currency0, 16) > int(currency1, 16):
        currency0, currency1 = currency1, currency0

    return {
        "currency0": currency0,
        "currency1": currency1,
        "fee": POOL_FEE,
        "tick_spacing": TICK_SPACING,
        "hooks": to_checksum_address(HOOKS_ADDRESS),
    }


def compute_pool_id():
    from eth_abi.abi import encode
    from eth_utils.crypto import keccak

    pool_key = build_pool_key()
    pool_key_encoded = encode(
        ["address", "address", "uint24", "int24", "address"],
        [
            pool_key["currency0"],
            pool_key["currency1"],
            pool_key["fee"],
            pool_key["tick_spacing"],
            pool_key["hooks"],
        ],
    )
    return "0x" + keccak(pool_key_encoded).hex()
