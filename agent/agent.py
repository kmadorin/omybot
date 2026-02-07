"""
LPAgent — main decision loop for managing Uniswap V4 LP positions on a Base fork.

Orchestrates StateReader and LPManager to autonomously manage LP positions
with rule-based decisions (rebalance, collect fees, or hold).
"""

import json
import logging
import os
import sys
import time

from web3 import Web3

# Sibling module imports
from state_reader import StateReader
from lp_manager import LPManager
import config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = os.path.dirname(__file__)
DECISIONS_LOG = os.path.join(LOG_DIR, "decisions.log")

logger = logging.getLogger("omybot.agent")
logger.setLevel(logging.DEBUG)

# Console handler
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_console)

# File handler — decisions log
_fh = logging.FileHandler(DECISIONS_LOG)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
logger.addHandler(_fh)


class LPAgent:
    """Autonomous Uniswap V4 LP manager."""

    # Default range: +/- 1000 ticks from current tick
    RANGE_TICKS = 1000

    # Default liquidity for initial mint
    DEFAULT_LIQUIDITY = 10**15
    # Minimum seconds between fee collection txs
    FEE_COLLECT_COOLDOWN = 300

    def __init__(self, private_key: str):
        rpc_url = config.LOCAL_RPC_URL if config.USE_FORK else config.BASE_RPC_URL
        logger.info("Connecting to RPC: %s", rpc_url)
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to RPC at {rpc_url}")
        logger.info("Connected. Chain ID: %d", self.w3.eth.chain_id)

        self.account = self.w3.eth.account.from_key(private_key)
        logger.info("Agent address: %s", self.account.address)

        self.state_reader = StateReader(self.w3, config)
        self.lp_manager = LPManager(self.w3, self.account, config)

        # Load any persisted positions
        self.positions: list[dict] = self.lp_manager.load_positions()
        logger.info("Loaded %d existing position(s)", len(self.positions))

        # Concurrency lock
        self.rebalance_pending = False
        self.last_fee_collect_ts = 0.0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        """Run approvals and mint an initial position if none exist."""
        logger.info("Running setup...")
        self.lp_manager.setup_approvals()

        if self.positions:
            logger.info("Existing positions found — skipping initial mint.")
            return

        logger.info("No existing positions — minting initial position.")
        pool_state = self.state_reader.get_pool_state()
        current_tick = pool_state["tick"]
        logger.info(
            "Current tick=%d  price=%.2f USDC/ETH", current_tick, pool_state["price"]
        )

        tick_lower = self.snap_tick(current_tick - self.RANGE_TICKS)
        tick_upper = self.snap_tick(current_tick + self.RANGE_TICKS)
        logger.info("Initial range: [%d, %d]", tick_lower, tick_upper)

        # Determine max amounts from available balances
        eth_balance = self.w3.eth.get_balance(self.account.address)
        usdc_contract = self.lp_manager.usdc
        usdc_balance = usdc_contract.functions.balanceOf(self.account.address).call()

        # Use up to 50% of balances for safety margin
        amount0_max = eth_balance // 2  # ETH (currency0)
        amount1_max = usdc_balance // 2  # USDC (currency1)

        logger.info(
            "Balances: ETH=%.4f  USDC=%.2f | Using 50%%: ETH_max=%d  USDC_max=%d",
            eth_balance / 10**18,
            usdc_balance / 10**6,
            amount0_max,
            amount1_max,
        )

        deadline = int(time.time()) + 600

        result = self.lp_manager.mint_position(
            tick_lower=tick_lower,
            tick_upper=tick_upper,
            liquidity=self.DEFAULT_LIQUIDITY,
            amount0_max=amount0_max,
            amount1_max=amount1_max,
            deadline=deadline,
        )

        token_id = result["token_id"]
        self.lp_manager.save_position(token_id, tick_lower, tick_upper)
        self.positions = self.lp_manager.load_positions()
        logger.info(
            "Initial position minted: token_id=%s  range=[%d, %d]",
            token_id,
            tick_lower,
            tick_upper,
        )

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def get_decision(self, pool_state: dict, position: dict) -> str:
        """Rule-based decision: REBALANCE, COLLECT_FEES, or HOLD.

        drift_ratio measures how far current tick has moved toward the nearest
        edge of the position.  0.0 = centered, 1.0 = fully out of range.
        """
        current_tick = pool_state["tick"]
        tick_lower = position["tick_lower"]
        tick_upper = position["tick_upper"]
        range_width = tick_upper - tick_lower

        if range_width == 0:
            return "REBALANCE"

        # Calculate drift ratio
        if current_tick >= tick_upper:
            drift_ratio = 1.0
        elif current_tick < tick_lower:
            drift_ratio = 1.0
        else:
            distance_to_nearest_edge = min(
                current_tick - tick_lower,
                tick_upper - current_tick,
            )
            drift_ratio = 1.0 - (distance_to_nearest_edge / (range_width / 2))

        # Decision
        if drift_ratio > config.REBALANCE_DRIFT_THRESHOLD:
            return "REBALANCE"

        # Simple fee check: if in range, fees are accruing — collect periodically
        # A more precise check would compare fee growth deltas, but for the
        # rule-based agent we use the configured threshold as a proxy.
        in_range = tick_lower <= current_tick < tick_upper
        if in_range and drift_ratio < 0.3:
            # When comfortably in range with low drift, collect fees periodically.
            # A precise check would compare fee growth deltas; for now, use drift as proxy.
            now = time.time()
            if now - self.last_fee_collect_ts >= self.FEE_COLLECT_COOLDOWN:
                return "COLLECT_FEES"

        return "HOLD"

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_decision(self, decision: str, position: dict):
        """Execute a REBALANCE or COLLECT_FEES decision."""
        deadline = int(time.time()) + 600

        if decision == "REBALANCE":
            pool_state = self.state_reader.get_pool_state()
            current_tick = pool_state["tick"]
            new_lower = self.snap_tick(current_tick - self.RANGE_TICKS)
            new_upper = self.snap_tick(current_tick + self.RANGE_TICKS)

            token_id = position["token_id"]

            # Get current liquidity of the position
            pm = self.lp_manager.position_manager
            old_liquidity = pm.functions.getPositionLiquidity(token_id).call()

            # Withdraw liquidity first so wallet balances reflect available funds
            logger.info("Withdrawing liquidity before re-minting...")
            self.lp_manager.decrease_liquidity(token_id, old_liquidity, deadline)

            # Determine amounts from updated balances
            eth_balance = self.w3.eth.get_balance(self.account.address)
            usdc_balance = self.lp_manager.usdc.functions.balanceOf(
                self.account.address
            ).call()
            amount0_max = eth_balance // 2
            amount1_max = usdc_balance // 2

            logger.info(
                "REBALANCE: token_id=%d old_liq=%d -> new range [%d, %d]",
                token_id,
                old_liquidity,
                new_lower,
                new_upper,
            )

            result = self.lp_manager.mint_position(
                tick_lower=new_lower,
                tick_upper=new_upper,
                liquidity=self.DEFAULT_LIQUIDITY,
                amount0_max=amount0_max,
                amount1_max=amount1_max,
                deadline=deadline,
            )

            new_token_id = result["token_id"]
            # Update persisted positions: remove old, add new
            self.lp_manager.save_position(new_token_id, new_lower, new_upper)
            # Remove old position entry
            self.positions = [
                p for p in self.lp_manager.load_positions() if p["token_id"] != token_id
            ]
            # Re-save without the old one
            positions_file = os.path.join(os.path.dirname(__file__), "positions.json")
            with open(positions_file, "w") as f:
                json.dump(self.positions, f, indent=2)

            # Reload
            self.positions = self.lp_manager.load_positions()
            logger.info(
                "Rebalance complete: new token_id=%s  range=[%d, %d]",
                new_token_id,
                new_lower,
                new_upper,
            )

        elif decision == "COLLECT_FEES":
            token_id = position["token_id"]
            logger.info("COLLECT_FEES: token_id=%d", token_id)
            result = self.lp_manager.collect_fees(token_id, deadline)
            logger.info("Fee collection tx: %s", result["tx_hash"])
            self.last_fee_collect_ts = time.time()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Main agent loop: read state -> decide -> execute -> sleep."""
        self.setup()
        logger.info(
            "Agent running. Checking every %ds. Press Ctrl+C to stop.",
            config.TRADE_INTERVAL,
        )

        while True:
            try:
                pool_state = self.state_reader.get_pool_state()
                logger.info(
                    "Pool: tick=%d  price=%.2f  liquidity=%d",
                    pool_state["tick"],
                    pool_state["price"],
                    pool_state["liquidity"],
                )

                for position in self.positions:
                    decision = self.get_decision(pool_state, position)
                    self.log_decision(pool_state, position, decision)

                    if decision != "HOLD" and not self.rebalance_pending:
                        self.rebalance_pending = True
                        try:
                            self.execute_decision(decision, position)
                        finally:
                            self.rebalance_pending = False

                time.sleep(config.TRADE_INTERVAL)

            except KeyboardInterrupt:
                logger.info("Agent stopped by user.")
                break
            except Exception as e:
                logger.error("Error in agent loop: %s", e, exc_info=True)
                time.sleep(config.TRADE_INTERVAL)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_decision(self, pool_state: dict, position: dict, decision: str):
        """Log decision details to decisions.log."""
        current_tick = pool_state["tick"]
        tick_lower = position["tick_lower"]
        tick_upper = position["tick_upper"]
        range_width = tick_upper - tick_lower

        if range_width == 0:
            drift_ratio = 1.0
        elif current_tick >= tick_upper:
            drift_ratio = 1.0
        elif current_tick < tick_lower:
            drift_ratio = 1.0
        else:
            distance = min(current_tick - tick_lower, tick_upper - current_tick)
            drift_ratio = 1.0 - (distance / (range_width / 2))

        in_range = tick_lower <= current_tick < tick_upper

        logger.debug(
            "DECISION: %s | tick=%d price=%.2f | range=[%d,%d] | "
            "drift=%.3f in_range=%s | token_id=%s",
            decision,
            current_tick,
            pool_state["price"],
            tick_lower,
            tick_upper,
            drift_ratio,
            in_range,
            position.get("token_id"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def snap_tick(self, tick: int) -> int:
        """Snap a tick value to the nearest multiple of TICK_SPACING."""
        spacing = config.TICK_SPACING
        return round(tick / spacing) * spacing


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    # Default: anvil account #0 private key
    private_key = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    )
    agent = LPAgent(private_key)
    agent.run()
