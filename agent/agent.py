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
from collections import deque
from datetime import datetime, timezone

import numpy as np

from web3 import Web3

# Sibling module imports
from state_reader import StateReader
from lp_manager import LPManager
import config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = os.path.dirname(__file__)
DECISIONS_DIR = os.path.join(LOG_DIR, "decisions")
os.makedirs(DECISIONS_DIR, exist_ok=True)
DECISIONS_LOG = os.path.join(DECISIONS_DIR, "decisions.log")
DECISIONS_JSONL = os.path.join(DECISIONS_DIR, "decisions.jsonl")

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
    # PPO observation normalization and bounds
    TICK_NORM = 100_000.0
    MAX_PPO_RANGE = 5000
    MAX_RANGE_WIDTH = 10_000
    MAX_REBALANCE_TIME = 86_400.0
    VOL_5M_WINDOW = 20
    VOL_1H_WINDOW = 240
    MIN_SETUP_USDC_RAW = 1_000_000  # 1 USDC minimum for first mint path
    REBALANCE_FAILURE_BASE_BACKOFF = 60
    REBALANCE_FAILURE_MAX_BACKOFF = 300

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
        self.last_rebalance_ts = time.time()
        self.consecutive_rebalance_failures = 0
        self.next_rebalance_allowed_ts = 0.0

        # Runtime feature tracking for PPO observations
        self.price_history: deque[float] = deque(maxlen=self.VOL_1H_WINDOW + 1)
        self.return_history: deque[float] = deque(maxlen=self.VOL_1H_WINDOW)
        self._last_history_signature: tuple[int, int, int] | None = None
        self.position_entry_prices: dict[int, float] = {}
        for position in self.positions:
            token_id = position.get("token_id")
            if token_id is not None and "entry_price" in position:
                self.position_entry_prices[token_id] = float(position["entry_price"])
        self._validate_persisted_positions()
        self._ensure_entry_prices_for_positions()

        # Optional PPO model loading
        self.model = None
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "ml", "models", "lp_ppo_model.zip"
        )
        model_path = os.path.abspath(model_path)
        if os.path.exists(model_path):
            try:
                from stable_baselines3 import PPO

                self.model = PPO.load(model_path)
                logger.info("Loaded PPO model from %s", model_path)
            except Exception as e:
                logger.error(
                    "Failed to load PPO model from %s, falling back to rule-based: %s",
                    model_path,
                    e,
                )
                self.model = None
        else:
            logger.info("No PPO model found, using rule-based decisions only")

    def _derive_entry_price_from_position(self, position: dict) -> float | None:
        """Derive entry price for legacy persisted positions without entry_price."""
        try:
            tick_lower = int(position["tick_lower"])
            tick_upper = int(position["tick_upper"])
            midpoint_tick = (tick_lower + tick_upper) // 2
            return float(self.state_reader.tick_to_price(midpoint_tick))
        except Exception:
            return None

    def _ensure_entry_prices_for_positions(self) -> None:
        """Backfill missing entry prices in memory and positions.json."""
        updated = False
        for position in self.positions:
            token_id = position.get("token_id")
            if token_id is None:
                continue
            if token_id in self.position_entry_prices:
                continue

            raw_entry = position.get("entry_price")
            if raw_entry is not None:
                try:
                    entry_price = float(raw_entry)
                except (TypeError, ValueError):
                    entry_price = None
            else:
                entry_price = None

            if entry_price is None:
                entry_price = self._derive_entry_price_from_position(position)
                if entry_price is None:
                    continue
                position["entry_price"] = entry_price
                updated = True

            self.position_entry_prices[token_id] = entry_price

        if updated:
            positions_file = os.path.join(os.path.dirname(__file__), "positions", "positions.json")
            with open(positions_file, "w") as f:
                json.dump(self.positions, f, indent=2)
            logger.info("Backfilled missing entry_price for legacy positions")

    def _save_positions(self) -> None:
        positions_file = os.path.join(os.path.dirname(__file__), "positions", "positions.json")
        with open(positions_file, "w") as f:
            json.dump(self.positions, f, indent=2)

    def _remove_position(self, token_id: int, reason: str) -> None:
        before = len(self.positions)
        self.positions = [p for p in self.positions if p.get("token_id") != token_id]
        self.position_entry_prices.pop(token_id, None)
        if len(self.positions) != before:
            self._save_positions()
            logger.warning("Removed stale position token_id=%s (%s)", token_id, reason)

    def _validate_persisted_positions(self) -> None:
        """Drop persisted positions that are invalid for the current fork/account."""
        if not self.positions:
            return

        valid_positions = []
        pm = self.lp_manager.position_manager
        for position in self.positions:
            token_id = position.get("token_id")
            if token_id is None:
                continue
            try:
                owner = pm.functions.ownerOf(token_id).call()
                if owner.lower() != self.account.address.lower():
                    logger.warning(
                        "Ignoring persisted token_id=%s (owner=%s, expected=%s)",
                        token_id,
                        owner,
                        self.account.address,
                    )
                    continue
                liquidity = pm.functions.getPositionLiquidity(token_id).call()
                if int(liquidity) <= 0:
                    logger.warning(
                        "Ignoring persisted token_id=%s (zero liquidity on current fork)",
                        token_id,
                    )
                    continue
            except Exception as e:
                logger.warning(
                    "Ignoring persisted token_id=%s (not readable on current fork: %s)",
                    token_id,
                    e,
                )
                continue

            valid_positions.append(position)

        if len(valid_positions) != len(self.positions):
            self.positions = valid_positions
            self._save_positions()
            logger.info("Retained %d valid persisted position(s)", len(self.positions))

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
        if usdc_balance < self.MIN_SETUP_USDC_RAW:
            raise RuntimeError(
                "Agent wallet has insufficient USDC for initial mint. "
                "Fund via whale impersonation (see research/CLAUDE_CODE_GUIDE.md Step 6.2) "
                "or run `python run_e2e.py` from omybot/."
            )

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
        self.lp_manager.save_position(
            token_id,
            tick_lower,
            tick_upper,
            entry_price=pool_state["price"],
        )
        self.positions = self.lp_manager.load_positions()
        self.position_entry_prices[token_id] = pool_state["price"]
        logger.info(
            "Initial position minted: token_id=%s  range=[%d, %d]",
            token_id,
            tick_lower,
            tick_upper,
        )

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _calculate_drift(self, pool_state: dict, position: dict) -> float:
        """Calculate 0..1 drift ratio toward range edges."""
        current_tick = pool_state["tick"]
        tick_lower = position["tick_lower"]
        tick_upper = position["tick_upper"]
        range_width = tick_upper - tick_lower

        if range_width <= 0:
            return 1.0
        if current_tick >= tick_upper or current_tick <= tick_lower:
            return 1.0

        distance_to_nearest_edge = min(
            current_tick - tick_lower,
            tick_upper - current_tick,
        )
        return 1.0 - (distance_to_nearest_edge / (range_width / 2))

    def _calculate_il_pct(self, entry_price: float, current_price: float) -> float:
        """Standard IL formula used by the training environment."""
        if entry_price <= 0 or current_price <= 0:
            return 0.0
        price_ratio = current_price / entry_price
        sqrt_ratio = np.sqrt(price_ratio)
        return float((2.0 * sqrt_ratio / (1.0 + price_ratio)) - 1.0)

    def _estimate_fees_accrued_norm(self, position: dict) -> float:
        """Estimate a normalized fee-accrual signal from fee growth inside range."""
        try:
            fee0, fee1 = self.state_reader.get_fee_growth_inside(
                position["tick_lower"], position["tick_upper"]
            )
            fee_growth_norm = (float(fee0) + float(fee1)) / float(2**128)
            return max(0.0, min(fee_growth_norm, 1.0))
        except Exception:
            return 0.0

    def _compute_volatility_features(self) -> tuple[float, float]:
        """Compute rolling volatility from live price log returns."""
        returns = np.array(self.return_history, dtype=np.float32)
        if returns.size == 0:
            return 0.0, 0.0

        vol_5m = float(np.std(returns[-self.VOL_5M_WINDOW :]))
        vol_1h = float(np.std(returns[-self.VOL_1H_WINDOW :]))
        return vol_5m, vol_1h

    def _update_market_history(self, pool_state: dict) -> None:
        """Update price/return history once per unique pool snapshot."""
        signature = (
            int(pool_state.get("sqrtPriceX96", 0)),
            int(pool_state.get("tick", 0)),
            int(pool_state.get("liquidity", 0)),
        )
        if signature == self._last_history_signature:
            return

        current_price = float(pool_state["price"])
        if self.price_history:
            prev_price = self.price_history[-1]
            if prev_price > 0 and current_price > 0:
                self.return_history.append(float(np.log(current_price / prev_price)))
        self.price_history.append(current_price)
        self._last_history_signature = signature

    def build_observation(self, pool_state: dict, position: dict) -> np.ndarray:
        """Build the 10-feature observation vector expected by the PPO env."""
        current_tick = float(pool_state["tick"])
        pool_liquidity = float(pool_state.get("liquidity", 0))
        current_price = float(pool_state["price"])

        self._update_market_history(pool_state)

        lower_tick = float(position["tick_lower"])
        upper_tick = float(position["tick_upper"])
        in_range = 1.0 if lower_tick <= current_tick <= upper_tick else 0.0

        fees_accrued_norm = self._estimate_fees_accrued_norm(position)

        token_id = position.get("token_id")
        if token_id is not None and token_id in self.position_entry_prices:
            entry_price = self.position_entry_prices[token_id]
        else:
            derived_entry_price = self._derive_entry_price_from_position(position)
            entry_price = (
                float(position["entry_price"])
                if position.get("entry_price") is not None
                else (
                    derived_entry_price
                    if derived_entry_price is not None
                    else current_price
                )
            )
            if token_id is not None and entry_price > 0:
                self.position_entry_prices[token_id] = entry_price
        il_pct = self._calculate_il_pct(entry_price, current_price)

        vol_5m, vol_1h = self._compute_volatility_features()
        time_since_rebalance_norm = min(
            (time.time() - self.last_rebalance_ts) / self.MAX_REBALANCE_TIME,
            1.0,
        )

        obs = np.array(
            [
                current_tick / self.TICK_NORM,
                np.log10(max(pool_liquidity, 1.0)),
                lower_tick / self.TICK_NORM,
                upper_tick / self.TICK_NORM,
                fees_accrued_norm,
                il_pct,
                vol_5m,
                vol_1h,
                time_since_rebalance_norm,
                in_range,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _ppo_proposed_range(self, current_tick: int, action: np.ndarray) -> tuple[int, int] | None:
        """Compute rebalance ticks from PPO action. Returns None when invalid."""
        new_lower = self.snap_tick(current_tick + int(float(action[0]) * self.MAX_PPO_RANGE))
        new_upper = self.snap_tick(current_tick + int(float(action[1]) * self.MAX_PPO_RANGE))

        if new_lower > new_upper:
            new_lower, new_upper = new_upper, new_lower

        min_width = config.TICK_SPACING * 2
        width = new_upper - new_lower
        if width < min_width:
            return None

        if width > self.MAX_RANGE_WIDTH:
            midpoint = (new_lower + new_upper) // 2
            half_width = self.MAX_RANGE_WIDTH // 2
            new_lower = self.snap_tick(midpoint - half_width)
            new_upper = self.snap_tick(midpoint + half_width)

            if new_upper - new_lower < min_width:
                return None

        # Reject PPO proposals that do not bracket the current price.
        # This prevents pathological perpetual out-of-range loops.
        if not (new_lower <= current_tick <= new_upper):
            return None

        return new_lower, new_upper

    def _rule_based_decision(self, pool_state: dict, position: dict) -> str:
        """Rule-based decision: REBALANCE, COLLECT_FEES, or HOLD."""
        drift_ratio = self._calculate_drift(pool_state, position)

        # Decision
        if drift_ratio > config.REBALANCE_DRIFT_THRESHOLD:
            return "REBALANCE"

        if self._should_collect_fees(pool_state, position, drift_ratio):
            return "COLLECT_FEES"

        return "HOLD"

    def _should_collect_fees(
        self, pool_state: dict, position: dict, drift_ratio: float | None = None
    ) -> bool:
        """Periodic fee collection policy shared by rule and PPO paths."""
        if drift_ratio is None:
            drift_ratio = self._calculate_drift(pool_state, position)

        current_tick = pool_state["tick"]
        tick_lower = position["tick_lower"]
        tick_upper = position["tick_upper"]
        in_range = tick_lower <= current_tick <= tick_upper
        if not in_range or drift_ratio >= 0.3:
            return False

        now = time.time()
        return (now - self.last_fee_collect_ts) >= self.FEE_COLLECT_COOLDOWN

    def get_decision(
        self, pool_state: dict, position: dict
    ) -> tuple[str, np.ndarray | None, str]:
        """Try PPO first, then fall back to rule-based logic."""
        if self.model is None:
            decision = self._rule_based_decision(pool_state, position)
            logger.info("Decision source=RULE decision=%s", decision)
            return decision, None, "RULE"

        try:
            obs = self.build_observation(pool_state, position)
            action, _ = self.model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)

            threshold = 0.5 + (float(action[3]) + 1.0) / 2.0 * 0.45
            drift_ratio = self._calculate_drift(pool_state, position)

            if drift_ratio > threshold:
                proposed = self._ppo_proposed_range(pool_state["tick"], action)
                if proposed is None:
                    decision = self._rule_based_decision(pool_state, position)
                    logger.info(
                        "Decision source=RULE fallback=invalid_ppo_range decision=%s",
                        decision,
                    )
                    return decision, None, "RULE"
                logger.info("Decision source=PPO decision=REBALANCE threshold=%.3f", threshold)
                return "REBALANCE", action, "PPO"

            if self._should_collect_fees(pool_state, position, drift_ratio):
                logger.info("Decision source=PPO+RULE decision=COLLECT_FEES")
                return "COLLECT_FEES", action, "PPO+RULE"

            logger.info("Decision source=PPO decision=HOLD threshold=%.3f", threshold)
            return "HOLD", action, "PPO"
        except Exception as e:
            logger.error("PPO decision failed, falling back to rules: %s", e, exc_info=True)
            decision = self._rule_based_decision(pool_state, position)
            logger.info("Decision source=RULE fallback=ppo_error decision=%s", decision)
            return decision, None, "RULE"

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_decision(
        self, decision: str, position: dict, ppo_action: np.ndarray | None = None
    ):
        """Execute a REBALANCE or COLLECT_FEES decision."""
        deadline = int(time.time()) + 600

        if decision == "REBALANCE":
            pool_state = self.state_reader.get_pool_state()
            current_tick = pool_state["tick"]
            if ppo_action is not None:
                proposed_range = self._ppo_proposed_range(current_tick, ppo_action)
                if proposed_range is not None:
                    new_lower, new_upper = proposed_range
                else:
                    new_lower = self.snap_tick(current_tick - self.RANGE_TICKS)
                    new_upper = self.snap_tick(current_tick + self.RANGE_TICKS)
                    logger.warning(
                        "Invalid PPO rebalance range at execution time; using rule-based default range"
                    )
            else:
                new_lower = self.snap_tick(current_tick - self.RANGE_TICKS)
                new_upper = self.snap_tick(current_tick + self.RANGE_TICKS)

            token_id = position["token_id"]

            # Get current liquidity of the position
            pm = self.lp_manager.position_manager
            try:
                owner = pm.functions.ownerOf(token_id).call()
                if owner.lower() != self.account.address.lower():
                    self._remove_position(token_id, "token no longer owned by agent")
                    return
                old_liquidity = pm.functions.getPositionLiquidity(token_id).call()
            except Exception as e:
                self._remove_position(token_id, f"token unreadable ({e})")
                return
            if int(old_liquidity) <= 0:
                self._remove_position(token_id, "zero liquidity")
                return

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

            liquidity_to_mint = self.DEFAULT_LIQUIDITY
            if ppo_action is not None:
                # action[2] is liquidity_fraction_raw in [-1, 1] -> [0.5, 1.0]
                liq_fraction = 0.5 + (float(ppo_action[2]) + 1.0) / 2.0 * 0.5
                baseline_liquidity = old_liquidity if old_liquidity > 0 else self.DEFAULT_LIQUIDITY
                liquidity_to_mint = max(1, int(baseline_liquidity * liq_fraction))
                logger.info(
                    "Applying PPO liquidity fraction: raw=%.4f fraction=%.4f mint_liquidity=%d",
                    float(ppo_action[2]),
                    liq_fraction,
                    liquidity_to_mint,
                )

            result = self.lp_manager.mint_position(
                tick_lower=new_lower,
                tick_upper=new_upper,
                liquidity=liquidity_to_mint,
                amount0_max=amount0_max,
                amount1_max=amount1_max,
                deadline=deadline,
            )

            new_token_id = result["token_id"]
            # Update persisted positions: remove old, add new
            self.lp_manager.save_position(
                new_token_id,
                new_lower,
                new_upper,
                entry_price=pool_state["price"],
            )
            self.position_entry_prices[new_token_id] = pool_state["price"]
            self.position_entry_prices.pop(token_id, None)
            # Remove old position entry
            self.positions = [
                p for p in self.lp_manager.load_positions() if p["token_id"] != token_id
            ]
            # Re-save without the old one
            self._save_positions()

            # Reload
            self.positions = self.lp_manager.load_positions()
            self.consecutive_rebalance_failures = 0
            self.next_rebalance_allowed_ts = 0.0
            self.last_rebalance_ts = time.time()
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
                    decision, ppo_action, source = self.get_decision(pool_state, position)
                    self.log_decision(pool_state, position, decision, source)

                    if decision != "HOLD" and not self.rebalance_pending:
                        if (
                            decision == "REBALANCE"
                            and time.time() < self.next_rebalance_allowed_ts
                        ):
                            cooldown_left = int(self.next_rebalance_allowed_ts - time.time())
                            logger.warning(
                                "Skipping REBALANCE due to backoff (%ds left)",
                                max(cooldown_left, 0),
                            )
                            continue
                        self.rebalance_pending = True
                        try:
                            self.execute_decision(decision, position, ppo_action)
                        except Exception as e:
                            if decision == "REBALANCE":
                                self.consecutive_rebalance_failures += 1
                                penalty = self.REBALANCE_FAILURE_BASE_BACKOFF * (
                                    2 ** max(0, self.consecutive_rebalance_failures - 1)
                                )
                                penalty = min(penalty, self.REBALANCE_FAILURE_MAX_BACKOFF)
                                self.next_rebalance_allowed_ts = time.time() + penalty
                                logger.error(
                                    "REBALANCE failed (%d consecutive). Backing off for %ds: %s",
                                    self.consecutive_rebalance_failures,
                                    penalty,
                                    e,
                                )
                            raise
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

    def log_decision(
        self, pool_state: dict, position: dict, decision: str, source: str = "RULE"
    ):
        """Log decision details to decisions.log."""
        current_tick = pool_state["tick"]
        tick_lower = position["tick_lower"]
        tick_upper = position["tick_upper"]
        drift_ratio = self._calculate_drift(pool_state, position)
        in_range = tick_lower <= current_tick <= tick_upper

        logger.debug(
            "DECISION: %s source=%s | tick=%d price=%.2f | range=[%d,%d] | "
            "drift=%.3f in_range=%s | token_id=%s",
            decision,
            source,
            current_tick,
            pool_state["price"],
            tick_lower,
            tick_upper,
            drift_ratio,
            in_range,
            position.get("token_id"),
        )

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "source": source,
            "tick": current_tick,
            "price": round(pool_state["price"], 2),
            "range": [tick_lower, tick_upper],
            "drift": round(drift_ratio, 4),
            "in_range": in_range,
            "token_id": position.get("token_id"),
        }
        with open(DECISIONS_JSONL, "a") as f:
            f.write(json.dumps(record) + "\n")

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
