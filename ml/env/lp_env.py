"""
Gymnasium environment for training a PPO agent to manage Uniswap V4 LP positions.

Simulates LP position management using historical swap data. The agent decides
when and how to rebalance its concentrated liquidity position.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class UniswapV4LPEnv(gym.Env):
    """
    RL environment for Uniswap V4 LP management.

    Observation (10 features):
        0: current_tick (normalized)
        1: pool_liquidity_log (log-scaled)
        2: position_lower_tick (normalized)
        3: position_upper_tick (normalized)
        4: fees_accrued_normalized
        5: impermanent_loss_pct
        6: volatility_5m
        7: volatility_1h
        8: time_since_rebalance_normalized
        9: position_in_range_flag

    Action (4 continuous):
        0: lower_tick_delta [-1, 1] -> tick offset below current
        1: upper_tick_delta [-1, 1] -> tick offset above current
        2: liquidity_fraction [-1, 1] -> [0.5, 1.0]
        3: rebalance_threshold [-1, 1] -> [0.5, 0.95]
    """

    metadata = {"render_modes": []}

    # Normalization constant for ticks (typical range is roughly -100k to +100k)
    TICK_NORM = 100_000.0
    # Maximum tick offset from current price for position edges
    MAX_TICK_RANGE = 5000
    # Fee rate for the 0.05% pool in decimal
    FEE_RATE = 500 / 1_000_000  # 0.0005
    # Cap for time_since_rebalance normalization (24 hours in seconds)
    MAX_REBALANCE_TIME = 86400.0

    def __init__(
        self,
        data_path: str,
        tick_spacing: int = 10,
        initial_capital: float = 10000.0,
        gas_cost_usd: float = 0.50,
    ):
        super().__init__()

        self.tick_spacing = tick_spacing
        self.initial_capital = initial_capital
        self.gas_cost_usd = gas_cost_usd

        # Load and preprocess data
        self.df = self._load_data(data_path)
        self.n_steps = len(self.df)

        # Observation: 10 continuous features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # Action: 4 continuous values in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # State variables (set in reset)
        self.current_step = 0
        self.position_lower = 0
        self.position_upper = 0
        self.position_liquidity = 0.0
        self.liquidity_fraction = 1.0
        self.entry_price = 0.0
        self.total_fees = 0.0
        self.total_il = 0.0
        self.total_gas = 0.0
        self.num_rebalances = 0
        self.last_rebalance_step = 0
        self.capital = initial_capital

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load CSV and precompute rolling features."""
        df = pd.read_csv(data_path)

        required_cols = {"timestamp", "price", "tick", "volume", "liquidity", "sqrtPriceX96"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Price returns (log returns for volatility calculation)
        df["price_returns"] = np.log(df["price"] / df["price"].shift(1))
        df["price_returns"] = df["price_returns"].fillna(0.0)

        # Estimate time delta between rows to compute rolling windows
        if len(df) > 1:
            median_dt = df["timestamp"].diff().median().total_seconds()
            if median_dt <= 0:
                median_dt = 60.0  # fallback to 1 minute
        else:
            median_dt = 60.0

        # Rolling volatility windows (number of rows)
        window_5m = max(int(300 / median_dt), 2)
        window_1h = max(int(3600 / median_dt), 2)

        df["volatility_5m"] = (
            df["price_returns"].rolling(window=window_5m, min_periods=1).std().fillna(0.0)
        )
        df["volatility_1h"] = (
            df["price_returns"].rolling(window=window_1h, min_periods=1).std().fillna(0.0)
        )

        # Volume moving average (5-min window)
        df["volume_ma"] = (
            df["volume"].rolling(window=window_5m, min_periods=1).mean().fillna(0.0)
        )

        # Convert timestamps to seconds for time-since-rebalance calc
        df["timestamp_s"] = (
            df["timestamp"] - df["timestamp"].iloc[0]
        ).dt.total_seconds()

        return df

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        row = self.df.iloc[0]

        current_tick = int(row["tick"])

        # Initialize position centered on current tick
        self.position_lower = self._snap_tick(current_tick - 1000)
        self.position_upper = self._snap_tick(current_tick + 1000)
        self.position_liquidity = self.initial_capital
        self.liquidity_fraction = 1.0
        self.entry_price = float(row["price"])
        self.total_fees = 0.0
        self.total_il = 0.0
        self.total_gas = 0.0
        self.num_rebalances = 0
        self.last_rebalance_step = 0
        self.capital = self.initial_capital

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        self.current_step += 1
        if self.current_step >= self.n_steps:
            obs = self._get_observation_terminal()
            return obs, 0.0, True, False, self._get_info()

        row = self.df.iloc[self.current_step]
        current_tick = int(row["tick"])
        current_price = float(row["price"])
        volume = float(row["volume"])
        pool_liquidity = float(row["liquidity"])

        # Decode action
        lower_delta = action[0]  # [-1, 1]
        upper_delta = action[1]  # [-1, 1]
        liq_frac_raw = action[2]  # [-1, 1] -> [0.5, 1.0]
        rebal_thresh_raw = action[3]  # [-1, 1] -> [0.5, 0.95]

        liquidity_fraction = 0.5 + (liq_frac_raw + 1.0) / 2.0 * 0.5
        rebalance_threshold = 0.5 + (rebal_thresh_raw + 1.0) / 2.0 * 0.45

        # Check if position is in range
        in_range = self.position_lower <= current_tick <= self.position_upper

        # Calculate drift ratio (how far toward the edge)
        range_width = self.position_upper - self.position_lower
        if range_width <= 0:
            drift_ratio = 1.0
        elif current_tick >= self.position_upper:
            drift_ratio = 1.0
        elif current_tick <= self.position_lower:
            drift_ratio = 1.0
        else:
            distance_to_nearest_edge = min(
                current_tick - self.position_lower,
                self.position_upper - current_tick,
            )
            drift_ratio = 1.0 - (distance_to_nearest_edge / (range_width / 2.0))

        # Decide whether to rebalance
        rebalanced = False
        gas_penalty = 0.0

        if drift_ratio > rebalance_threshold:
            # Execute rebalance
            new_lower = self._snap_tick(
                current_tick + int(lower_delta * self.MAX_TICK_RANGE)
            )
            new_upper = self._snap_tick(
                current_tick + int(upper_delta * self.MAX_TICK_RANGE)
            )

            # Ensure lower < upper with minimum width
            if new_lower >= new_upper:
                new_lower = self._snap_tick(current_tick - self.tick_spacing * 10)
                new_upper = self._snap_tick(current_tick + self.tick_spacing * 10)

            min_width = self.tick_spacing * 2
            if new_upper - new_lower < min_width:
                mid = (new_lower + new_upper) // 2
                new_lower = self._snap_tick(mid - self.tick_spacing * 5)
                new_upper = self._snap_tick(mid + self.tick_spacing * 5)

            self.position_lower = new_lower
            self.position_upper = new_upper
            self.liquidity_fraction = liquidity_fraction
            self.position_liquidity = self.capital * self.liquidity_fraction
            self.entry_price = current_price
            self.last_rebalance_step = self.current_step
            self.num_rebalances += 1

            gas_penalty = self.gas_cost_usd
            self.total_gas += gas_penalty
            rebalanced = True

        # Simulate fee accrual
        step_fees = self._simulate_fees(
            volume, self.position_liquidity, pool_liquidity, in_range
        )
        self.total_fees += step_fees

        # Calculate IL since last rebalance
        il = self._calculate_il(self.entry_price, current_price)
        il_loss = abs(il) * self.position_liquidity
        self.total_il = il_loss  # cumulative IL from entry

        # Update capital
        self.capital = self.initial_capital + self.total_fees - self.total_il - self.total_gas

        # Reward: fees - IL - gas (per step)
        reward = step_fees - abs(il) * self.position_liquidity * 0.01 - gas_penalty

        obs = self._get_observation()
        terminated = False
        truncated = False
        info = self._get_info()
        info["rebalanced"] = rebalanced

        return obs, float(reward), terminated, truncated, info

    def _snap_tick(self, tick: int) -> int:
        """Snap tick to nearest multiple of tick_spacing."""
        if self.tick_spacing <= 0:
            return tick
        return round(tick / self.tick_spacing) * self.tick_spacing

    def _calculate_il(self, entry_price: float, current_price: float) -> float:
        """
        Standard impermanent loss formula.
        Returns a negative value representing the loss fraction.
        IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
        """
        if entry_price <= 0 or current_price <= 0:
            return 0.0
        price_ratio = current_price / entry_price
        sqrt_ratio = np.sqrt(price_ratio)
        il = 2.0 * sqrt_ratio / (1.0 + price_ratio) - 1.0
        return il  # negative value

    def _simulate_fees(
        self,
        volume: float,
        position_liquidity: float,
        pool_liquidity: float,
        in_range: bool,
    ) -> float:
        """
        Simulate fee earnings for one step.
        fees = volume * fee_rate * (position_liquidity / pool_liquidity) * in_range
        """
        if not in_range or pool_liquidity <= 0 or position_liquidity <= 0:
            return 0.0
        liquidity_share = position_liquidity / pool_liquidity
        fees = volume * self.FEE_RATE * liquidity_share
        return fees

    def _get_observation(self) -> np.ndarray:
        """Build the 10-feature observation vector."""
        row = self.df.iloc[self.current_step]

        current_tick = float(row["tick"])
        pool_liquidity = float(row["liquidity"])
        current_price = float(row["price"])

        # 0: current_tick normalized
        tick_norm = current_tick / self.TICK_NORM

        # 1: pool_liquidity log-scaled
        liq_log = np.log10(max(pool_liquidity, 1.0))

        # 2: position_lower normalized
        lower_norm = float(self.position_lower) / self.TICK_NORM

        # 3: position_upper normalized
        upper_norm = float(self.position_upper) / self.TICK_NORM

        # 4: fees_accrued_normalized (ratio to initial capital)
        fees_norm = self.total_fees / max(self.initial_capital, 1.0)

        # 5: impermanent_loss_pct
        il_pct = self._calculate_il(self.entry_price, current_price)

        # 6: volatility_5m
        vol_5m = float(row["volatility_5m"])

        # 7: volatility_1h
        vol_1h = float(row["volatility_1h"])

        # 8: time_since_rebalance_normalized
        if self.current_step > self.last_rebalance_step and self.last_rebalance_step < len(self.df):
            current_ts = float(row["timestamp_s"])
            rebal_ts = float(self.df.iloc[self.last_rebalance_step]["timestamp_s"])
            time_since = current_ts - rebal_ts
        else:
            time_since = 0.0
        time_norm = min(time_since / self.MAX_REBALANCE_TIME, 1.0)

        # 9: position_in_range_flag
        in_range = 1.0 if self.position_lower <= int(row["tick"]) <= self.position_upper else 0.0

        obs = np.array(
            [tick_norm, liq_log, lower_norm, upper_norm, fees_norm,
             il_pct, vol_5m, vol_1h, time_norm, in_range],
            dtype=np.float32,
        )

        # Replace any NaN/inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _get_observation_terminal(self) -> np.ndarray:
        """Return the last valid observation when episode is done."""
        # Point to last valid row
        saved_step = self.current_step
        self.current_step = min(self.current_step, self.n_steps - 1)
        obs = self._get_observation()
        self.current_step = saved_step
        return obs

    def _get_info(self) -> dict:
        """Return episode info dict."""
        return {
            "total_fees": self.total_fees,
            "total_il": self.total_il,
            "total_gas": self.total_gas,
            "num_rebalances": self.num_rebalances,
            "capital": self.capital,
            "net_pnl": self.capital - self.initial_capital,
            "step": self.current_step,
        }
