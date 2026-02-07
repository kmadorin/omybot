"""Train PPO for Uniswap V4 LP management."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env.lp_env import UniswapV4LPEnv


class SaveOnBestRewardCallback(BaseCallback):
    """Save model when mean training reward improves."""

    def __init__(self, check_freq: int = 1_000, save_dir: str = "models", verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_dir = Path(save_dir)
        self.best_mean_reward = -np.inf
        self.best_model_path = self.save_dir / "best_model" / "best_lp_ppo_model"

    def _init_callback(self) -> None:
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        mean_reward = self.model.logger.name_to_value.get("rollout/ep_rew_mean")
        if mean_reward is None:
            return True

        if self.verbose > 0:
            print(
                f"Num timesteps: {self.num_timesteps} | "
                f"Best mean reward: {self.best_mean_reward:.6f} | "
                f"Last mean reward: {mean_reward:.6f}"
            )

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = float(mean_reward)
            if self.verbose > 0:
                print(f"Saving new best model to {self.best_model_path}.zip")
            self.model.save(str(self.best_model_path))
        return True


def _latest_processed_csv() -> Path | None:
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    if not processed_dir.exists():
        return None
    csv_files = sorted(processed_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csv_files[0] if csv_files else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent on UniswapV4LPEnv")
    parser.add_argument("--data", type=str, default=None, help="Path to processed CSV data")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Training timesteps")
    parser.add_argument("--log-dir", type=str, default="omybot/ml/logs", help="TensorBoard/log directory")
    parser.add_argument("--models-dir", type=str, default="omybot/ml/models", help="Directory to save models")
    return parser.parse_args()


def make_env(data_path: str, monitor_path: str | None = None):
    def _factory():
        env = UniswapV4LPEnv(data_path=data_path)
        if monitor_path is not None:
            return Monitor(env, monitor_path)
        return Monitor(env)

    return _factory


def main() -> None:
    args = parse_args()

    data_path = Path(args.data) if args.data else _latest_processed_csv()
    if data_path is None:
        raise FileNotFoundError(
            "No processed CSV found. Provide --data or place CSV in omybot/ml/data/processed/."
        )
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}")

    log_dir = Path(args.log_dir)
    models_dir = Path(args.models_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training data: {data_path}")
    print(f"Log dir: {log_dir}")
    print(f"Models dir: {models_dir}")

    env = DummyVecEnv([make_env(str(data_path), str(log_dir / "train_monitor.csv"))])
    eval_env = DummyVecEnv([make_env(str(data_path), str(log_dir / "eval_monitor.csv"))])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(log_dir),
    )

    save_best_callback = SaveOnBestRewardCallback(
        check_freq=1_000,
        save_dir=str(models_dir),
        verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir / "eval_best"),
        log_path=str(log_dir / "eval"),
        eval_freq=10_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    print("Starting PPO training...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[save_best_callback, eval_callback],
        progress_bar=False,
    )

    final_model_path = models_dir / "lp_ppo_model"
    model.save(str(final_model_path))
    print(f"Saved final model: {final_model_path}.zip")


if __name__ == "__main__":
    main()
