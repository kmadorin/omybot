"""Evaluate PPO policy and compare against simple baselines."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env.lp_env import UniswapV4LPEnv


def _latest_file(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _pct(value: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (value / base) * 100.0


def _simulate_buy_and_hold(df: pd.DataFrame, initial_capital: float) -> dict[str, float]:
    if df.empty:
        return {"net_pnl": 0.0, "fees": 0.0, "il": 0.0, "gas": 0.0, "rebalances": 0.0}

    first = float(df.iloc[0]["price"])
    last = float(df.iloc[-1]["price"])
    if first <= 0:
        net = 0.0
    else:
        net = initial_capital * ((last / first) - 1.0)

    return {"net_pnl": net, "fees": 0.0, "il": 0.0, "gas": 0.0, "rebalances": 0.0}


def _simulate_static_lp(env: UniswapV4LPEnv) -> dict[str, float]:
    obs, _ = env.reset()
    done = False
    truncated = False
    info = {}
    # High threshold and neutral deltas to minimize rebalancing attempts.
    static_action = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    while not done and not truncated:
        obs, reward, done, truncated, info = env.step(static_action)
    del obs, reward
    return {
        "net_pnl": float(info.get("net_pnl", 0.0)),
        "fees": float(info.get("total_fees", 0.0)),
        "il": float(info.get("total_il", 0.0)),
        "gas": float(info.get("total_gas", 0.0)),
        "rebalances": float(info.get("num_rebalances", 0.0)),
    }


def evaluate_ppo(
    model: PPO,
    data_path: str,
    episodes: int,
) -> tuple[dict[str, float], list[float], list[np.ndarray]]:
    totals = defaultdict(float)
    portfolio_curves: list[list[float]] = []
    actions_all: list[np.ndarray] = []

    for episode in range(episodes):
        env = UniswapV4LPEnv(data_path=data_path)
        obs, _ = env.reset()
        done = False
        truncated = False
        info = {}
        capital_curve = [env.initial_capital]

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            actions_all.append(np.array(action, dtype=np.float32))
            obs, reward, done, truncated, info = env.step(action)
            capital_curve.append(float(info.get("capital", env.initial_capital)))
            del reward

        totals["net_pnl"] += float(info.get("net_pnl", 0.0))
        totals["fees"] += float(info.get("total_fees", 0.0))
        totals["il"] += float(info.get("total_il", 0.0))
        totals["gas"] += float(info.get("total_gas", 0.0))
        totals["rebalances"] += float(info.get("num_rebalances", 0.0))
        portfolio_curves.append(capital_curve)

        print(
            f"Episode {episode + 1}/{episodes} | net_pnl=${info.get('net_pnl', 0.0):.2f} | "
            f"fees=${info.get('total_fees', 0.0):.2f} | il=${info.get('total_il', 0.0):.2f} | "
            f"gas=${info.get('total_gas', 0.0):.2f} | rebalances={info.get('num_rebalances', 0)}"
        )

    means = {k: (v / max(episodes, 1)) for k, v in totals.items()}
    return means, [curve[-1] for curve in portfolio_curves], actions_all


def plot_results(portfolio_values: list[float], actions: list[np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(portfolio_values)
    plt.title("PPO Final Portfolio Value by Episode")
    plt.xlabel("Episode")
    plt.ylabel("Portfolio Value (USD)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "portfolio_value_over_time.png", dpi=150)
    plt.close()

    if actions:
        arr = np.array(actions, dtype=np.float32)
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        labels = [
            "lower_tick_delta",
            "upper_tick_delta",
            "liquidity_fraction_raw",
            "rebalance_threshold_raw",
        ]
        for i, ax in enumerate(axes.flatten()):
            ax.hist(arr[:, i], bins=30, alpha=0.8)
            ax.set_title(labels[i])
            ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "action_distribution.png", dpi=150)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO model on UniswapV4LPEnv")
    parser.add_argument("--model", type=str, default="omybot/ml/models/lp_ppo_model.zip", help="Path to model .zip")
    parser.add_argument("--data", type=str, default=None, help="Path to processed CSV")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--output-dir", type=str, default="omybot/ml/eval_outputs", help="Directory for plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    data_path = Path(args.data) if args.data else _latest_file(
        Path(__file__).resolve().parent.parent / "data" / "processed",
        "*.csv",
    )
    if data_path is None or not data_path.exists():
        raise FileNotFoundError(
            "Processed CSV not found. Provide --data or place CSV in omybot/ml/data/processed/."
        )

    model = PPO.load(str(model_path))
    ppo_metrics, ppo_portfolios, actions = evaluate_ppo(model, str(data_path), args.episodes)

    baseline_env = UniswapV4LPEnv(data_path=str(data_path))
    static_metrics = _simulate_static_lp(baseline_env)
    buy_hold_metrics = _simulate_buy_and_hold(baseline_env.df, baseline_env.initial_capital)

    initial_capital = baseline_env.initial_capital

    print()
    print("Strategy          | Net PnL | Fees | IL   | Gas  | Rebalances")
    print(
        "PPO Agent         | "
        f"{_pct(ppo_metrics['net_pnl'], initial_capital):+5.2f}% | "
        f"{_pct(ppo_metrics['fees'], initial_capital):5.2f}% | "
        f"{_pct(ppo_metrics['il'], initial_capital):5.2f}% | "
        f"{_pct(ppo_metrics['gas'], initial_capital):5.2f}% | "
        f"{ppo_metrics['rebalances']:.0f}"
    )
    print(
        "Static LP         | "
        f"{_pct(static_metrics['net_pnl'], initial_capital):+5.2f}% | "
        f"{_pct(static_metrics['fees'], initial_capital):5.2f}% | "
        f"{_pct(static_metrics['il'], initial_capital):5.2f}% | "
        f"{_pct(static_metrics['gas'], initial_capital):5.2f}% | "
        f"{static_metrics['rebalances']:.0f}"
    )
    print(
        "Buy & Hold        | "
        f"{_pct(buy_hold_metrics['net_pnl'], initial_capital):+5.2f}% | "
        f"{_pct(buy_hold_metrics['fees'], initial_capital):5.2f}% | "
        f"{_pct(buy_hold_metrics['il'], initial_capital):5.2f}% | "
        f"{_pct(buy_hold_metrics['gas'], initial_capital):5.2f}% | "
        f"{buy_hold_metrics['rebalances']:.0f}"
    )

    plot_results(ppo_portfolios, actions, Path(args.output_dir))
    print(f"\nSaved plots to: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
