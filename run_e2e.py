#!/usr/bin/env python3
"""Step 6 E2E runner: start anvil, fund accounts, run agent, simulate swaps, verify logs."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Proc:
    name: str
    process: subprocess.Popen


class E2ERunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path(__file__).resolve().parent
        self.agent_dir = self.root / "agent"
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.anvil_log = self.logs_dir / "step6_anvil.log"
        self.agent_stdout_log = self.logs_dir / "step6_agent_stdout.log"
        self.swap_log = self.logs_dir / "step6_swap_simulation.log"

        self.positions_file = self.agent_dir / "positions.json"
        self.decisions_log = self.agent_dir / "decisions.log"

        self.procs: list[Proc] = []

    def run(self) -> int:
        self._require_cmd("anvil")
        self._require_cmd("cast")

        python_bin = self._python_bin()
        self._ensure_python_deps(python_bin)

        try:
            self._start_anvil()
            self._prepare_state()
            self._fund_accounts()
            self._start_agent(python_bin)
            self._run_swap_simulation(python_bin)
            time.sleep(self.args.post_swap_wait)
            return self._verify()
        finally:
            if not self.args.keep_running:
                self._stop_all()

    def _python_bin(self) -> str:
        venv_python = self.root / ".venv" / "bin" / "python"
        if venv_python.exists() and os.access(venv_python, os.X_OK):
            return str(venv_python)
        return sys.executable or "python3"

    def _require_cmd(self, cmd: str) -> None:
        if shutil.which(cmd) is None:
            raise RuntimeError(f"missing required command: {cmd}")

    def _ensure_python_deps(self, python_bin: str) -> None:
        test = subprocess.run(
            [python_bin, "-c", "import web3, eth_abi"],
            capture_output=True,
            text=True,
        )
        if test.returncode == 0:
            return

        print("[step6] installing python deps from agent/requirements.txt", flush=True)
        subprocess.run(
            [python_bin, "-m", "pip", "install", "-r", str(self.agent_dir / "requirements.txt")],
            check=True,
        )

    def _start_anvil(self) -> None:
        for path in (self.anvil_log, self.agent_stdout_log, self.swap_log):
            if path.exists():
                path.unlink()

        print(f"[step6] starting anvil fork: {self.args.fork_url}", flush=True)
        anvil_log_f = self.anvil_log.open("w")
        proc = subprocess.Popen(
            [
                "anvil",
                "--fork-url",
                self.args.fork_url,
                "--chain-id",
                str(self.args.chain_id),
                "--block-time",
                str(self.args.block_time),
            ],
            stdout=anvil_log_f,
            stderr=subprocess.STDOUT,
            cwd=self.root,
        )
        self.procs.append(Proc("anvil", proc))

        for _ in range(45):
            if self._cast(["block-number"]).returncode == 0:
                print("[step6] anvil is ready", flush=True)
                return
            time.sleep(1)

        raise RuntimeError("anvil did not become ready in time")

    def _prepare_state(self) -> None:
        self._backup_if_exists(self.positions_file)
        self._backup_if_exists(self.decisions_log)
        for path in (self.positions_file, self.decisions_log):
            if path.exists():
                path.unlink()

    def _backup_if_exists(self, path: Path) -> None:
        if not path.exists():
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = path.with_suffix(path.suffix + f".bak.{ts}")
        shutil.copy2(path, backup)

    def _cast(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["cast", *args, "--rpc-url", self.args.rpc_url],
            capture_output=True,
            text=True,
        )

    def _cast_send_transfer(self, to_addr: str, amount_raw: int) -> None:
        subprocess.run(
            [
                "cast",
                "send",
                self.args.usdc,
                "transfer(address,uint256)",
                to_addr,
                str(amount_raw),
                "--from",
                self.args.whale,
                "--unlocked",
                "--rpc-url",
                self.args.rpc_url,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def _fund_accounts(self) -> None:
        print("[step6] funding agent and swap accounts with USDC", flush=True)
        self._cast_send_transfer(self.args.agent_addr, self.args.agent_usdc_raw)
        self._cast_send_transfer(self.args.swap_addr, self.args.swap_usdc_raw)

    def _start_agent(self, python_bin: str) -> None:
        print("[step6] starting agent", flush=True)
        agent_log_f = self.agent_stdout_log.open("w")
        proc = subprocess.Popen(
            [python_bin, "agent.py", self.args.agent_pk],
            cwd=self.agent_dir,
            stdout=agent_log_f,
            stderr=subprocess.STDOUT,
        )
        self.procs.append(Proc("agent", proc))

        for _ in range(40):
            if proc.poll() is not None:
                tail = self._tail(self.agent_stdout_log, 80)
                raise RuntimeError(f"agent exited unexpectedly\n{tail}")

            if "Agent running" in self._tail(self.agent_stdout_log, 100):
                return
            time.sleep(1)

        tail = self._tail(self.agent_stdout_log, 120)
        raise RuntimeError(f"agent did not reach running state in time\n{tail}")

    def _run_swap_simulation(self, python_bin: str) -> None:
        print("[step6] running swap simulation", flush=True)
        with self.swap_log.open("w") as f:
            subprocess.run(
                [
                    python_bin,
                    "simulate_swaps.py",
                    "--rpc-url",
                    self.args.rpc_url,
                    "--private-key",
                    self.args.swap_pk,
                    "--cycles",
                    str(self.args.swap_cycles),
                    "--interval",
                    str(self.args.swap_interval),
                    "--eth-in",
                    str(self.args.swap_eth_in),
                    "--usdc-in",
                    str(self.args.swap_usdc_in),
                ],
                cwd=self.agent_dir,
                check=True,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

    def _verify(self) -> int:
        print("[step6] verification checklist", flush=True)

        checks = [
            (
                "Initial position minted",
                self._contains(self.agent_stdout_log, "Initial position minted"),
            ),
            (
                "Pool state reads every loop",
                self._count_contains(self.agent_stdout_log, "Pool: tick=") >= 2,
            ),
            (
                "Swap simulation produced price movement logs",
                self._count_contains(self.swap_log, "cycle=") >= 2,
            ),
            (
                "Decisions log generated",
                self.decisions_log.exists() and self._contains(self.decisions_log, "DECISION:"),
            ),
            (
                "Rebalance decision observed",
                self._contains(self.agent_stdout_log, "decision=REBALANCE")
                or self._contains(self.decisions_log, "DECISION: REBALANCE"),
            ),
            (
                "Rebalance execution observed",
                self._contains(self.agent_stdout_log, "Rebalance complete"),
            ),
        ]

        all_passed = True
        for label, ok in checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {label}", flush=True)
            all_passed = all_passed and ok

        if not all_passed:
            print("[step6] one or more checks failed", flush=True)
            print("[step6] last agent logs:", flush=True)
            print(self._tail(self.agent_stdout_log, 80), flush=True)
            print("[step6] last swap logs:", flush=True)
            print(self._tail(self.swap_log, 80), flush=True)
            return 1

        print("[step6] all checks passed", flush=True)
        print("[step6] logs:", flush=True)
        print(f"  anvil: {self.anvil_log}", flush=True)
        print(f"  agent: {self.agent_stdout_log}", flush=True)
        print(f"  swaps: {self.swap_log}", flush=True)
        print(f"  decisions: {self.decisions_log}", flush=True)
        if self.args.keep_running:
            print("[step6] keep-running enabled, anvil+agent are still running", flush=True)

        return 0

    def _contains(self, path: Path, needle: str) -> bool:
        if not path.exists():
            return False
        return needle in path.read_text(errors="ignore")

    def _count_contains(self, path: Path, needle: str) -> int:
        if not path.exists():
            return 0
        return path.read_text(errors="ignore").count(needle)

    def _tail(self, path: Path, lines: int) -> str:
        if not path.exists():
            return ""
        content = path.read_text(errors="ignore").splitlines()
        return "\n".join(content[-lines:])

    def _stop_all(self) -> None:
        for proc in reversed(self.procs):
            if proc.process.poll() is not None:
                continue
            proc.process.terminate()
            try:
                proc.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.process.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 6 E2E on local Base fork")

    parser.add_argument("--rpc-url", default=os.getenv("ANVIL_RPC_URL", "http://localhost:8545"))
    parser.add_argument("--fork-url", default=os.getenv("FORK_URL", "https://mainnet.base.org"))
    parser.add_argument("--chain-id", type=int, default=int(os.getenv("CHAIN_ID", "8453")))
    parser.add_argument("--block-time", type=int, default=int(os.getenv("BLOCK_TIME", "2")))

    parser.add_argument(
        "--agent-pk",
        default=os.getenv(
            "AGENT_PK",
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        ),
    )
    parser.add_argument(
        "--swap-pk",
        default=os.getenv(
            "SWAP_PK",
            "0x59c6995e998f97a5a0044966f094538e5b0d2cefe4f7fcca7e5fca9c7e5e5fbb",
        ),
    )

    parser.add_argument(
        "--agent-addr",
        default=os.getenv("AGENT_ADDR", "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"),
    )
    parser.add_argument(
        "--swap-addr",
        default=os.getenv("SWAP_ADDR", "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"),
    )
    parser.add_argument(
        "--usdc",
        default=os.getenv("USDC", "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"),
    )
    parser.add_argument(
        "--whale",
        default=os.getenv("WHALE", "0xBaeD383EDE0e5d9d72430661f3285DAa77E9439F"),
    )

    parser.add_argument("--agent-usdc-raw", type=int, default=int(os.getenv("AGENT_USDC_RAW", "10000000000")))
    parser.add_argument("--swap-usdc-raw", type=int, default=int(os.getenv("SWAP_USDC_RAW", "50000000000")))

    parser.add_argument("--swap-cycles", type=int, default=int(os.getenv("SWAP_CYCLES", "6")))
    parser.add_argument("--swap-interval", type=int, default=int(os.getenv("SWAP_INTERVAL", "20")))
    parser.add_argument("--swap-eth-in", type=float, default=float(os.getenv("SWAP_ETH_IN", "3.0")))
    parser.add_argument("--swap-usdc-in", type=float, default=float(os.getenv("SWAP_USDC_IN", "6000.0")))

    parser.add_argument("--post-swap-wait", type=int, default=int(os.getenv("POST_SWAP_WAIT", "20")))
    parser.add_argument("--keep-running", action="store_true", default=os.getenv("KEEP_RUNNING", "0") == "1")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = E2ERunner(args)
    try:
        return runner.run()
    except KeyboardInterrupt:
        print("\n[step6] interrupted", flush=True)
        return 130
    except Exception as exc:
        print(f"[step6] failed: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
