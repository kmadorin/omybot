#!/usr/bin/env python3
"""One-command demo runner: bootstrap fork + agent, then launch dashboard."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from run_e2e import E2ERunner, parse_args as parse_e2e_args


def parse_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Run full OmyBot demo stack")
    parser.add_argument("--dashboard-port", type=int, default=5005)
    parser.add_argument(
        "--dashboard-host", default="0.0.0.0", help="Host for Flask dashboard bind"
    )
    parser.add_argument(
        "--live-swap-period",
        type=int,
        default=30,
        help="Seconds between live one-swap injections after startup (0 disables)",
    )
    parser.add_argument(
        "--live-swap-eth-in",
        type=float,
        default=3.0,
        help="ETH amount for live ETH->USDC swaps",
    )
    parser.add_argument(
        "--live-swap-usdc-in",
        type=float,
        default=6000.0,
        help="USDC amount for live USDC->ETH swaps",
    )
    demo_args, e2e_argv = parser.parse_known_args()

    # Reuse run_e2e argument schema for everything else.
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *e2e_argv]
        e2e_args = parse_e2e_args()
    finally:
        sys.argv = original_argv

    e2e_args.keep_running = True
    return demo_args, e2e_args


def main() -> int:
    demo_args, e2e_args = parse_args()
    runner = E2ERunner(e2e_args)
    dashboard_proc: subprocess.Popen | None = None

    try:
        runner._require_cmd("anvil")
        runner._require_cmd("cast")
        python_bin = runner._python_bin()
        runner._ensure_python_deps(python_bin)
        runner._start_anvil()
        runner._prepare_state()
        runner._fund_accounts()
        runner._start_agent(python_bin)

        dashboard_dir = runner.root / "dashboard"
        dashboard_log = runner.logs_dir / "e2e_dashboard.log"
        print(
            f"[demo] starting dashboard on http://localhost:{demo_args.dashboard_port}",
            flush=True,
        )
        with dashboard_log.open("w") as logf:
            dashboard_proc = subprocess.Popen(
                [python_bin, "server.py"],
                cwd=dashboard_dir,
                env={
                    **os.environ,
                    "DASHBOARD_HOST": demo_args.dashboard_host,
                    "DASHBOARD_PORT": str(demo_args.dashboard_port),
                },
                stdout=logf,
                stderr=subprocess.STDOUT,
            )

        print("[demo] running swap simulation", flush=True)
        runner._run_swap_simulation(python_bin)
        time.sleep(e2e_args.post_swap_wait)

        verify_rc = runner._verify()
        if verify_rc != 0:
            print("[demo] verification failed; keeping stack running for inspection", flush=True)

        print("[demo] stack is live", flush=True)
        if demo_args.live_swap_period > 0:
            print(
                f"[demo] live swaps enabled: one swap every {demo_args.live_swap_period}s",
                flush=True,
            )
        print("[demo] Ctrl+C to stop anvil + agent + dashboard", flush=True)

        live_swap_log = runner.logs_dir / "e2e_swap_live.log"
        next_live_swap_ts = time.time() + max(demo_args.live_swap_period, 1)
        next_is_eth_to_usdc = True
        while True:
            if dashboard_proc.poll() is not None:
                print(f"[demo] dashboard exited with code {dashboard_proc.returncode}", flush=True)
                return 1
            if demo_args.live_swap_period > 0 and time.time() >= next_live_swap_ts:
                direction = "eth_to_usdc" if next_is_eth_to_usdc else "usdc_to_eth"
                with live_swap_log.open("a") as f:
                    f.write(f"\n--- live swap ({direction}) ---\n")
                    try:
                        subprocess.run(
                            [
                                python_bin,
                                "simulate_swaps.py",
                                "--rpc-url",
                                e2e_args.rpc_url,
                                "--private-key",
                                e2e_args.swap_pk,
                                "--cycles",
                                "1",
                                "--interval",
                                "0",
                                "--eth-in",
                                str(demo_args.live_swap_eth_in),
                                "--usdc-in",
                                str(demo_args.live_swap_usdc_in),
                                "--start-direction",
                                direction,
                            ],
                            cwd=runner.agent_dir,
                            check=True,
                            stdout=f,
                            stderr=subprocess.STDOUT,
                        )
                    except Exception as e:
                        print(f"[demo] live swap failed ({direction}): {e}", flush=True)
                next_is_eth_to_usdc = not next_is_eth_to_usdc
                next_live_swap_ts = time.time() + demo_args.live_swap_period
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[demo] interrupted", flush=True)
        return 130
    finally:
        if dashboard_proc and dashboard_proc.poll() is None:
            dashboard_proc.terminate()
            try:
                dashboard_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dashboard_proc.kill()
        runner._stop_all()


if __name__ == "__main__":
    raise SystemExit(main())
