"""
OmyBot Dashboard — Flask backend with background collector.

Reads pool state, wallet balances, positions, decisions, and swaps from the
anvil fork and agent files. Serves a single-page dashboard at localhost:5005.
"""

import json
import logging
import math
import os
import re
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from pathlib import Path

from flask import Flask, jsonify, render_template
from web3 import Web3
from web3._utils.events import get_event_data

getcontext().prec = 40
logger = logging.getLogger("omybot.dashboard")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
sys.path.insert(0, str(AGENT_DIR))
import config  # loads agent/.env automatically
from state_reader import StateReader

AGENT_ADDRESS = os.environ.get(
    "AGENT_ADDR", "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
)

POSITIONS_FILE = AGENT_DIR / "positions" / "positions.json"
DECISIONS_JSONL = AGENT_DIR / "decisions" / "decisions.jsonl"
DECISIONS_LOG = AGENT_DIR / "decisions" / "decisions.log"
PRICE_HISTORY_FILE = Path(__file__).resolve().parent / "price_history.json"

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global snapshot — written by collector, read by API
# ---------------------------------------------------------------------------
snapshot = {
    "status": "initializing",
    "updated_at": None,
    "age_seconds": 999,
    "errors": [],
    "pool": {},
    "wallet": {},
    "positions": [],
    "decisions": [],
    "recent_swaps": [],
    "price_history": [],
}
snapshot_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Background Collector
# ---------------------------------------------------------------------------
class Collector:
    def __init__(self):
        rpc_url = config.LOCAL_RPC_URL if config.USE_FORK else config.BASE_RPC_URL
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.state_reader = StateReader(self.w3, config)

        # Load ABIs
        ABI_DIR = AGENT_DIR / "abi"
        with open(ABI_DIR / "erc20.json") as f:
            erc20_abi = json.load(f)
        with open(ABI_DIR / "position_manager.json") as f:
            pm_abi = json.load(f)
        with open(ABI_DIR / "pool_manager.json") as f:
            pool_mgr_abi = json.load(f)

        self.usdc = self.w3.eth.contract(
            address=Web3.to_checksum_address(config.USDC_ADDRESS), abi=erc20_abi
        )
        self.position_manager = self.w3.eth.contract(
            address=Web3.to_checksum_address(config.POSITION_MANAGER), abi=pm_abi
        )
        self.pool_manager = self.w3.eth.contract(
            address=Web3.to_checksum_address(config.POOL_MANAGER), abi=pool_mgr_abi
        )
        self.swap_event_abi = next(
            item
            for item in pool_mgr_abi
            if item.get("type") == "event" and item.get("name") == "Swap"
        )
        self.swap_topic0 = self.w3.keccak(
            text="Swap(bytes32,address,int128,int128,uint160,uint128,int24,uint24)"
        ).hex()

        # Swap cursor
        self.last_swap_block = 0
        self.swap_cache = []
        self.swap_seen = set()

        # Price history
        self.price_history = self._load_price_history()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run_forever(self):
        while True:
            try:
                self._collect()
            except Exception:
                traceback.print_exc()
            time.sleep(2)

    def _collect(self):
        global snapshot
        errors = []

        pool = self._safe_call(self._get_pool, errors)
        wallet = self._safe_call(self._get_wallet, errors)
        positions = self._safe_call(lambda: self._get_positions(pool), errors)
        decisions = self._safe_call(self._get_decisions, errors)
        # Swap feed should not degrade core health if log RPC is temporarily flaky.
        self._safe_call(self._get_swaps, [])

        if pool:
            self._update_price_history(pool["price"])

        any_fresh = pool is not None
        status = "ok" if not errors else ("degraded" if pool else "error")

        with snapshot_lock:
            update = {
                "status": status,
                "errors": errors,
                "pool": pool or snapshot.get("pool", {}),
                "wallet": wallet or snapshot.get("wallet", {}),
                "positions": positions
                if positions is not None
                else snapshot.get("positions", []),
                "decisions": decisions
                if decisions is not None
                else snapshot.get("decisions", []),
                "recent_swaps": self.swap_cache[-30:],
                "price_history": self.price_history[-200:],
            }
            if any_fresh:
                update["updated_at"] = datetime.now(timezone.utc).isoformat()
                update["age_seconds"] = 0
            snapshot.update(update)

    def _safe_call(self, fn, errors):
        try:
            return fn()
        except Exception as e:
            errors.append(str(e))
            return None

    # ------------------------------------------------------------------
    # Data sources
    # ------------------------------------------------------------------
    def _get_pool(self):
        ps = self.state_reader.get_pool_state()
        return {
            "tick": ps["tick"],
            "price": round(ps["price"], 2),
            "liquidity": str(ps["liquidity"]),
            "lpFee": ps["lpFee"],
        }

    def _get_wallet(self):
        addr = Web3.to_checksum_address(AGENT_ADDRESS)
        eth_raw = self.w3.eth.get_balance(addr)
        usdc_raw = self.usdc.functions.balanceOf(addr).call()
        return {
            "address": AGENT_ADDRESS,
            "eth_balance": round(eth_raw / 10**18, 6),
            "usdc_balance": round(usdc_raw / 10**6, 2),
        }

    def _get_positions(self, pool):
        if pool is None:
            return None
        if not POSITIONS_FILE.exists():
            return []
        positions = json.loads(POSITIONS_FILE.read_text())
        current_tick = pool["tick"]
        current_price = pool["price"]
        enriched = []
        for p in positions:
            token_id = p["token_id"]
            tick_lower = int(p["tick_lower"])
            tick_upper = int(p["tick_upper"])
            try:
                liq = self.position_manager.functions.getPositionLiquidity(
                    token_id
                ).call()
            except Exception:
                liq = 0
            in_range = tick_lower <= current_tick <= tick_upper
            if in_range:
                distance_ticks = 0
                distance_side = "in_range"
            elif current_tick < tick_lower:
                distance_ticks = tick_lower - current_tick
                distance_side = "below_range"
            else:
                distance_ticks = current_tick - tick_upper
                distance_side = "above_range"
            entry_price = p.get("entry_price", current_price)

            price_change_pct = (
                ((current_price - entry_price) / entry_price * 100)
                if entry_price > 0
                else 0
            )

            ratio = current_price / entry_price if entry_price > 0 else 1
            il_fraction = (2 * math.sqrt(ratio) / (1 + ratio)) - 1
            il_est_pct = il_fraction * 100

            enriched.append(
                {
                    "token_id": token_id,
                    "tick_lower": tick_lower,
                    "tick_upper": tick_upper,
                    "price_lower": round(
                        self.state_reader.tick_to_price(tick_lower), 2
                    ),
                    "price_upper": round(
                        self.state_reader.tick_to_price(tick_upper), 2
                    ),
                    "liquidity": str(liq),
                    "in_range": in_range,
                    "position_distance_ticks": int(distance_ticks),
                    "position_distance_side": distance_side,
                    "entry_price": round(entry_price, 2),
                    "price_change_pct": round(price_change_pct, 2),
                    "il_est_pct": round(il_est_pct, 4),
                    "created_at": p.get("created_at", ""),
                }
            )
        return enriched

    def _get_decisions(self, limit=50):
        if DECISIONS_JSONL.exists():
            lines = DECISIONS_JSONL.read_text().splitlines()
            decisions = []
            for line in lines[-limit:]:
                try:
                    decisions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return decisions

        # Fallback: regex parse decisions.log
        return self._parse_decisions_log_fallback(limit)

    def _parse_decisions_log_fallback(self, limit=50):
        if not DECISIONS_LOG.exists():
            return []
        lines = DECISIONS_LOG.read_text().splitlines()
        pattern = re.compile(
            r"DECISION: (\w+) source=(\w+) \| tick=(-?\d+) price=([\d.]+) \| "
            r"range=\[(-?\d+),(-?\d+)\] \| drift=([\d.]+) in_range=(\w+) \| "
            r"token_id=(\S+)"
        )
        decisions = []
        for line in lines[-limit * 2 :]:
            m = pattern.search(line)
            if m:
                ts_match = re.match(r"([\d\-: ,]+)", line)
                decisions.append(
                    {
                        "ts": ts_match.group(1).strip() if ts_match else "",
                        "decision": m.group(1),
                        "source": m.group(2),
                        "tick": int(m.group(3)),
                        "price": float(m.group(4)),
                        "range": [int(m.group(5)), int(m.group(6))],
                        "drift": float(m.group(7)),
                        "in_range": m.group(8) == "True",
                        "token_id": None
                        if m.group(9) == "None"
                        else int(m.group(9)),
                    }
                )
        return decisions[-limit:]

    def _get_swaps(self):
        current_block = self.w3.eth.block_number
        from_block = max(self.last_swap_block + 1, current_block - 200)
        if from_block > current_block:
            return self.swap_cache

        pool_id_topic = "0x" + config.POOL_ID[2:].rjust(64, "0")
        try:
            logs = self.w3.eth.get_logs(
                {
                    "address": Web3.to_checksum_address(config.POOL_MANAGER),
                    "fromBlock": from_block,
                    "toBlock": current_block,
                    "topics": [self.swap_topic0, pool_id_topic],
                }
            )
        except Exception as e:
            # Some RPC providers intermittently reject the stricter topic query.
            logger.warning("Swap log query failed, retrying with broader filter: %s", e)
            logs = self.w3.eth.get_logs(
                {
                    "address": Web3.to_checksum_address(config.POOL_MANAGER),
                    "fromBlock": max(self.last_swap_block + 1, current_block - 50),
                    "toBlock": current_block,
                    "topics": [self.swap_topic0],
                }
            )

        for raw_log in logs:
            ev = get_event_data(self.w3.codec, self.swap_event_abi, raw_log)
            event_pool_id = ev["args"].get("id")
            if event_pool_id is not None:
                event_pool_id_hex = (
                    event_pool_id.hex()
                    if hasattr(event_pool_id, "hex")
                    else str(event_pool_id)
                )
                if not event_pool_id_hex.startswith("0x"):
                    event_pool_id_hex = "0x" + event_pool_id_hex
                if event_pool_id_hex.lower() != config.POOL_ID.lower():
                    continue
            key = (ev["transactionHash"].hex(), int(ev["logIndex"]))
            if key in self.swap_seen:
                continue
            self.swap_seen.add(key)

            args = ev["args"]
            sqrt_p = Decimal(args["sqrtPriceX96"])
            price = float((sqrt_p / Decimal(2**96)) ** 2 * Decimal(10**12))

            self.swap_cache.append(
                {
                    "tx_hash": ev["transactionHash"].hex(),
                    "log_index": ev["logIndex"],
                    "amount0": str(args["amount0"]),
                    "amount1": str(args["amount1"]),
                    "price_after": round(price, 2),
                    "tick_after": args["tick"],
                    "block_number": ev["blockNumber"],
                }
            )

        if len(self.swap_cache) > 100:
            self.swap_cache = self.swap_cache[-50:]
            self.swap_seen = {
                (s["tx_hash"], s["log_index"]) for s in self.swap_cache
            }

        self.last_swap_block = current_block
        return self.swap_cache

    # ------------------------------------------------------------------
    # Price history
    # ------------------------------------------------------------------
    def _load_price_history(self):
        if PRICE_HISTORY_FILE.exists():
            try:
                return json.loads(PRICE_HISTORY_FILE.read_text())[-200:]
            except Exception:
                return []
        return []

    def _update_price_history(self, price):
        entry = {"t": time.time(), "p": round(price, 2)}
        if not self.price_history or abs(self.price_history[-1]["p"] - price) > 0.01:
            self.price_history.append(entry)
            if len(self.price_history) > 200:
                self.price_history = self.price_history[-200:]
            tmp = PRICE_HISTORY_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.price_history))
            tmp.rename(PRICE_HISTORY_FILE)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    with snapshot_lock:
        s = dict(snapshot)
        if s["updated_at"]:
            age = (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(s["updated_at"])
            ).total_seconds()
            s["age_seconds"] = round(age, 1)
            if age > 10 and s["status"] == "ok":
                s["status"] = "stale"
    return jsonify(s)


@app.route("/health")
def health():
    with snapshot_lock:
        if snapshot["updated_at"] is not None:
            return jsonify({"ok": True, "status": snapshot["status"]}), 200
        return jsonify({"ok": False, "status": "initializing"}), 503


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    collector = Collector()
    t = threading.Thread(target=collector.run_forever, daemon=True)
    t.start()
    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.environ.get("DASHBOARD_PORT", "5005"))
    print(f"OmyBot Dashboard starting on http://localhost:{port}")
    app.run(host=host, port=port, debug=False)
