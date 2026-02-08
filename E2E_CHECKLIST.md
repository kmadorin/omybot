# Step 6 E2E Checklist

Run:

```bash
cd omybot
python run_e2e.py
```

Full demo stack (fork + funding + agent + dashboard):

```bash
cd omybot
python run_demo.py
```

The runner automates Step 6 from `research/CLAUDE_CODE_GUIDE.md`:

1. Starts Anvil fork (`chainId=8453`, `block-time=2s`).
2. Uses `--auto-impersonate` and funds account #0 (agent) and account #1 (swap simulator) with USDC from whale.
3. Starts `agent/agent.py`.
4. Runs `agent/simulate_swaps.py` with alternating ETH->USDC and USDC->ETH swaps.
5. Verifies logs for:
   - initial position mint,
   - recurring pool state reads,
   - swap activity,
   - decisions log output,
   - rebalance decision,
   - rebalance completion.

It also backs up and resets local runtime state before each run:
- `agent/positions/positions.json`
- `agent/decisions/decisions.log`
- `agent/decisions/decisions.jsonl`

Useful flags:

- `--fork-url`, `--rpc-url`, `--swap-cycles`, `--swap-interval`
- `--swap-eth-in`, `--swap-usdc-in`
- `--agent-usdc-raw`, `--swap-usdc-raw`
- `--keep-running`
