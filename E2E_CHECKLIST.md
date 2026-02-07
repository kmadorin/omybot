# Step 6 E2E Checklist

Run:

```bash
cd omybot
python run_e2e.py
```

The runner automates Step 6 from `research/CLAUDE_CODE_GUIDE.md`:

1. Starts Anvil fork (`chainId=8453`, `block-time=2s`).
2. Funds account #0 (agent) and account #1 (swap simulator) with USDC by impersonating the whale address.
3. Starts `agent/agent.py`.
4. Runs `agent/simulate_swaps.py` with alternating ETH->USDC and USDC->ETH swaps.
5. Verifies logs for:
   - initial position mint,
   - recurring pool state reads,
   - swap activity,
   - decisions log output,
   - rebalance decision,
   - rebalance completion.

Useful flags:

- `--fork-url`, `--rpc-url`, `--swap-cycles`, `--swap-interval`
- `--swap-eth-in`, `--swap-usdc-in`
- `--agent-usdc-raw`, `--swap-usdc-raw`
- `--keep-running`
