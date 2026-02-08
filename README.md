# OmyBot

**AI-powered autonomous liquidity manager for Uniswap V4 on Base.**

OmyBot continuously monitors ETH/USDC pool state on Uniswap V4 and autonomously manages concentrated liquidity positions. It uses a PPO reinforcement learning model (with a rule-based fallback) to decide when and how to rebalance positions, collect fees, or hold — then executes those decisions atomically on-chain.

Built for [HackMoney 2026](https://ethglobal.com/events/hackmoney2026) by EthGlobal.

---

## How It Works

```
                    +-------------------+
                    |   Uniswap V4      |
                    |   StateView       |
                    | (pool tick, price |
                    |  liquidity, fees) |
                    +--------+----------+
                             |
                     RPC reads every 15s
                             |
                    +--------v----------+
                    |     LP Agent      |
                    |                   |
                    | 1. Read pool state|
                    | 2. Build 10-dim   |
                    |    observation    |
                    | 3. PPO predict    |
                    |    (or rule-based)|
                    | 4. Execute action |
                    +--------+----------+
                             |
                  modifyLiquidities()
                             |
                    +--------v----------+
                    | V4 PositionManager|
                    | (mint, burn,      |
                    |  rebalance)       |
                    +-------------------+
```

**Decision types:**

| Decision | Trigger | Action |
|----------|---------|--------|
| **REBALANCE** | Price drifts >75% toward position edge (rule-based) or PPO threshold exceeded | Withdraw liquidity, mint new position centered on current tick |
| **COLLECT_FEES** | Position in-range, low drift, cooldown elapsed | DECREASE with liquidity=0 to harvest accrued fees |
| **HOLD** | Position healthy, no action needed | No-op |

**PPO observation vector (10 features):**
`current_tick`, `pool_liquidity_log`, `position_lower_tick`, `position_upper_tick`, `fees_accrued_norm`, `impermanent_loss_pct`, `volatility_5m`, `volatility_1h`, `time_since_rebalance_norm`, `position_in_range_flag`

**PPO action space (4 continuous dims):**
`lower_tick_delta`, `upper_tick_delta`, `liquidity_fraction`, `rebalance_threshold`

---

## Project Structure

```
omybot/
├── agent/                    # Core autonomous LP agent
│   ├── agent.py              # Main decision loop
│   ├── config.py             # Addresses, pool params, thresholds
│   ├── lp_manager.py         # V4 PositionManager command encoding
│   ├── state_reader.py       # StateView RPC queries
│   ├── simulate_swaps.py     # Swap simulator for price movement
│   ├── abi/                  # Contract ABIs
│   ├── positions/            # Persisted LP position state
│   ├── decisions/            # Decision logs (human + JSONL)
│   └── requirements.txt
├── ml/                       # ML pipeline
│   ├── env/lp_env.py         # Gymnasium RL environment
│   ├── rl/train_ppo.py       # PPO training (Stable-Baselines3)
│   ├── rl/evaluate.py        # Policy evaluation & plotting
│   ├── data/                 # Data collection & processing
│   ├── models/               # Trained model artifacts
│   └── requirements.txt
├── dashboard/                # Real-time monitoring web UI
│   ├── server.py             # Flask backend + background collector
│   └── templates/index.html  # Single-page dashboard
├── contracts/                # Foundry project (hook contract)
│   ├── src/                  # Solidity sources
│   ├── test/                 # Forge tests
│   └── foundry.toml
├── deploy/                   # Docker deployment
│   ├── docker-compose.yml    # Demo + mainnet profiles
│   ├── Dockerfile            # Multi-stage build
│   ├── .env.demo             # Demo config (safe Anvil test keys)
│   └── .env.mainnet.example  # Mainnet config template
├── run_demo.py               # One-command demo launcher
├── run_e2e.py                # E2E test runner
└── package.json
```

---

## Prerequisites

- **Python 3.10+**
- **Foundry** (`anvil`, `cast`, `forge`) — [install](https://getfoundry.sh)
- **Node.js 18+** (optional, for `npm run` scripts)
- **Docker + Docker Compose** (optional, for containerized deployment)

Install Foundry:

```bash
curl -L https://foundry.paradigm.xyz | bash
foundryup
```

---

## Quick Start: Run the Demo

The demo runs everything locally using an Anvil fork of Base mainnet. No real funds, no private keys, no testnet tokens needed.

### Option 1: Python script (recommended)

```bash
cd omybot

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r agent/requirements.txt

# Launch the full demo stack
python run_demo.py
```

This single command:

1. Starts an **Anvil fork** of Base mainnet (chain ID 8453, 2s block time)
2. **Funds** the agent and swap simulator accounts with USDC (impersonating a whale)
3. **Starts the LP agent** — mints an initial ETH/USDC position and begins the decision loop
4. **Launches the dashboard** at [http://localhost:5005](http://localhost:5005)
5. **Runs swap simulations** to generate price movement and trigger rebalances
6. **Injects live swaps** every 30s to keep the market moving

Press `Ctrl+C` to stop all processes.

#### Demo flags

```bash
# Customize swap parameters
python run_demo.py --live-swap-period 15 --live-swap-eth-in 5.0 --live-swap-usdc-in 10000.0

# Change dashboard port
python run_demo.py --dashboard-port 8080

# Pass-through E2E flags
python run_demo.py --swap-cycles 10 --swap-eth-in 30.0 --swap-usdc-in 50000.0
```

### Option 2: Docker Compose

```bash
cd omybot/deploy

# Copy and review the demo config
cp .env.demo .env

# Start the demo stack
docker compose --profile demo up --build
```

Services:

| Service | Description | Port |
|---------|-------------|------|
| `anvil` | Base mainnet fork | 8545 (internal) |
| `demo-agent` | LP agent (funds accounts on startup) | — |
| `demo-dashboard` | Monitoring dashboard | localhost:5006 |
| `swap-simulator` | Continuous swap generator | — |

### Option 3: E2E test only (no dashboard)

```bash
cd omybot
python run_e2e.py
```

Runs the agent + swap simulation, verifies all checks pass, then exits. Use `--keep-running` to leave Anvil and the agent alive after verification.

---

## Running Each Component Individually

### 1. Start an Anvil fork

```bash
anvil --fork-url https://mainnet.base.org --chain-id 8453 --block-time 2 --auto-impersonate
```

### 2. Fund accounts

Use `cast` to transfer USDC from a whale to your agent address:

```bash
# Fund agent with 10,000 USDC
cast send 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913 \
  'transfer(address,uint256)' \
  0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266 \
  10000000000 \
  --from 0xBaeD383EDE0e5d9d72430661f3285DAa77E9439F \
  --unlocked --rpc-url http://localhost:8545
```

### 3. Start the agent

```bash
cd omybot/agent
AGENT_PK=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80 python agent.py
```

Or pass the key as a CLI argument:

```bash
python agent.py 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
```

### 4. Simulate swaps (generate price movement)

```bash
cd omybot/agent
python simulate_swaps.py \
  --rpc-url http://localhost:8545 \
  --private-key 0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d \
  --cycles 6 --interval 20 --eth-in 20.0 --usdc-in 40000.0
```

### 5. Launch the dashboard

```bash
cd omybot/dashboard
python server.py
```

Open [http://localhost:5005](http://localhost:5005) in your browser.

---

## ML Pipeline

### Training data collection

Collect Swap events from the Base PoolManager:

```bash
cd omybot/ml/data
python collect_raw_data.py
```

Process into OHLCV bars with volatility features:

```bash
python process_raw_data.py
```

### Train the PPO model

```bash
cd omybot/ml/rl
pip install -r ../requirements.txt
python train_ppo.py
```

Training uses Stable-Baselines3 PPO with a custom Gymnasium environment (`ml/env/lp_env.py`). The trained model is saved to `ml/models/lp_ppo_model.zip`.

Monitor training progress:

```bash
tensorboard --logdir ml/logs/
```

### Evaluate

```bash
python evaluate.py
```

Compares PPO performance against baseline strategies (buy-and-hold, static LP) and generates PnL, fee, IL, and gas cost plots.

### Using the model

The agent automatically loads `ml/models/lp_ppo_model.zip` on startup if present. If the model file is missing, the agent falls back to rule-based decisions.

---

## Configuration

All configuration is in `agent/config.py`, with environment variable overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_FORK` | `true` | Use local Anvil fork vs. live Base |
| `LOCAL_RPC_URL` | `http://localhost:8545` | Anvil RPC endpoint |
| `BASE_RPC_URL` | `https://mainnet.base.org` | Base mainnet RPC |
| `EXPECTED_CHAIN_ID` | `8453` | Chain ID safety check |
| `AGENT_PK` | — | Agent wallet private key |

### Key contract addresses (Base mainnet)

| Contract | Address |
|----------|---------|
| PoolManager | `0x498581fF718922c3f8e6A244956aF099B2652b2b` |
| PositionManager | `0x7C5f5A4bBd8fD63184577525326123B519429bDc` |
| StateView | `0xA3c0c9b65baD0b08107Aa264b0f3dB444b867A71` |
| Permit2 | `0x000000000022D473030F116dDEE9F6B43aC78BA3` |
| USDC | `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913` |

### Agent parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TRADE_INTERVAL` | 15s | Seconds between decision cycles |
| `REBALANCE_DRIFT_THRESHOLD` | 0.75 | Rebalance when price drifts >75% toward range edge |
| `FEE_COLLECT_THRESHOLD` | 0.001 | Collect fees when >0.1% of position value |
| `POOL_FEE` | 500 | 0.05% pool fee (pips) |
| `TICK_SPACING` | 10 | Tick granularity |

---

## Dashboard

The dashboard provides real-time monitoring of the agent's activity:

- **Pool state**: Current tick, price (USDC/ETH), total liquidity
- **Wallet balances**: ETH and USDC held by the agent
- **Active positions**: Range, liquidity, in-range status, IL estimate, entry price
- **Decision log**: Timestamped history of HOLD/REBALANCE/COLLECT_FEES decisions with drift ratios
- **Recent swaps**: On-chain Swap events for the monitored pool
- **Price chart**: Historical price movement

The dashboard auto-refreshes every 2 seconds via a background collector thread.

---

## Uniswap V4 Integration Details

OmyBot interacts with V4 through the command-based PositionManager:

- **Mint**: `MINT_POSITION` + `SETTLE_PAIR` + `TAKE_PAIR` — opens a new concentrated liquidity position
- **Collect fees**: `DECREASE_LIQUIDITY` (liquidity=0) + `TAKE_PAIR` — harvests accrued fees without removing liquidity
- **Rebalance**: `DECREASE_LIQUIDITY` (full withdrawal) then `MINT_POSITION` + `CLOSE_CURRENCY` + `SWEEP` — atomic position migration to new tick range

All operations are encoded as packed action bytes + ABI-encoded params and executed via `modifyLiquidities(unlockData, deadline)`.

The agent persists position token IDs to `agent/positions/positions.json` because V4's PositionManager is not ERC721Enumerable — positions cannot be discovered on-chain without indexing.

---

## Logs

| File | Contents |
|------|----------|
| `agent/decisions/decisions.log` | Human-readable decision journal |
| `agent/decisions/decisions.jsonl` | Machine-readable decision records |
| `agent/positions/positions.json` | Active LP position state |
| `logs/e2e_agent_stdout.log` | Agent stdout from E2E/demo runs |
| `logs/e2e_anvil.log` | Anvil fork output |
| `logs/e2e_swap_simulation.log` | Swap simulator output |
| `logs/e2e_swap_live.log` | Live swap injections (demo mode) |
| `logs/e2e_dashboard.log` | Dashboard server output |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Chain | Base (Ethereum L2) |
| DEX | Uniswap V4 (PoolManager, PositionManager, StateView) |
| Agent | Python, web3.py, eth-abi |
| RL | PPO via Stable-Baselines3, Gymnasium |
| Dashboard | Flask, Chart.js, Tailwind CSS |
| Local dev | Anvil (Foundry) fork of Base mainnet |
| Contracts | Solidity, Foundry |
| Deployment | Docker Compose, Caddy reverse proxy |

---

## License

Built for HackMoney 2026.
