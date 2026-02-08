#!/bin/bash
set -e
echo "=== OmyBot Startup Validation ==="

# Health endpoints
echo -n "Demo health:    "; curl -sf https://omybotdemo.simula.online/health || echo "FAIL"
echo -n "Mainnet health: "; curl -sf https://omybot.simula.online/health || echo "FAIL"

# Chain ID from agent logs
echo -n "Demo chain ID:    "; docker compose -f deploy/docker-compose.yml logs demo-agent 2>&1 | grep -o "Chain ID: [0-9]*" | head -1 || echo "NOT FOUND"
echo -n "Mainnet chain ID: "; docker compose -f deploy/docker-compose.yml logs mainnet-agent 2>&1 | grep -o "Chain ID: [0-9]*" | head -1 || echo "NOT FOUND"

# PPO model loaded
echo -n "Demo PPO model:    "; docker compose -f deploy/docker-compose.yml logs demo-agent 2>&1 | grep -c "Loaded PPO model" || echo "NOT LOADED"
echo -n "Mainnet PPO model: "; docker compose -f deploy/docker-compose.yml logs mainnet-agent 2>&1 | grep -c "Loaded PPO model" || echo "NOT LOADED"

echo "=== Done ==="
