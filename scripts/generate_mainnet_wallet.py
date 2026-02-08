#!/usr/bin/env python3
"""Generate a new EVM wallet for mainnet deployment env vars."""

from eth_account import Account


def main() -> int:
    account = Account.create()
    private_key_hex = account.key.hex()
    if not private_key_hex.startswith("0x"):
        private_key_hex = f"0x{private_key_hex}"

    print("Generated new wallet (store private key securely):")
    print(f"Address: {account.address}")
    print()
    print("Paste into deploy/.env.mainnet:")
    print(f"MAINNET_AGENT_PK={private_key_hex}")
    print(f"MAINNET_AGENT_ADDR={account.address}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
