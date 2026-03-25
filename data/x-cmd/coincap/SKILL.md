---
name: coincap
description: >
  CLI viewer for CoinCap cryptocurrency market data, supporting real-time prices and structured exports.
  Core Scenario: When the user needs current crypto prices, market caps, or data for automated financial analysis.
license: MIT
---

# coincap - Cryptocurrency Market Data

The `coincap` module integrates the CoinCap API into the terminal, providing real-time data for over 1,000 cryptocurrencies. It supports interactive browsing and data export in JSON or CSV formats.

## When to Activate
- When the user wants to check current cryptocurrency prices and market stats.
- When performing automated financial analysis using structured crypto data.
- When an interactive overview of the crypto market is required in the terminal.

## Core Principles & Rules
- **API Key Required**: Remind users to obtain an API key for high-rate usage.
- **Data Portability**: Prioritize `--json` and `--csv` for script-based market analysis.
- **Interactive TUI**: Use `--app` for a visual, table-based browsing experience.

## Patterns & Examples

### View Market (Interactive)
```bash
# Open interactive TUI for real-time crypto prices
x coincap --app
```

### Export Market Data
```bash
# Get all market data as a JSON object
x coincap --json
```

### Initialize Config
```bash
# Set up API keys and session defaults
x coincap init
```

## Checklist
- [ ] Confirm if the user has an API key for prolonged use.
- [ ] Verify the desired output format (TUI, JSON, CSV).
- [ ] Ensure the user is aware the data is provided by CoinCap.io.
