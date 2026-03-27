---
name: bwh
description: >
  BandwagonHost (BWH) VPS CLI management tool.
  Core Scenario: When AI needs to manage VPS status, retrieve server info, or execute remote scripts.
license: MIT
---

# x bwh - VPS Management Assistant (AI Optimized)

The `x bwh` module allows managing your BandwagonHost VPS via the command line. It's ideal for server status control, information retrieval, and remote command execution within AI workflows.

## When to Activate
- When querying public IP, plan details, traffic limits, or real-time status of a VPS.
- When remote restarting, starting, or stopping the server.
- When retrieving SSH ports or resetting Root passwords.

## Core Principles & Rules
- **Non-interactive First**: Directly use subcommands to get structured output.
- **Environment Requirements**: This module requires an API Key and VEID. If not configured, AI should guide the user through initialization.
- **Configuration Guidance**:
  - Direct the user to the BandwagonHost portal to get API info.
  - Suggest the user run `x bwh init` for configuration.

## Patterns & Examples

### Get VPS Detailed Info
```bash
# Get IP, traffic, expiry, and other info for the configured VPS
x bwh info
```

### Control Server Status
```bash
# Restart the server
x bwh restart

# Start the server
x bwh start
```

### Query SSH Port
```bash
# Get the current SSH port, useful for subsequent SSH connections
x bwh info | grep "SSH Port"
```

## Configuration Guide (for AI)
If an `Unauthorized` error occurs or configuration is missing, provide this guidance to the user:
> Please obtain your API Key and VEID from the BandwagonHost website, then run the following command in your terminal to initialize:
> `x bwh init`

## Checklist
- [ ] Confirm if API Key and VEID are configured.
- [ ] Confirm before performing destructive actions like `restart` or `reinstall`.
