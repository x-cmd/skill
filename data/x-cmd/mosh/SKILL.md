---
name: mosh
description: >
  Enhanced interface for Mosh (Mobile Shell), a robust SSH replacement for unstable networks.
  Core Scenario: When the user needs a persistent, roaming remote terminal connection over UDP.
license: MIT
---

# mosh - Robust Remote Shell (UDP)

The `mosh` module provides an enhanced wrapper for the Mobile Shell, ensuring availability by automatically downloading it via pixi if it is missing from the system. It is ideal for roaming and high-latency connections.

## When to Activate
- When connecting to remote servers over unstable or high-latency networks (e.g., cellular data).
- When the user needs a connection that persists through network changes (roaming).
- When seeking an interactive SSH alternative with local echo and line editing features.

## Core Principles & Rules
- **UDP Requirement**: Uses UDP ports (default 60000-61000) instead of TCP.
- **SSH dependency**: Requires a functioning SSH command to establish the initial handshake.
- **Cross-Platform**: Automatically handles setup via `pixi` on supported platforms (macOS/Linux).

## Patterns & Examples

### Basic Connection
```bash
# Connect to a remote server using Mosh
x mosh user@remote-host.com
```

### Specific Port and SSH Command
```bash
# Use a custom SSH port and specific Mosh UDP port
x mosh -p 80 --ssh="ssh -p 2222" root@server.com
```

## Checklist
- [ ] Ensure the remote server also has `mosh-server` installed.
- [ ] Verify that UDP ports (60000-61000) are open on the remote firewall.
- [ ] Confirm the target user and hostname.
