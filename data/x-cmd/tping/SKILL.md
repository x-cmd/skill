---
name: tping
description: >
  Ping TCP ports using curl to test service availability and network connectivity.
  Core Scenario: When the user needs to check if a specific TCP port (e.g., 80, 443) is open and responsive.
license: MIT
---

# tping - TCP Port Connectivity Test

The `tping` module uses `curl` to perform TCP connection tests to specific ports on a target host. It is ideal for verifying service availability when ICMP (standard ping) is blocked.

## When to Activate
- When testing if a web server (port 80/443) or custom service is reachable.
- When standard ICMP ping is disabled by a firewall on the target.
- When needing visual metrics (heatmap/bar) for TCP connection times.

## Core Principles & Rules
- **Port Specificity**: Allows testing connectivity to any TCP port (default is 80).
- **Visualization**: Supports the same heatmap and bar chart modes as the enhanced `ping` module.

## Patterns & Examples

### Basic Port Check
```bash
# Test TCP connectivity to port 80 of a domain
x tping bing.com
```

### Custom Port with Visualization
```bash
# Check port 443 with a real-time bar chart
x tping --bar bing.com:443
```

## Checklist
- [ ] Confirm the target host and port number.
- [ ] Verify if standard ping is insufficient for the task.
