---
name: ip
description: >
  Network diagnostics and IP geolocation query.
  Core Scenario: When AI needs to query local network configuration or geographic info of remote server IPs.
license: MIT
---

# x ip - Network Diagnostics & Geolocation (AI Optimized)

The `x ip` module provides a concise interface for querying IP-related information, covering both local NIC configurations and global IP geographic data.

## When to Activate
- When retrieving the local public IP or internal network interface addresses.
- When querying geographic location, ISP, or other info for a specific IP.
- When scanning active hosts or open ports in a subnet (network diagnostics).

## Core Principles & Rules
- **Non-interactive First**: Avoid interactive configurations; retrieve data directly via subcommands.
- **Structured Thinking**: Use subcommands like `geolite` for detailed geographic insights.

## Patterns & Examples

### Query Local Public IP and Location
```bash
# Get detailed geographic info for the current public IP
x ip geolite
```

### Query Geolocation for a Specific IP
```bash
# Query detailed location info for 8.8.8.8
x ip geolite 8.8.8.8
```

### List All Local Interface Addresses
```bash
# Get a list of all IP addresses on the machine (non-interactive)
x ip ls
```

### Network Scanning (Diagnostics)
```bash
# Scan active hosts in a subnet (may require privileges)
x ip map 192.168.1.0/24

# Quickly scan common ports for a specific IP
x ip tps 127.0.0.1
```

## Checklist
- [ ] Confirm whether querying a specific IP or the local egress IP.
- [ ] Check privilege requirements before running scan tasks.
