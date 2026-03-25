---
name: ip
description: >
  Query IP address information, including local addresses, geolocation, and network scanning.
  Core Scenario: When the user needs to identify their IP, find the location of an IP, or scan for active hosts in a subnet.
license: MIT
---

# ip - IP Address & Geolocation Utility

The `ip` module provides a comprehensive set of tools for identifying IP addresses and their geographical locations, as well as performing basic network reconnaissance like subnet mapping and port scanning.

## When to Activate
- When the user wants to know their local or public IP address.
- When retrieving geolocation data (country, city, ISP) for a specific IP.
- When identifying active hosts within a CIDR range.
- When performing fast TCP port scans on a target.

## Core Principles & Rules
- **External Data**: Uses Geolite (ipinfo.io) for geolocation subcommands.
- **Network Recon**: Supports `map` for discovering active IPs and `tps` for port scanning.
- **Platform Agnostic**: Provides a unified interface for `ifconfig` and `ip addr` across different OS environments.

## Patterns & Examples

### Local IP Info
```bash
# List all local network interface IP addresses
x ip ls
```

### Geolocation
```bash
# Get the geographical location of a public IP (e.g., 8.8.8.8)
x ip geolite 8.8.8.8
```

### Network Mapping
```bash
# Discover active hosts in a specific subnet
x ip map 192.168.1.0/24
```

## Checklist
- [ ] Confirm if the user needs local info or geolocation for a remote IP.
- [ ] Verify the CIDR range if performing a network map.
- [ ] Ensure the user is aware that some subcommands rely on external web services.
