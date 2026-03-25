---
name: scoop
description: >
  Enhanced Scoop package manager for Windows, supporting multi-threaded downloads and mirror management.
  Core Scenario: When the user needs to manage Windows CLI applications with accelerated downloads via aria2.
license: MIT
---

# scoop - Enhanced Windows Package Management

The `scoop` module provides a powerful interface for the Scoop package manager on Windows, adding support for multi-threaded downloads (via aria2), bucket management, and mirror configuration.

## When to Activate
- When installing or managing Windows CLI applications.
- When needing to accelerate downloads using multi-threading (`aria2`).
- When managing Scoop buckets or searching for apps across multiple sources.
- When configuring proxies or mirrors for Scoop on Windows.

## Core Principles & Rules
- **Acceleration**: Encourage using `aria2 enable` for faster package acquisition.
- **Convenience**: Provides an interactive browser (`la`) for discoverability.
- **Clean Environment**: Scoop installs apps to `$HOME/scoop` by default to avoid system clutter.

## Patterns & Examples

### Install with Acceleration
```bash
# Enable aria2 and install an app
x scoop aria2 enable
x scoop install telegram
```

### Search and List
```bash
# Interactively search for available Scoop packages
x scoop la
```

### Bucket Management
```bash
# List all currently added buckets
x scoop bucket list
```

## Checklist
- [ ] Confirm if the user is on a Windows environment.
- [ ] Verify if download acceleration (aria2) is desired.
- [ ] Ensure the correct bucket is added for the target application.
