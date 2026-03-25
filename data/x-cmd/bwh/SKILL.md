---
name: bwh
description: >
  CLI manager for BandwagonHost VPS servers, providing lifecycle management and server diagnostics.
  Core Scenario: When the user needs to start, stop, restart, or backup their BandwagonHost VPS.
license: MIT
---

# bwh - BandwagonHost VPS Management

The `bwh` module allows users to manage their BandwagonHost VPS servers directly from the command line. It supports common operations like viewing info, power management, and advanced features like snapshots and reinstallation.

## When to Activate
- When the user wants to check the status or details of their VPS.
- When performing power operations (start, stop, restart) on a VPS.
- When managing SSH keys or executing remote shell commands.
- When performing advanced maintenance like creating backups, snapshots, or OS reinstallation.

## Core Principles & Rules
- **Configuration Management**: Remind users to configure their VPS API details via `cfg` or `current`.
- **Caution**: Subcommands like `kill`, `reinstall`, and `resetrootpassword` should be handled with extra care.

## Patterns & Examples

### VPS Status
```bash
# View detailed information about the current VPS
x bwh info
```

### Power Management
```bash
# Restart the active VPS instance
x bwh restart
```

### Remote Shell
```bash
# Execute a command on the VPS via the manager
x bwh sh "uptime"
```

## Checklist
- [ ] Verify the target VPS configuration is active.
- [ ] Confirm if the user intends to perform a destructive action (reinstall/kill).
- [ ] Ensure API credentials are correctly set up.
