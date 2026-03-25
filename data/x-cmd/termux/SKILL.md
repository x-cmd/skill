---
name: termux
description: >
  Comprehensive utility for Termux on Android, providing package management, API integration, and PRoot distros.
  Core Scenario: When the user needs to interact with Android hardware (SMS, vibrate, say) or manage Linux distros via PRoot.
license: MIT
---

# termux - Termux Android Enhancement

The `termux` module integrates various Android-specific features into the Termux terminal, allowing for hardware interaction and easy management of isolated Linux environments.

## When to Activate
- When the user wants to interact with Android sensors or hardware (battery, vibration, volume).
- When performing SMS management (reading or sending messages) via CLI.
- When using text-to-speech (`say`) capabilities on an Android device.
- When installing or managing Linux distributions (Ubuntu, Alpine, Debian) using PRoot without root access.

## Core Principles & Rules
- **API Dependencies**: Remind users that `termux-api` must be installed for hardware subcommands.
- **Rootless environments**: Emphasize that `proot-distro` allows running full Linux environments without rooting the device.
- **Shortcuts**: Support for the `m` alias (default shortcut for `x termux`).

## Patterns & Examples

### Hardware Interaction
```bash
# Make the phone vibrate for 500ms
x termux vibrate 500
# Speak text using the Android voice engine
x termux say "System update complete"
```

### PRoot Management
```bash
# Install and run an Ubuntu environment
x termux pd install ubuntu
x termux ubu bash
```

### SMS Management
```bash
# Send an SMS message to a specific number
x termux sms send -n 123456789 "Hello from x-cmd"
```

## Checklist
- [ ] Verify if the `termux-api` package is installed for hardware tasks.
- [ ] Confirm the target Linux distribution name for PRoot commands.
- [ ] Ensure the Android device has the necessary permissions granted to Termux.
