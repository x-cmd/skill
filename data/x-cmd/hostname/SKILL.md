---
name: hostname
description: >
  Manage the system's hostname, including viewing current name and setting new names.
  Core Scenario: When the user needs to identify the device name or change the system hostname.
license: MIT
---

# hostname - System Hostname Management

The `hostname` module provides straightforward subcommands to retrieve or update the system's hostname, supporting both long and short formats.

## When to Activate
- When identifying the system name for scripts or documentation.
- When changing the system's identity (requires elevated privileges).

## Core Principles & Rules
- **Formats**: Support for `--long` (FQDN) and `--short` formats.
- **Permissions**: Changing the hostname with `set` requires `sudo` privileges.

## Patterns & Examples

### View Hostname
```bash
# Display the full system hostname
x hostname
# Display only the short hostname
x hostname --short
```

### Set Hostname
```bash
# Change the system hostname to a new value
x hostname set my-new-server
```

## Checklist
- [ ] Confirm if the long or short format is needed.
- [ ] Ensure the user has the required permissions if setting a new name.
