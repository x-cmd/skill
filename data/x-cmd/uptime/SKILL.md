---
name: uptime
description: >
  Display system uptime and load averages in structured YAML format.
  Core Scenario: When the user needs to check system busy state (load) or total runtime in a script-friendly format.
license: MIT
---

# uptime - Structured System Uptime & Load

The `uptime` module provides a structured YAML output of the system's runtime and load averages (1, 5, and 15 minutes), supporting various platforms including Windows via the `cosmo` backend.

## When to Activate
- When the user wants to check how long the system has been running.
- When monitoring system busy states (CPU/IO load trends).
- When a script requires structured (YAML) uptime data for monitoring or logging.

## Core Principles & Rules
- **Multi-Platform**: Automatically uses appropriate backends (native, busybox, or cosmo) based on the OS.
- **Structure**: Default output is YAML for easy parsing by humans and machines.

## Patterns & Examples

### Structured Uptime
```bash
# Display system runtime and load averages in YAML
x uptime
```

### Raw Output
```bash
# Get the original, unstructured uptime command output
x uptime --raw
```

## Checklist
- [ ] Verify if the user prefers structured YAML or raw output.
- [ ] Check if the system busy state (loadavg) is the primary interest.
