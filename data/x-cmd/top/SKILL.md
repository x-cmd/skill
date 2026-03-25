---
name: top
description: >
  Enhanced process viewer with an interactive UI, using bottom (btm) when available.
  Core Scenario: When the user needs to monitor system processes and resource usage in real-time with an interactive interface.
license: MIT
---

# top - Interactive Process Viewer

The `top` module provides an enhanced, interactive interface for monitoring system processes. It prioritizes using `bottom` (btm) for a superior visual experience and falls back to the standard `top` command in non-TTY environments.

## When to Activate
- When the user wants to monitor CPU, memory, and process activity in real-time.
- When seeking a more modern, interactive alternative to the standard `top` command.
- When performing quick system resource diagnostics via terminal.

## Core Principles & Rules
- **Interactive UI**: Automatically downloads and runs `bottom` if available in an interactive terminal.
- **Auto-Fallback**: Reverts to the system's native `top` when the output is piped or not in a TTY.
- **Subcommand Access**: Use `btm` to explicitly launch the enhanced viewer or `--` for the native command.

## Patterns & Examples

### Enhanced Monitoring
```bash
# Launch the enhanced interactive process viewer (btm)
x top
```

### Native Top
```bash
# Force execution of the system's native top command
x top --
```

### Basic Mode
```bash
# Start bottom in a simplified interface mode
x top btm --basic
```

## Checklist
- [ ] Confirm if an interactive UI is needed.
- [ ] Verify if the output needs to be piped (which triggers fallback).
