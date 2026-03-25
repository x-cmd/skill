---
name: htop
description: >
  Enhanced htop module for real-time monitoring of processes, CPU, and memory across diverse environments.
  Core Scenario: When the user needs a classic, interactive process manager and htop is not installed on the system.
license: MIT
---

# htop - Classic Process & Resource Monitor

The `htop` module ensures that users can always access the classic interactive process viewer, even in environments where it is not pre-installed, by leveraging the `pixi` package manager.

## When to Activate
- When the user wants to view and manage running processes interactively.
- When performing real-time monitoring of CPU, memory, and swap usage.
- When quick installation and execution of `htop` are needed without system-wide changes.

## Core Principles & Rules
- **Environment Agnostic**: Uses `pixi` to provide `htop` in environments where it's missing.
- **Full Argument Support**: Supports all standard `htop` flags and options through the `--` separator.

## Patterns & Examples

### Standard Monitor
```bash
# Launch the interactive process monitor
x htop
```

### Filtered View
```bash
# Monitor processes for a specific user
x htop -- -u root
```

### Help and Version
```bash
# Access htop internal help
x htop -- -h
```

## Checklist
- [ ] Verify if the user wants to monitor a specific user or PID.
- [ ] Confirm if any specific sorting or filtering is required on launch.
