---
name: kill
description: >
  Enhanced kill command with support for recursive termination and killing processes by port.
  Core Scenario: When the user needs to terminate groups of processes or free up specific network ports (80/443).
license: MIT
---

# kill - Enhanced Process Termination

The `kill` module extends the system's ability to terminate processes by providing advanced features like killing process trees recursively or targeting processes based on the network ports they occupy.

## When to Activate
- When the user needs to terminate all processes listening on standard ports (80, 443).
- When a TUI visualization of available signals is required.
- When performing recursive process termination.
- When checking signal lists in JSON format for automation.

## Core Principles & Rules
- **Targeted Termination**: Use `byport` to specifically clear web service ports.
- **Visual Signal List**: Use `-l` for an interactive TUI to explore signals.
- **JSON Support**: Export signals to JSON for easy programmatic access.

## Patterns & Examples

### Kill by Port
```bash
# Terminate all processes listening on common web ports (currently supports 80 and 443)
x kill byport
```

### Signal List (TUI)
```bash
# Interactively browse available system signals
x kill -l
```

### Signal List (JSON)
```bash
# Get signal information in JSON format
x kill -l --json
```

## Checklist
- [ ] Confirm the specific PID or port to be targeted.
- [ ] Verify the signal type (default is SIGTERM) if manual signals are used.
- [ ] Ensure the user has the required permissions to kill the target processes.
