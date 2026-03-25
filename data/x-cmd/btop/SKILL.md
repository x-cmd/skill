---
name: btop
description: >
  Modern resource monitor showing CPU, memory, disks, network, and processes with interactive filtering.
  Core Scenario: When the user needs a detailed, visual resource monitor with support for memory analysis and process filtering.
license: MIT
---

# btop - System Resource Monitor

`btop` is a high-performance system monitor that displays real-time statistics for CPU, memory, disks, network, and processes. It features preset views and interactive filtering for deep system analysis.

## When to Activate
- When the user wants a detailed, visual breakdown of system resource usage.
- When performing memory analysis using the process-centric view.
- When filtering and managing processes interactively based on PID or name.
- When requiring a resource monitor that works across Linux, macOS, and Windows without complex setup.

## Core Principles & Rules
- **Preset Views**: Utilize presets (`-p`) to focus on different resources (e.g., `-p 1` for memory/processes).
- **Interactive Controls**: Leverage keyboard shortcuts like `m` (sort by memory), `c` (sort by CPU), and `/` (filter) for rapid navigation.
- **Zero-Dependency Run**: Automatically downloads and runs via `pixi` or `scoop` if not found locally.

## Patterns & Examples

### Default View
```bash
# Start btop showing all system resources
x btop
```

### Memory-Centric View
```bash
# Start with the process-centric preset for easier memory analysis
x btop -p 1
```

### CPU-Centric Preset
```bash
# Launch with CPU performance as the primary focus
x btop -- --preset 2
```

## Checklist
- [ ] Confirm if a specific preset view (memory, CPU, network) is preferred.
- [ ] Ensure the user is aware of interactive shortcuts (`?` for help).
