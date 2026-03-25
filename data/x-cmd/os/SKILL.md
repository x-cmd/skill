---
name: os
description: >
  Retrieve operating system information including architecture, release version, and uptime.
  Core Scenario: When the user needs to check system specs, platform type, or runtime duration for scripting or diagnostics.
license: MIT
---

# os - Operating System Information

The `os` module provides detailed information about the current operating system, including its name, architecture, release details, load averages, and more. It is essential for writing cross-platform scripts.

## When to Activate
- When the user needs to identify the system architecture (e.g., x86_64, arm64).
- When checking if the environment is Linux, macOS, WSL, or Termux.
- When monitoring system load or uptime.
- When retrieving release information for software compatibility checks.

## Core Principles & Rules
- **Efficient Scripting**: Use variables like `name_`, `arch_`, and `uptime_` in scripts for high-performance data retrieval.
- **Cache Control**: Subcommands starting with `get` (e.g., `getname`) bypass caches for real-time detection.
- **Cross-Platform Readiness**: Leverage `os is <type>` for robust conditional logic in shell scripts.

## Patterns & Examples

### Basic Information
```bash
# Show OS name and architecture
x os name
x os arch
```

### Environment Check
```bash
# Test if running in WSL (returns 0 or 1)
x os is wsl
```

### System Performance
```bash
# Display uptime and load averages
x os uptime
x os loadavg
```

## Checklist
- [ ] Confirm if real-time detection is required (bypassing cache).
- [ ] Verify the specific information requested (name vs release).
