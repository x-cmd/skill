---
name: ccmd
description: >
  Cache command execution results to avoid repeated execution of time-consuming tasks.
  Core Scenario: When the user needs to speed up workflows involving slow network requests (curl) or complex local queries.
license: MIT
---

# ccmd - Command Output Caching

The `ccmd` module caches the stdout of commands for a specified duration, preventing unnecessary re-execution of slow or resource-intensive processes.

## When to Activate
- When performing repeated network requests (e.g., `curl`).
- When running slow diagnostic tools or complex file system searches.
- When a user wants to maintain a "snapshot" of a command's output for a period.

## Core Principles & Rules
- **Duration Format**: Supports time units like `s`, `m`, `h`, `d`, `w` (e.g., `1h` for one hour). Default is `1d`.
- **Command Separation**: The `--` separator is mandatory when providing the command to be executed.
- **Cache Management**: Use `invalidate` to force a refresh or `clear` to remove all cached data.

## Patterns & Examples

### Cache HTTP Request
```bash
# Cache the result of a curl command for 1 hour
x ccmd 1h -- curl https://api.example.com/data
```

### Cache Local Query
```bash
# Cache a geographical IP lookup for 30 minutes
x ccmd 30m -- x ips geo 1.1.1.1
```

### Force Refresh
```bash
# Clear the cache for a specific command to ensure next run is live
x ccmd invalidate curl
```

## Checklist
- [ ] Confirm the desired cache duration.
- [ ] Ensure the `--` separator is correctly placed.
- [ ] Verify if the command output is suitable for caching (i.e., not real-time sensitive).
