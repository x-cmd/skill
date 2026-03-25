---
name: ll
description: >
  Search and list information about files, archives, and operating system resources using fzf.
  Core Scenario: When the user needs an interactive way to explore file metadata or system resource summaries.
license: MIT
---

# ll - Interactive Resource Listing

The `ll` module provides an interactive interface, primarily leveraging `fzf`, to search and list information about the file system and key operating system resources like CPU, memory, and processes.

## When to Activate
- When the user wants to search for files or directories interactively using `fzf`.
- When performing a quick interactive check of system CPU, memory, or process states.

## Core Principles & Rules
- **Interactive Search**: Use the `--fzfapp` (or default `x ll`) to search through files and system data.
- **Resource Shortcuts**: Supports prefixed subcommands (e.g., `:cpu`, `:mem`) for quick system snapshots.

## Patterns & Examples

### Interactive File Search
```bash
# Start an interactive fzf session to browse files and directories
x ll
```

### Quick System Check
```bash
# View current memory or process state via ll
x ll :mem
x ll :ps
```

## Checklist
- [ ] Confirm if the user needs an interactive search or a specific resource snapshot.
