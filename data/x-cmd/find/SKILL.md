---
name: find
description: >
  Search for files and directories in a directory hierarchy with an interactive FZF application.
  Core Scenario: When the user needs to find files based on name, size, type, or other metadata.
license: MIT
---

# find - Enhanced File Search

The `find` module extends the standard system find utility, offering an interactive FZF-powered application for rapid file discovery and metadata-based filtering.

## When to Activate
- When the user wants to locate files or directories based on patterns or attributes.
- When an interactive, searchable list of files is required (`--fzfapp`).
- When finding files by size (e.g., larger than 100MB) or modification time.

## Core Principles & Rules
- **Interactive Mode**: Automatically launches a TUI app for easy browsing if no specific criteria are given.
- **Attribute-Based**: Supports standard find flags like `-type`, `-name`, `-size`, and `-mtime`.
- **Actionable Results**: Can execute commands on found items using `-exec`.

## Patterns & Examples

### Interactive Discovery
```bash
# Open interactive FZF app to find files or folders
x find
```

### Search by Name
```bash
# Find all text files in the current directory
x find . -name "*.txt"
```

### Large File Search
```bash
# Find files larger than 100MB in the home directory
x find /home -size +100M
```

## Checklist
- [ ] Confirm the search path and criteria (name, size, type).
- [ ] Verify if an interactive view or direct command execution (`-exec`) is needed.
- [ ] Ensure the user is aware of the recursive nature of the search.
