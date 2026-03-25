---
name: grep
description: >
  Pattern search tool with an interactive FZF-powered application for real-time result viewing.
  Core Scenario: When the user needs to search for text patterns and explore results interactively.
license: MIT
---

# grep - Interactive Pattern Search

The `grep` module enhances the standard global regular expression search tool by integrating FZF, providing an interactive environment to search files and view results in real-time.

## When to Activate
- When the user wants to search for text patterns within files or directories.
- When an interactive, searchable list of grep matches is required (`--fz`).
- When performing rapid code or log investigation via terminal.

## Core Principles & Rules
- **Interactive First**: Use `--fzfapp` (or default `x grep`) for a dynamic search experience.
- **Scripting Fallback**: Supports the standard `grep` flags for non-interactive use via the `cmd` subcommand.
- **Context Support**: Fully supports standard flags like `-A`, `-B`, and `-C` for context lines.

## Patterns & Examples

### Interactive Search
```bash
# Start an interactive search in the current directory
x grep
```

### Search via Pipe
```bash
# Pipe command output into an interactive grep session
x ascii | x grep
```

### Search Specific Dir
```bash
# Open interactive TUI to search inside a specific path
x grep --fzfapp ~/.x-cmd.root
```

## Checklist
- [ ] Confirm the search pattern or regex.
- [ ] Verify if an interactive view or standard output is preferred.
- [ ] Ensure the search scope (file or directory) is correctly targeted.
