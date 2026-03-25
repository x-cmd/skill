---
name: rg
description: >
  Modern, high-speed line-oriented search tool (ripgrep) with an interactive FZF interface.
  Core Scenario: When the user needs to perform recursive text searches in large codebases or directories.
license: MIT
---

# rg - High-Performance Text Search

The `rg` module provides a powerful CLI for ripgrep, one of the fastest search tools available. It enhances the experience by adding an interactive FZF application for real-time searching and filtering.

## When to Activate
- When the user needs to search for text patterns recursively across directories.
- When an interactive, searchable TUI is required for exploring code search results.
- When searching within compressed files (`-z`).
- When performing multi-line regex matching.

## Core Principles & Rules
- **High Performance**: Emphasize that it handles large datasets efficiently.
- **Interactive Mode**: Use the default `x rg` (or `--fzfapp`) for a dynamic search experience.
- **Convenient Shortcuts**: In interactive mode, use `alt-a` to select all or `ctrl-o` to edit matches.

## Additional Scenarios
- **File Type Filtering**: Use `-t` to limit search to specific file types (e.g., `python`, `js`).
- **Structured Data**: Supports `--json` for programmatic result processing.

## Patterns & Examples

### Interactive Search
```bash
# Start an interactive search in the current directory
x rg
```

### Search Specific Directory
```bash
# Open interactive TUI to search inside a specific path
x rg --fzfapp ~/.x-cmd.root
```

### Search Compressed Files
```bash
# Recursively search for patterns inside .zip or .gz files
x rg -z "search pattern"
```

## Checklist
- [ ] Confirm the search pattern or regex.
- [ ] Verify if an interactive or standard output is preferred.
- [ ] Check if the search should be limited to specific file types or depths.
