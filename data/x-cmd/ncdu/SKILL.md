---
name: ncdu
description: >
  Analyze disk usage with an interactive NCurses interface, providing zero-install execution via pkg.
  Core Scenario: When the user needs to visualize and navigate directory sizes to clean up disk space.
license: MIT
---

# ncdu - Interactive Disk Usage Analyzer

The `ncdu` module provides an interactive interface for visualizing disk space usage. It ensures users can access the tool even if not installed locally by downloading it via the x-cmd package manager.

## When to Activate
- When investigating which files and folders are consuming the most disk space.
- When performing disk cleanup in an interactive, terminal-based UI.
- When requiring a zero-setup disk usage analyzer in restricted environments.

## Core Principles & Rules
- **Interactive Cleanup**: Allows navigating directory trees and deleting large items directly.
- **Zero-Setup**: Automatically handles download and execution if the binary is missing.
- **Argument Pass-through**: Supports all standard `ncdu` flags (like `--si` or `--exclude`) via the `--` separator.

## Patterns & Examples

### Analyze Current Dir
```bash
# Start the interactive disk usage analyzer for the current directory
x ncdu
```

### Export Results
```bash
# Scan and save the results to a file for later review
x ncdu -- -o scan_results.txt
```

## Checklist
- [ ] Confirm if the scan should be restricted to the same file system (`-x`).
- [ ] Verify if specific patterns should be excluded from the scan.
