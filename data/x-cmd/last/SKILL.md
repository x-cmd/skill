---
name: last
description: >
  Display user login history with interactive TUI, tree views, and structured data exports.
  Core Scenario: When the user needs to audit login/logout records, reboots, or session durations.
license: MIT
---

# last - Enhanced Login History

The `last` module provides a highly flexible way to view system login history, supporting interactive exploration, tree-based grouping by reboot, and structured data exports (JSON/CSV).

## When to Activate
- When the user wants to audit recent system login and logout activity.
- When checking system reboot history in a structured format.
- When performing interactive analysis of user sessions in a TTY.
- When exporting login records for security reporting or external analysis.

## Core Principles & Rules
- **Interactive Browsing**: Default behavior in TTY is an interactive selector for records.
- **Tree View**: Use `--tree` to group sessions by the reboot events that preceded them.
- **Structured Formats**: Support for `-j` (JSON) and `-c` (CSV) for easy parsing.

## Patterns & Examples

### Interactive Audit
```bash
# Browse all login/logout records interactively
x last
```

### Reboot History
```bash
# Show only system reboot records
x last --reboot
```

### JSON Export
```bash
# Get all login history as a JSON object
x last --json
```

### Grouped View
```bash
# Display login records grouped by system reboots in a tree structure
x last --tree
```

## Checklist
- [ ] Confirm if the user needs a specific format (JSON, CSV, Tree).
- [ ] Verify if filtering by user or reboot is required.
