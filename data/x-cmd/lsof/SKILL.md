---
name: lsof
description: >
  Enhanced lsof interface for viewing open files with TUI support and structured data exports (TSV, CSV).
  Core Scenario: When the user needs to identify which files are open by which processes, supporting interactive search.
license: MIT
---

# lsof - List Open Files Enhancement

The `lsof` module provides a more intuitive way to interact with the system's open file list. It supports an interactive TUI for exploration and allows exporting results in common data formats.

## When to Activate
- When investigating file locking issues or identifying process ownership of files.
- When performing interactive searches of open files using FZF.
- When exporting open file data to CSV or TSV for script processing.
- When needing thread-level details (`--task`) for open files.

## Core Principles & Rules
- **Interactive TUI**: Automatically displays a TUI when connected to a terminal; otherwise, defaults to TSV.
- **Data-Centric**: Use `--csv` or `--tsv` when the output is intended for automation or external analysis.
- **Searchable**: Leverage the `fz` subcommand for fast interactive filtering.

## Patterns & Examples

### Interactive Search
```bash
# Interactively search and view open files via FZF
x lsof fz
```

### Export to CSV
```bash
# Output current open files as a CSV for processing
x lsof --csv
```

### View Thread Info
```bash
# Display task (thread) IDs and names for open files
x lsof --task
```

## Checklist
- [ ] Confirm if the user needs an interactive view or a static data export.
- [ ] Verify if thread-level information is required.
- [ ] Ensure the correct export format (CSV/TSV) is selected if needed.
