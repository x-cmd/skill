---
name: ps
description: >
  Display running process information with support for interactive UI and structured exports (CSV, JSON, TSV).
  Core Scenario: When the user needs to analyze system processes or export process data for external automation.
license: MIT
---

# ps - Enhanced Process Status

The `ps` module extends the standard process viewing capability by adding interactive TUI features and the ability to export process snapshots in structured data formats like JSON and CSV.

## When to Activate
- When the user wants to browse processes using an interactive UI (FZF or CSV app).
- When exporting current process lists to JSON, CSV, or TSV for scripting.
- When converting legacy `ps aux` output into structured data formats.

## Core Principles & Rules
- **Interactive Exploration**: Use `fz` for a fast, searchable TUI experience.
- **Data Exporting**: Prioritize `--json` or `--csv` when the user intends to process the data with other tools (like `jq`).
- **Piping Compatibility**: Can consume output from standard `ps` commands and transform it.

## Patterns & Examples

### Interactive UI
```bash
# View processes in a searchable interactive TUI
x ps fz
```

### Export to JSON
```bash
# Output current process information as a JSON array
x ps --json
```

### Data Transformation
```bash
# Convert a specific ps command output to CSV
ps -ef | x ps --tocsv
```

## Checklist
- [ ] Confirm if the user needs an interactive view or a static data export.
- [ ] Verify the desired output format (JSON, CSV, TSV).
- [ ] Ensure specific `ps` flags are used if narrowing down the process list.
