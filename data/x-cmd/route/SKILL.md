---
name: route
description: >
  Display the system's network routing table with support for structured data export.
  Core Scenario: When the user needs to inspect the routing table for network path diagnostics.
license: MIT
---

# route - Network Routing Table Viewer

The `route` module provides an interface to view the system's network routing table, allowing users to inspect how network traffic is directed. It supports exporting the table in structured formats.

## When to Activate
- When diagnosing network reachability or routing issues.
- When needing a structured list (CSV) of the current routing table for analysis.

## Core Principles & Rules
- **Experimental Status**: Be aware that this module is currently in an experimental phase.
- **Data Export**: Supports `--csv` for easy parsing.

## Patterns & Examples

### View Routing Table
```bash
# Display the current routing table
x route
```

### Export to CSV
```bash
# Output the routing table as a CSV
x route --csv
```

## Checklist
- [ ] Confirm if the user needs a general view or a structured export.
- [ ] Verify the system environment as routing table formats may vary.
