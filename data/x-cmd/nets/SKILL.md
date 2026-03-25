---
name: nets
description: >
  Enhanced netstat interface for viewing internet connections, routing tables, and network statistics.
  Core Scenario: When the user needs an interactive way to audit network sockets (TCP/UDP) or analyze network tables.
license: MIT
---

# nets - Enhanced Network Statistics

The `nets` module provides a user-friendly way to display network statistics, improving upon traditional `netstat`. It supports interactive views and structured data formats like CSV and TSV.

## When to Activate
- When the user wants to audit active TCP/UDP connections (internet table).
- When analyzing the system's routing table in a readable format.
- When needing structured network data (CSV/TSV) for monitoring scripts.
- When performing a quick interactive overview of all network tables.

## Core Principles & Rules
- **Interactive Browsing**: Default mode allows users to select which network table to view.
- **Structured Formats**: Support for `--csv` and `--tsv` for integration with other data processing tools.
- **Refresh Control**: Use `update` to ensure the table data is current.

## Patterns & Examples

### View Connections
```bash
# Show all active internet (TCP/UDP) connections
x nets view internet
```

### Export Routing Table
```bash
# Output the routing table as a TSV for analysis
x nets view --tsv route
```

## Checklist
- [ ] Confirm the specific network table (internet, route, etc.) to be viewed.
- [ ] Verify if the output needs to be structured (CSV/TSV) for automation.
