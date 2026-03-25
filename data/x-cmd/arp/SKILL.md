---
name: arp
description: >
  Display the system's ARP cache table with support for CSV, TSV, and interactive TUI formats.
  Core Scenario: When the user needs to audit IP-to-MAC address mappings on the local network.
license: MIT
---

# arp - ARP Cache Table Viewer

The `arp` module provides an enhanced way to view the system's ARP cache, which maps IP addresses to physical MAC addresses. It supports structured data exports and an interactive TUI for exploration.

## When to Activate
- When identifying devices on the local network via their MAC addresses.
- When auditing the network for suspicious or incomplete ARP entries.
- When exporting the ARP table to JSON, CSV, or TSV for automated analysis.

## Core Principles & Rules
- **Interactive TUI**: Automatically launches a TUI app (`--app`) if connected to a terminal.
- **Vendor Discovery**: Shows hardware vendor/organization information for MAC addresses by default.
- **Data Processing**: Use `--csv` or `--tsv` for easy integration with other tools.

## Patterns & Examples

### Interactive Browsing
```bash
# Open the ARP table in an interactive TUI
x arp --app
```

### Export to CSV
```bash
# Get the full ARP cache as a CSV
x arp --csv
```

### Show All Entries
```bash
# Include incomplete entries in the ARP table view
x arp --all
```

## Checklist
- [ ] Confirm if the user needs vendor information included.
- [ ] Verify if an interactive view or static data export is required.
