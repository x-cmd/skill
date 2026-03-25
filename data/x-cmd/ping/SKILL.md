---
name: ping
description: >
  Enhanced ping command with visual data representations like heatmaps and bar charts.
  Core Scenario: When the user needs to monitor network reachability with visual metrics.
license: MIT
---

# ping - Enhanced Network Reachability

The `ping` module provides an enhanced interface for the standard connectivity test, supporting multiple visualization modes to better understand network performance and latency.

## When to Activate
- When checking if a remote host is reachable via ICMP.
- When needing a visual representation (heatmap, bar chart) of network latency over time.
- When exporting reachability data to CSV or TSV for analysis.

## Core Principles & Rules
- **Visualization**: Use `--heatmap` or `--bar` for immediate visual feedback on latency stability.
- **Data-Friendly**: Supports structured output formats for integration with monitoring tools.

## Patterns & Examples

### Visual Monitoring
```bash
# Ping a host and display a real-time heatmap of latency
x ping --heatmap bing.com
```

### Data Export
```bash
# Ping a host 10 times and output results as CSV
x ping -c 10 --csv 8.8.8.8
```

## Checklist
- [ ] Confirm the target host address.
- [ ] Verify if a specific visualization mode is requested.
