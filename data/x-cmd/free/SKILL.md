---
name: free
description: >
  Display memory usage with support for CSV/TSV formats and cross-platform (Linux/macOS) compatibility.
  Core Scenario: When the user needs to monitor RAM usage, buffer/cache, or export memory data for automation.
license: MIT
---

# free - Memory Usage Reporting

The `free` module enhances the standard memory reporting tool by adding structured data support (CSV/TSV) and providing a native implementation for macOS users who typically lack the `free` command.

## When to Activate
- When monitoring total, used, and free system memory.
- When performing memory usage analysis in scripts using CSV/TSV formats.
- When checking memory compression details on macOS.
- When needing periodic memory reporting (polling).

## Core Principles & Rules
- **Cross-Platform**: Provides a consistent interface for memory stats on both Linux and macOS.
- **Automation-Ready**: Supports `--csv` and `--tsv` for easy parsing without headers.
- **Real-Time Polling**: Use `-s` and `-c` to repeat the output at intervals.

## Patterns & Examples

### Human-Readable View
```bash
# Display system memory usage in a clear format
x free
```

### Export for Scripting
```bash
# Get memory data as a single-line CSV without headers
x free --csv --no-header
```

### Periodic Monitoring
```bash
# Refresh memory stats every 2 seconds for 10 iterations
x free -s 2 -c 10
```

## Checklist
- [ ] Verify if the output needs to be structured (CSV/TSV) for a script.
- [ ] Confirm if headers should be included or omitted.
- [ ] Ensure the correct polling interval is set if monitoring over time.
