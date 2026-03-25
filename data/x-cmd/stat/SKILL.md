---
name: stat
description: >
  Display detailed file or file system status with support for structured data exports (JSON, CSV).
  Core Scenario: When the user needs precise metadata about files or file systems for diagnostics or automation.
license: MIT
---

# stat - Enhanced File & File System Status

The `stat` module provides detailed metadata for files and file systems, including permissions, ownership, sizes, and timestamps, with the ability to export this data in structured formats.

## When to Activate
- When the user needs detailed file attributes (permissions, inode, last modified).
- When checking file system usage and limits (e.g., total space, free inodes).
- When exporting file metadata to JSON, CSV, or TSV for automated analysis.
- When browsing file details using an interactive TUI.

## Core Principles & Rules
- **Precision**: Provides detailed, technical metadata beyond standard `ls` output.
- **Export Formats**: Prioritize `--json` or `--csv` for data processing tasks.
- **File System Support**: Use the `-f` flag to target entire file systems instead of individual files.

## Patterns & Examples

### File Metadata
```bash
# View detailed information for a specific file
x stat myfile.txt
```

### File System Status
```bash
# Check the status of the root file system
x stat -f /
```

### JSON Export
```bash
# Get file status as a JSON object
x stat --json myfile.txt
```

## Checklist
- [ ] Confirm if the target is an individual file or a file system.
- [ ] Verify the desired output format (Human-readable, JSON, CSV).
