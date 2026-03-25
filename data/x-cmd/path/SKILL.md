---
name: path
description: >
  Manage the system PATH variable with subcommands for adding, removing, and deduplicating directories.
  Core Scenario: When the user needs to modify their environment's search path or clean up duplicate entries.
license: MIT
---

# path - PATH Variable Management

The `path` module provides a simple and effective way to manipulate the shell's PATH variable. It supports pushing/unshifting directories and deduplicating entries to maintain a clean environment.

## When to Activate
- When the user wants to add a new directory to the start (`unshift`) or end (`push`) of the PATH.
- When removing specific directories from the PATH.
- When cleaning up duplicate directory entries in the PATH variable.
- When viewing the current PATH structure in an interactive app.

## Core Principles & Rules
- **Non-Destructive**: Provides explicit subcommands for modification rather than direct string editing.
- **Interactive UI**: Use the default `x path` to browse the current PATH entries in a structured list.
- **Ordering**: Be mindful of whether a path should take priority (start of PATH) or be a fallback (end of PATH).

## Patterns & Examples

### Add to PATH
```bash
# Add a directory to the beginning of the PATH
x path unshift /usr/local/custom/bin
```

### Remove Duplicates
```bash
# Remove all redundant entries from the PATH variable
x path uniq
```

### View and List
```bash
# List all directories currently in the PATH
x path ls
```

## Checklist
- [ ] Confirm if the directory should be added to the front or back of the PATH.
- [ ] Verify the target directory path exists and is correct.
