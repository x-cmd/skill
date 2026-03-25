---
name: tmp
description: >
  Cross-platform management of temporary files and directories.
  Core Scenario: When the user needs to generate temporary paths or create temporary storage for scripts and applications.
license: MIT
---

# tmp - Temporary File & Directory Management

The `tmp` module provides a cross-platform way to handle temporary storage, automatically identifying the system's temporary directory ($TMPDIR) and providing tools for safe creation.

## When to Activate
- When a script needs a safe path for intermediate files.
- When creating temporary directories with random suffixes to avoid collisions.
- When managing a custom temporary root for a session or application.

## Core Principles & Rules
- **Cross-Platform Safety**: Always rely on `x tmp path` instead of hardcoding `/tmp` to ensure compatibility across OS types (including Termux).
- **Efficiency for Scripts**: Use underscored subcommands (e.g., `path_`, `mkdir_`) in scripts to store results directly in the `x_` variable.
- **Prefixing**: Use `--prefix` when creating directories to ensure unique, identifiable temporary storage.

## Patterns & Examples

### Get Temporary Path
```bash
# Get a safe temporary path for a specific app subdirectory
x tmp path myapp/cache
```

### Create Unique Directory
```bash
# Create a temporary directory with a specific prefix
x tmp mkdir --prefix mytask_
```

### Create Empty Temp File
```bash
# Quickly generate an empty file in the system temp directory
x tmp mkfile session_data.txt
```

## Checklist
- [ ] Verify if a specific subdirectory structure is needed within the temp path.
- [ ] Confirm if the directory should have a random suffix (`--prefix`).
- [ ] Check if a custom temporary root is required for the current session.
