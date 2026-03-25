---
name: ls
description: >
  Enhanced ls command providing unified access to system resources like CPU, memory, and processes.
  Core Scenario: When the user needs a quick summary of files and system resource states via enhanced subcommands.
license: MIT
---

# ls - Enhanced Resource Listing

The `ls` module extends the traditional file listing command to provide rapid access to various system resources (CPU, Memory, Processes, PATH) through a unified interface.

## When to Activate
- When the user wants to view file info alongside system resource summaries.
- When performing a quick check of PATH variables or process lists within a listing workflow.
- When using an interactive app (`:app`) to browse file metadata.

## Core Principles & Rules
- **Resource Focused**: Use prefixed subcommands (starting with `:`) to target specific system data.
- **Interactive UI**: Use `:app` for a richer metadata exploration than standard listing.

## Patterns & Examples

### System Summary
```bash
# Quickly view CPU and Memory info using enhanced ls
x ls :cpu
x ls :mem
```

### Environment Check
```bash
# List all directories currently in the PATH variable
x ls :path
```

### Metadata Explorer
```bash
# Open an interactive viewer for files and directory information
x ls :app
```

## Checklist
- [ ] Confirm if the user needs file listing or a system resource summary.
- [ ] Verify if an interactive view (`:app`) is preferred over static output.
