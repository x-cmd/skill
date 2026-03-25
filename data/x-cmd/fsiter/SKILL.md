---
name: fsiter
description: >
  Flexible file system iteration and search tool with DFS support and filtering.
  Core Scenario: When the user needs to traverse directory trees, filter files/folders, or check for existence/emptiness.
license: MIT
---

# fsiter - File System Iteration & Search

The `fsiter` module is a powerful alternative to `ls` and `find`, providing flexible filtering, depth-first search (DFS) traversal, and script-friendly queries.

## When to Activate
- When the user wants to list specific file types (e.g., hidden-only, files-only).
- When performing recursive directory traversal with depth limits.
- When checking if a directory is empty or if specific patterns exist.
- When counting files or directories in a tree for reporting.

## Core Principles & Rules
- **Depth-First Search**: Use `--dfs` for controlled tree traversal with custom callbacks.
- **Efficient Filtering**: Combine `--file`, `--folder`, and `--hidden` to pinpoint targets.
- **Script-Ready**: Subcommands like `--filecount` and `--dirempty` provide reliable exit codes and values for automation.

## Additional Scenarios
- **Existence Check**: Quickly verify if patterns exist using `--exist`.
- **Recursive Processing**: Use `--dfs` with depth limits to process files in specific tree levels.

## Patterns & Examples

### List Specific Items
```bash
# List all hidden files in the current directory
x fsiter --ls --file --hidden-only
```

### Depth-Limited Search
```bash
# Traverse the tree up to 2 levels deep and echo item names
x fsiter --dfs /path/to/dir 0 2 echo
```

### Query Statistics
```bash
# Count items in a log directory
x fsiter --filecount /var/log
```

### Conditional Logic
```bash
# Delete a directory only if it is empty
x fsiter --dirempty ./temp && rmdir ./temp
```

## Checklist
- [ ] Confirm if the search should include hidden items.
- [ ] Verify the required depth for DFS traversal.
- [ ] Ensure the callback command for `--dfs` is correctly specified.
