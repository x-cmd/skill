---
name: ws
description: >
  Workspace management tool providing isolated environments for command execution and file management.
  Core Scenario: When the user needs to execute commands or manage files within a specific project workspace.
license: MIT
---

# ws - Workspace Management Utility

The `ws` module provides tools for managing project workspaces, allowing for isolated command execution and file access within a defined workspace root.

## When to Activate
- When the user wants to identify the root directory of the current workspace.
- When executing commands or script files relative to the workspace root.
- When viewing file contents within the workspace environment.

## Core Principles & Rules
- **Isolation**: Commands executed via `ws` are contextually aware of the workspace root.
- **Path Awareness**: Uses relative paths from the workspace `.x-cmd` directory for internal operations.

## Patterns & Examples

### Show Workspace Root
```bash
# Display the root directory of the current workspace
x ws --root
```

### Execute in Workspace
```bash
# Run a specific command within the workspace context
x ws exec my-command
```

## Checklist
- [ ] Confirm if the command should be run in the current directory or workspace root.
- [ ] Verify the workspace has been properly identified.
