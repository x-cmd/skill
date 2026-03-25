---
name: pick
description: >
  Interactive choice list generator for terminal, used to select items from stdin or command output.
  Core Scenario: When the user needs to create an interactive selection menu for scripts or CLI workflows.
license: MIT
---

# pick - Interactive Selection Menu

The `pick` module allows users to create interactive, searchable selection lists in the terminal. It is ideal for choosing files, processes, or custom options within a shell script.

## When to Activate
- When a script needs to present a list of options for the user to select from.
- When choosing specific files from a large directory listing (`ls | x pick`).
- When performing multi-selection tasks within a terminal workflow (`--limit`).

## Core Principles & Rules
- **Piping Input**: Automatically reads from `stdin` to generate the list.
- **Customization**: Supports setting the number of columns, rows, and selection limits.
- **Output**: Returns the selected item(s) to `stdout`, making it perfect for subshell execution.

## Patterns & Examples

### Select File
```bash
# Interactively select a file and view its status
stat `ls | x pick`
```

### Multi-Selection
```bash
# Select up to two items from a list using Tab
ls | x pick --limit 2
```

### Styled Menu
```bash
# Create a 2-column selection menu with a custom prompt
ls | x pick --col 2 --ask "Choose a project:"
```

## Checklist
- [ ] Confirm the input data source (pipe or argument).
- [ ] Verify if single or multiple selection is required.
- [ ] Ensure the prompt message (`--ask`) is clear.
