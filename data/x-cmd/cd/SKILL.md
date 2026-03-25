---
name: cd
description: >
  Enhanced cd command with history navigation, fuzzy search, and directory aliases.
  Core Scenario: When the user needs to quickly jump between historical paths or search for subdirectories using keywords.
license: MIT
---

# cd - Enhanced Directory Navigation

The `cd` module is a powerful replacement for the standard `cd` command, implemented in shell and awk. It offers interactive history navigation, keyword-based jumping, and directory aliasing.

## When to Activate
- When the user wants to jump to a directory based on a keyword in its path.
- When navigating back to a parent directory using a keyword match (`-b`).
- When performing a forward search for subdirectories (`-f`).
- When managing or using directory "realms" (aliases).
- When executing a command in another directory and returning immediately.

## Core Principles & Rules
- **Seamless Jump**: Use `,` for interactive history search or `,keyword` for direct jumps.
- **Bi-directional Navigation**: Use `-b` to go "backwards" (up the tree) and `-f` to go "forwards" (down into subfolders).
- **Execution Wrappers**: Support for executing commands in a target directory without permanently changing the session path.

## Patterns & Examples

### Smart Jumping
```bash
# Open interactive history search
c
# Jump to a historical path matching 'project'
c ,project
```

### Contextual Navigation
```bash
# Go up the tree to the nearest directory matching 'src'
c -b src
# Find and jump into a subfolder matching 'test'
c -f test
```

### Temporary Execution
```bash
# Run 'ls -la' in /bin and then return automatically
c /bin ls -la
```

## Checklist
- [ ] Confirm if the jump is based on history (`,`) or parent/child relationship.
- [ ] Verify if the user intended to stay in the target directory or just run a command.
- [ ] Check if a directory alias (`:realm`) exists if used.
