---
name: uninstall
description: >
  Unified software uninstaller for various package managers (brew, scoop, apt, etc.).
  Core Scenario: When the user needs to uninstall software regardless of the package manager used.
license: MIT
---

# uninstall - Unified Package Uninstaller

The `uninstall` module provides a single interface to query and remove installed software across multiple package managers including Homebrew, Scoop, APT, DNF, and more.

## When to Activate
- When the user wants to uninstall a program without remembering which manager installed it.
- When performing a system cleanup of installed packages.
- When uninstalling x-cmd itself.

## Core Principles & Rules
- **Interactive Cleanup**: Launches an interactive list of all installed packages if no arguments are given.
- **Multimanager Support**: Automatically detects and delegates removal to the appropriate backend (e.g., brew, scoop).

## Patterns & Examples

### Interactive Uninstall
```bash
# Browse all installed software and select items to remove
x uninstall
```

### Targeted Removal
```bash
# Explicitly uninstall 'bat' from Homebrew
x uninstall brew bat
```

### Self-Removal
```bash
# Remove x-cmd from the system
x uninstall self
```

## Checklist
- [ ] Confirm the name of the software to be removed.
- [ ] Verify if a specific package manager should be targeted.
- [ ] Remind the user that uninstallation is typically irreversible.
