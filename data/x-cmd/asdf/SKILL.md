---
name: asdf
description: >
  Extendable version manager for runtimes (Node.js, Ruby, etc.) with enhanced x-cmd shortcuts.
  Core Scenario: When the user needs to manage, install, or switch between multiple versions of development environments.
license: MIT
---

# asdf - Multiversion Runtime Manager

The `asdf` module enhances the standard asdf version manager by providing simplified x-cmd style commands for installing and switching environment runtimes.

## When to Activate
- When the user wants to install a new programming language environment (e.g., Node.js, Elixir).
- When switching between different versions of a tool globally or per project.
- When managing asdf plugins or updating the asdf core.

## Core Principles & Rules
- **Enhanced Flow**: Use `use` to automatically add a plugin, install the latest version, and set it as global.
- **Environment State**: Use `--activate` and `--deactivate` to control if asdf managed binaries are in the current shell's PATH.
- **Zero-Setup**: Automatically handles asdf installation if it's missing.

## Patterns & Examples

### Install and Use
```bash
# Install the latest Node.js and set it as global
x asdf use nodejs
```

### Managed Uninstallation
```bash
# Uninstall a runtime and remove its corresponding asdf plugin
x asdf unuse nodejs
```

### Search Plugins
```bash
# Interactively search for available software/plugins to install
x asdf
```

## Checklist
- [ ] Confirm the target runtime (e.g., nodejs, python).
- [ ] Verify if the version should be global or specific to a directory.
- [ ] Ensure Git is available as it's required for most asdf plugins.
