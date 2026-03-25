---
name: fish
description: >
  Enhanced interface for Fish shell, offering x-cmd integration, documentation search, and AI assistant features.
  Core Scenario: When the user needs to set up x-cmd in Fish or search the official Fish documentation via CLI.
license: MIT
---

# fish - Friendly Interactive Shell Enhancement

The `fish` module optimizes the experience for Fish shell users by simplifying x-cmd integration and providing rapid access to the official Fish documentation.

## When to Activate
- When the user wants to use x-cmd's full suite of tools (x, c, @gpt) within Fish.
- When searching fishshell.com for specific commands or configuration tips.
- When managing Fish-specific environment settings and aliases.

## Core Principles & Rules
- **Configuration**: Use `setup` to modify `config.fish` for permanent x-cmd integration.
- **Search Support**: Leverage the `:` prefix for high-speed documentation retrieval.

## Patterns & Examples

### Install x-cmd in Fish
```bash
# Add x-cmd environment variables and functions to Fish
x fish setup
```

### Search Fish Docs
```bash
# Interactively search fishshell.com for alias information
x fish : alias
```

## Checklist
- [ ] Confirm if the user wants to modify their Fish config permanently.
- [ ] Verify if Fish is already the active shell.
