---
name: tldr
description: >
  Collaborative cheat sheets for console commands, providing concise usage examples and explanations.
  Core Scenario: When the user needs a quick reference for common command arguments and practical examples.
license: MIT
---

# tldr - Concise Command Cheat Sheets

The `tldr` module provides simplified, community-driven documentation for thousands of CLI tools. Instead of exhaustive manuals, it focuses on the most common use cases and practical examples.

## When to Activate
- When the user wants a quick summary of how to use a specific command.
- When seeking practical examples for command sub-tasks (e.g., `git checkout`).
- When browsing for command references interactively using `fzf`.

## Core Principles & Rules
- **Conciseness**: Emphasize that this is for "quick lookups," not detailed study.
- **Language Support**: Use `--lang` to retrieve documentation in preferred languages (e.g., `zh`).
- **Interactive Browsing**: Support for the `tlfz` shortcut for fast command discovery.

## Patterns & Examples

### Quick Reference
```bash
# View common usage examples for the 'tar' command
x tldr tar
```

### Subcommand Example
```bash
# Get specific help for git checkout
x tldr git checkout
```

### Interactive Discovery
```bash
# Search for commands interactively using FZF
x tldr --fz
```

## Checklist
- [ ] Confirm the command name the user is inquiring about.
- [ ] Verify if a specific language version is preferred.
