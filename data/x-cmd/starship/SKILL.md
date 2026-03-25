---
name: starship
description: >
  Enhanced interface for Starship prompt, featuring one-click theme switching and interactive previews.
  Core Scenario: When the user needs to configure, preview, or switch between different Starship prompt themes.
license: MIT
---

# starship - Enhanced Starship Prompt Management

The `starship` module provides a powerful wrapper for the Starship prompt, simplifying theme discovery and configuration through interactive FZF previews and easy switching commands.

## When to Activate
- When the user wants to preview different Starship themes interactively.
- When switching between preset or custom Starship configurations.
- When managing Starship environment variables and prompt features.

## Core Principles & Rules
- **Interactive FZF**: Default behavior launches a searchable preview of all available themes.
- **Config Overrides**: Using this module will override the `STARSHIP_CONFIG` environment variable.
- **Session Trial**: Use `try` to test a theme in the current session (bash/zsh only).

## Patterns & Examples

### Interactive Preview
```bash
# Browse and preview Starship themes interactively
x starship
```

### Apply Theme
```bash
# Permanently set a specific theme
x starship use gruvbox-rainbow
```

### Current Config
```bash
# View details of the currently active theme
x starship current
```

## Checklist
- [ ] Confirm if the theme should be applied globally or just for the session.
- [ ] Verify if Starship is properly initialized in the user's shell.
