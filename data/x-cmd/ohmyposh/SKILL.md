---
name: ohmyposh
description: >
  Enhanced interface for oh-my-posh prompt, offering theme downloads and interactive previews.
  Core Scenario: When the user needs to explore, install, or switch between oh-my-posh prompt themes.
license: MIT
---

# ohmyposh - Enhanced oh-my-posh Management

The `ohmyposh` module facilitates the use of oh-my-posh by providing tools for theme discovery, interactive FZF previews, and one-click application of styles across multiple shells.

## When to Activate
- When the user wants to browse oh-my-posh themes with a live preview.
- When installing or switching between oh-my-posh styles.
- When managing fonts or upgrading the oh-my-posh tool.

## Core Principles & Rules
- **Live Previews**: Default command (`x ohmyposh`) provides a searchable, interactive preview of themes.
- **Config Management**: Overrides the `POSH_THEME` environment variable to ensure consistency.
- **Shell Compatibility**: Trial features (`try`/`untry`) are limited to bash and zsh.

## Patterns & Examples

### Preview and Search
```bash
# Start the interactive FZF preview for oh-my-posh themes
x ohmyposh
```

### Set Global Theme
```bash
# Permanently use the 'montys' theme
x ohmyposh use montys
```

## Checklist
- [ ] Confirm if the user needs to install specific fonts for the theme.
- [ ] Verify the shell type for trial compatibility.
