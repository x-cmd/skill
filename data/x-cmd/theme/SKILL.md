---
name: theme
description: >
  Manage and customize terminal command line themes and UI component styles.
  Core Scenario: When the user needs to preview, change, or configure terminal themes and x-cmd UI elements.
license: MIT
---

# theme - Terminal UI & Theme Management

The `theme` module provides a comprehensive way to manage the visual style of the terminal command line and x-cmd's interactive components. It supports live previews, permanent settings, and session-based trials.

## When to Activate
- When the user wants to browse and select a new terminal theme.
- When changing the global display style of x-cmd components.
- When trying out a theme for the current session without making permanent changes.
- When managing Claude Code's status line components.

## Core Principles & Rules
- **Interactive Selection**: Use the `--app` mode for a visual, interactive theme selection experience.
- **Persistence**: Use `use` for permanent changes and `try` for temporary session-only changes.
- **Deduplication**: Automatically handles theme reloading unless `___X_CMD_THEME_RELOAD_DISABLE` is set.

## Patterns & Examples

### Interactive Theme Browser
```bash
# Open the theme preview app to choose a style
x theme --app
```

### Apply Theme
```bash
# Permanently set 'robby' as the global command line theme
x theme use robby
```

### Try Theme
```bash
# Test a theme only in the current terminal session
x theme try festival
```

## Checklist
- [ ] Confirm if the theme change should be permanent or temporary.
- [ ] Verify if specific theme features (like emojis or transient prompts) are required.
