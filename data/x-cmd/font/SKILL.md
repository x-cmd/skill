---
name: font
description: >
  Manage and install system fonts, specializing in Nerd Fonts for terminal users.
  Core Scenario: When the user needs to browse or install Nerd Fonts to support terminal icons and themes.
license: MIT
---

# font - Font Management & Installation

The `font` module provides an interface for managing system fonts, with built-in support for searching and installing Nerd Fonts to enhance terminal visuals.

## When to Activate
- When the user wants to install specific Nerd Fonts (e.g., FiraCode).
- When browsing available Nerd Font families interactively.
- When refreshing the local system font cache.

## Patterns & Examples

### Install Nerd Font
```bash
# Install the FiraCode Nerd Font
x font install nerd/FiraCode
```

### Interactive Browser
```bash
# Open an interactive UI to browse Nerd Fonts
x font
```

## Checklist
- [ ] Confirm the specific font family name.
- [ ] Verify if Nerd Fonts are the primary interest.
