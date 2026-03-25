---
name: winget
description: >
  Enhanced interface for Microsoft's WinGet package manager, supporting mirrors and proxy configuration.
  Core Scenario: When the user needs to manage Windows applications using the official native package manager.
license: MIT
---

# winget - Enhanced Windows Package Manager

The `winget` module provides a more user-friendly interface for Microsoft's official Windows Package Manager, simplifying tasks like mirror switching and proxy setup.

## When to Activate
- When installing, upgrading, or removing Windows software using the official WinGet tool.
- When configuring WinGet to use regional mirrors for better speed.
- When searching for available Windows applications via terminal.

## Core Principles & Rules
- **Native Support**: Uses Microsoft's official backend for software acquisition.
- **Regional Optimization**: Supports switching to regional mirrors (e.g., USTC).

## Patterns & Examples

### Install Software
```bash
# Use WinGet to install 7-zip
x winget install 7zip
```

### Configure Mirror
```bash
# Set WinGet to use a regional mirror for faster downloads
x winget mirror set ustc
```

## Checklist
- [ ] Confirm if the user is on a Windows environment.
- [ ] Verify the exact software name or ID for installation.
