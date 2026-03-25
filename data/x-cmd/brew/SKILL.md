---
name: brew
description: >
  Enhanced Homebrew wrapper for macOS and Linux, managing packages, mirrors, and proxies.
  Core Scenario: When the user needs to install, manage, or configure Homebrew packages with optimized downloads.
license: MIT
---

# brew - Enhanced Homebrew Wrapper

The `brew` module enhances the standard Homebrew experience by providing easier access to common tasks like mirror configuration, proxy setup, and interactive package selection.

## When to Activate
- When installing, removing, or listing Homebrew packages on macOS or Linux.
- When configuring Homebrew to use optimized regional mirrors (e.g., TUNA).
- When managing Homebrew's proxy settings for restricted networks.
- When performing interactive package browsing (`x brew` app).

## Core Principles & Rules
- **Interactive Browsing**: Use the default `x brew` to choose packages via TUI.
- **Mirror Support**: Simplify mirror switching with the `mirror` subcommand.
- **Privacy Focus**: Easily disable Homebrew analytics via the `analytics` subcommand.

## Patterns & Examples

### Install Packages
```bash
# Install multiple packages at once
x brew install curl wget git
```

### Configure Mirror
```bash
# Set Homebrew to use the Tsinghua University mirror
x brew mirror set tuna
```

### Interactive App
```bash
# Launch interactive TUI to manage Homebrew packages
x brew
```

## Checklist
- [ ] Confirm the names of the packages to be installed or managed.
- [ ] Verify if a mirror or proxy setup is needed for better connectivity.
- [ ] Check if the system is macOS or Linux (supported platforms).
