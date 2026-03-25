---
name: pixi
description: >
  Cross-platform package manager and workflow tool based on Conda, with mirror acceleration.
  Core Scenario: When the user needs fast, isolated environments for Python, C++, or other multi-language projects.
license: MIT
---

# pixi - Fast Package & Environment Manager

The `pixi` module provides an enhanced interface for Pixi, offering mirror-accelerated downloads and simplified global package management using x-cmd conventions.

## When to Activate
- When the user wants to install tools globally in their user environment (`use`).
- When managing conda-based projects with isolated dependencies.
- When creating a new project or adding dependencies to an existing pixi project.
- When running commands in a temporary, isolated environment (`exec`).

## Core Principles & Rules
- **Global Management**: Prefer `x pixi use <pkg>` for user-wide tool installations.
- **Mirror Acceleration**: Automatically uses optimized mirrors for faster downloads in specific regions.
- **Environment Isolation**: Supports `shell` and `exec` for running code within a project's locked environment.

## Patterns & Examples

### Install Global Tools
```bash
# Install 'fish' shell and 'jq' globally via pixi
x pixi use fish jq
```

### Isolated Command Execution
```bash
# Run a command within a temporary pixi environment
x pixi exec python --version
```

### Interactive Browsing
```bash
# Interactively search for available pixi packages
x pixi la
```

## Checklist
- [ ] Confirm if the package should be installed globally or locally to a project.
- [ ] Verify if an isolated environment (`exec`) is preferred.
- [ ] Ensure the user is aware of the mirror acceleration benefits.
