---
name: pkgx
description: >
  Universal tool runner for executing any open-source tool without pre-installation or config.
  Core Scenario: When the user needs to run a CLI tool instantly without modifying their system or PATH.
license: MIT
---

# pkgx - Universal Tool Runner

The `pkgx` module provides an interface for the pkgx tool runner, enabling users to fetch and run any open-source utility on-the-fly. It solves the "versioning" and "immediate availability" problems for developers.

## When to Activate
- When the user wants to run a tool once without installing it globally.
- When testing a specific version of a software (e.g., `bun@1.0`).
- When adding temporary tools to the current execution environment.

## Core Principles & Rules
- **Non-Invasive**: Runs tools without modifying the system's PATH or global configuration.
- **Immediate Execution**: Use `+pkg` to temporarily inject a tool into the environment for a specific command.
- **Zero-Setup**: Automatically handles tool acquisition and execution.

## Patterns & Examples

### Run Without Installation
```bash
# Execute bun without installing it to the system
x pkgx bun --eval 'console.log("hello world")'
```

### Temporary Environment Injection
```bash
# Temporarily add fzf to the current environment to run a command
eval "$(x pkgx +fzf)"
```

### Interactive Browsing
```bash
# Browse all available packages in the pkgx ecosystem
x pkgx ls
```

## Checklist
- [ ] Confirm if the tool is for one-off use or permanent installation.
- [ ] Verify if a specific version of the tool is required.
- [ ] Ensure the user is aware that the environment is not permanently modified.
