---
name: python
description: >
  Enhanced Python environment manager, providing version management, pip integration, and automated setup.
  Core Scenario: When the user needs to install specific Python versions, manage packages, or run scripts.
license: MIT
---

# python - Python Environment Management

The `python` module provides a unified CLI for managing Python versions and environments, integrating with x-cmd's package system for seamless setup across diverse OS platforms.

## When to Activate
- When installing or switching between Python versions.
- When managing Python packages using the integrated `pip` functionality.
- When running Python scripts or entering interactive REPLs.

## Core Principles & Rules
- **Integration**: Works with asdf and other version managers to ensure availability.
- **Submodule Support**: Use `pip` for managing dependencies within the current Python environment.

## Patterns & Examples

### Install Packages
```bash
# Install a package globally within the managed Python environment
x python pip install requests
```

### Run Script
```bash
# Execute a Python script with the managed runtime
x python ./script.py
```

## Checklist
- [ ] Confirm the target Python version if multiple are available.
- [ ] Verify the package name for pip installation.
