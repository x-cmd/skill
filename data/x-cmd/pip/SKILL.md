---
name: pip
description: >
  Enhanced interface for pip, the Python package installer.
  Core Scenario: When the user needs to manage Python dependencies or search for packages on PyPI.
license: MIT
---

# pip - Python Package Management

The `pip` module provides a convenient CLI for managing Python packages, ensuring that dependencies are handled correctly within the managed Python environment.

## When to Activate
- When installing, upgrading, or removing Python packages.
- When listing currently installed Python dependencies.

## Patterns & Examples

### Install Package
```bash
# Install a specific Python package
x pip install pandas
```

## Checklist
- [ ] Confirm the package name.
- [ ] Verify the target Python environment.
