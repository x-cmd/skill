---
name: luarocks
description: >
  Enhanced interface for Luarocks, the package manager for Lua modules.
  Core Scenario: When the user needs to install, manage, or search for Lua dependencies.
license: MIT
---

# luarocks - Lua Package Management

The `luarocks` module provides a streamlined CLI for managing Lua modules, integrating with x-cmd's package system for easy installation.

## When to Activate
- When installing or removing Lua packages (Rocks).
- When searching the Luarocks repository for modules.

## Patterns & Examples

### Install Module
```bash
# Install a specific Lua module
x luarocks install luasocket
```

## Checklist
- [ ] Confirm the module name.
- [ ] Verify if the Lua runtime is correctly set up.
