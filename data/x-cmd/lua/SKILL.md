---
name: lua
description: >
  Enhanced Lua development module, supporting project initialization, module installation via Luarocks, and static compilation.
  Core Scenario: When the user needs to set up Lua projects, manage libraries, or compile Lua scripts to static binaries.
license: MIT
---

# lua - Enhanced Lua Development

The `lua` module provides a comprehensive suite of tools for Lua developers, simplifying project setup, dependency management, and script execution.

## When to Activate
- When initializing a new Lua project structure (`init`).
- When installing Lua modules and libraries using Luarocks integration (`install`).
- When formatting, checking, or linting Lua source code.
- When compiling Lua scripts into standalone static binaries.

## Core Principles & Rules
- **Luarocks Integration**: Use the `install` (or `i`) subcommand to manage dependencies easily.
- **Static Compilation**: Leverage the `static` subcommand for creating zero-dependency Lua executables.
- **Code Quality**: Supports `check` and `format` for maintaining Lua code standards.

## Patterns & Examples

### Install Library
```bash
# Install the 'lua-cjson' module using the integrated Luarocks
x lua i lua-cjson
```

### Static Build
```bash
# Compile a Lua script into a single static binary
x lua static main.lua
```

### Initialize Project
```bash
# Set up a new Lua project environment
x lua init
```

## Checklist
- [ ] Confirm if the user needs to manage dependencies or execute a script.
- [ ] Verify the target platform for static compilation.
