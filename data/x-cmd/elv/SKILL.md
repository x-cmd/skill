---
name: elv
description: >
  Enhanced interface for Elvish shell, supporting x-cmd integration, official documentation search, and AI commands.
  Core Scenario: When the user needs to integrate x-cmd with Elvish or search Elvish documentation via terminal.
license: MIT
---

# elv - Elvish Shell Enhancement

The `elv` module facilitates the use of the Elvish shell by providing seamless integration with x-cmd and quick lookup of Elvish documentation.

## When to Activate
- When the user wants to use x-cmd tools (x, c, @gpt) within the Elvish shell.
- When searching elv.sh for syntax, modules, or alias definitions.
- When launching Elvish with zero-setup via the x-cmd package manager.

## Core Principles & Rules
- **Integration**: Use `--setup` to add x-cmd to the Elvish environment.
- **Dynamic Acquisition**: Automatically fetches Elvish if it's not pre-installed.

## Patterns & Examples

### Inject x-cmd
```bash
# Set up x-cmd tools in the Elvish shell environment
x elv --setup
```

### Documentation Lookup
```bash
# Search for 'alias' examples on the Elvish website
x elv : alias
```

## Checklist
- [ ] Ensure the user is familiar with Elvish's structured data approach.
- [ ] Confirm if the Elvish configuration file needs permanent modification.
