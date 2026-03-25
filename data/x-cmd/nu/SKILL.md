---
name: nu
description: >
  Enhanced interface for Nushell, enabling x-cmd integration, documentation search, and AI-powered command generation.
  Core Scenario: When the user needs to set up x-cmd in Nushell or search Nushell documentation via CLI.
license: MIT
---

# nu - Nushell Enhancement & Integration

The `nu` module improves the Nushell experience by providing easy integration with the x-cmd ecosystem and providing quick access to Nushell's official documentation and AI-generated commands.

## When to Activate
- When the user wants to inject x-cmd tools (x, c, @gpt) into their Nushell environment.
- When searching for specific Nushell syntax or alias information on nushell.sh.
- When launching Nushell with automatic installation via x-cmd pkg.

## Core Principles & Rules
- **Integration**: Use `--setup` to automatically modify `env.nu` for x-cmd support.
- **On-demand Setup**: Automatically handles Nushell installation if missing from the system.

## Patterns & Examples

### Setup x-cmd
```bash
# Inject x-cmd utilities into the Nushell configuration
x nu --setup
```

### Search Documentation
```bash
# Search nushell.sh for 'alias' information interactively
x nu : alias
```

## Checklist
- [ ] Confirm if the user is using Nushell as their primary shell.
- [ ] Verify if x-cmd should be injected permanently or just tested.
