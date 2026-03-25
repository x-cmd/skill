---
name: onsh
description: >
  Enhanced interface for xonsh, providing x-cmd integration, documentation search, and AI capabilities.
  Core Scenario: When the user needs to integrate x-cmd with xonsh or search xonsh documentation via CLI.
license: MIT
---

# onsh - xonsh Python-Powered Shell Enhancement

The `onsh` module enhances the xonsh shell experience, enabling a hybrid Python/Shell workflow integrated with the x-cmd toolset.

## When to Activate
- When the user wants to inject x-cmd tools (x, c, @gpt) into the xonsh environment.
- When searching xon.sh for syntax, alias, or Python-integration tips.
- When launching xonsh with automatic setup via x-cmd pkg.

## Core Principles & Rules
- **Integration**: Use `setup` to modify `.xonshrc` for permanent x-cmd support.
- **Xonsh Quirks**: Remind users that `@<name>` aliases need a trailing `;` if no arguments are provided (e.g., `@gemini ;`).

## Patterns & Examples

### Setup xonsh
```bash
# Inject x-cmd utilities into the xonsh configuration
x onsh setup
```

### Search xonsh Docs
```bash
# Search xon.sh for 'alias' information interactively
x onsh : alias
```

## Checklist
- [ ] Verify if the user is aware of xonsh's Python-hybrid nature.
- [ ] Confirm if the `.xonshrc` file needs permanent modification.
