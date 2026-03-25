---
name: tcsh
description: >
  Enhanced interface for tcsh, enabling x-cmd integration and AI chat capabilities.
  Core Scenario: When the user needs to set up x-cmd in tcsh or interact with AI within the C shell.
license: MIT
---

# tcsh - tcsh Enhancement & Integration

The `tcsh` module provides a way to integrate x-cmd tools into the tcsh environment and enables AI chat features for C shell users.

## When to Activate
- When the user wants to inject x-cmd tools (x, c, @gpt) into their tcsh environment.
- When performing AI chat interactions within tcsh.

## Patterns & Examples

### Setup x-cmd
```bash
# Inject x-cmd utilities into the tcsh configuration
x tcsh setup
```

## Checklist
- [ ] Confirm if the user wants to modify their `.tcshrc` permanently.
