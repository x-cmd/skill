---
name: gitconfig
description: >
  Manage Git configurations using YAML files for structured and portable setup.
  Core Scenario: When the user needs to apply complex Git configurations or migrate settings via YAML.
license: MIT
---

# gitconfig - YAML-Based Git Configuration

The `gitconfig` module allows users to manage their Git settings through structured YAML files, providing a more readable and portable way to configure Git aliases, user info, and behaviors.

## When to Activate
- When applying a set of Git configurations from a YAML template.
- When managing multiple Git profiles or complex alias sets.

## Core Principles & Rules
- **Portability**: Focuses on using YAML as the source of truth for Git settings.
- **Batch Application**: Use the `apply` subcommand to set multiple Git options at once.

## Patterns & Examples

### Apply Config
```bash
# Update Git settings based on a YAML configuration file
x gitconfig apply my-config.yml
```

## Checklist
- [ ] Verify the YAML configuration file format.
- [ ] Confirm the target Git scope (global/local).
