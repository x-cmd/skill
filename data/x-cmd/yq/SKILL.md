---
name: yq
description: >
  Portable command-line YAML, JSON, and XML processor with an interactive REPL for exploring data.
  Core Scenario: When the user needs to query, extract, or interactively browse YAML configurations or structured data.
license: MIT
---

# yq - YAML, JSON & XML Processor

The `yq` module is a versatile tool for handling structured data formats, primarily YAML. The x-cmd version adds an interactive REPL powered by FZF to make browsing complex configuration files intuitive.

## When to Activate
- When the user needs to query or modify YAML configuration files.
- When converting between YAML, JSON, and other formats.
- When interactively exploring nested YAML structures with live feedback.
- When extracting specific values from deeply nested keys.

## Core Principles & Rules
- **Interactive Exploration**: Use the `repl` (or `r`) subcommand for a searchable TUI to browse YAML data.
- **Multiformat**: Supports YAML, JSON, XML, and properties files as output or input formats.
- **In-place Edits**: Support for modifying files directly via the `-i` flag.

## Patterns & Examples

### Interactively Explore Config
```bash
# Browse a complex configuration file with FZF
x yq r config.yml
```

### Extract Value
```bash
# Get a specific nested value from a YAML file
x yq '.database.host' config.yml
```

### Convert Formats
```bash
# Output a YAML file as JSON
x yq -o json config.yml
```

## Checklist
- [ ] Confirm if the target file is YAML, JSON, or XML.
- [ ] Verify if an interactive view or a specific extraction is needed.
- [ ] Check if the file should be modified in-place (`-i`).
