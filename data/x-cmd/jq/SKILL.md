---
name: jq
description: >
  Lightweight and flexible JSON processor with an interactive REPL for data exploration.
  Core Scenario: When the user needs to filter, transform, or interactively explore JSON data in the terminal.
license: MIT
---

# jq - JSON Processor & Interactive REPL

The `jq` module provides a powerful CLI interface for processing JSON data. The x-cmd version enhances the experience by providing an interactive REPL powered by FZF for exploring complex data structures.

## When to Activate
- When the user needs to extract specific fields or transform JSON structures.
- When interactively exploring large JSON files or API responses.
- When filtering arrays or selecting objects based on specific criteria.
- When pretty-printing JSON for better readability.

## Core Principles & Rules
- **Interactive Exploration**: Use the `repl` (or `r`) subcommand to explore data with live feedback.
- **Piping Support**: Highly optimized for consuming data from other commands (e.g., `curl ... | x jq`).
- **Zero-Dependency**: Automatically handles `jq` installation if it is missing from the environment.

## Patterns & Examples

### Extract Field
```bash
# Get the 'name' field from a JSON file
x jq '.name' data.json
```

### Interactive REPL
```bash
# Explore an API response interactively
curl https://api.example.com/data | x jq r
```

### Filter Array
```bash
# Select all users older than 18
x jq '.[] | select(.age > 18)' users.json
```

## Checklist
- [ ] Confirm the JSON structure or path to the target field.
- [ ] Verify if an interactive exploration (`repl`) is more suitable for the task.
