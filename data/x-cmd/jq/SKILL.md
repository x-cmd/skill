---
name: jq
description: >
  Lightweight and flexible JSON processor.
  Core Scenario: When the AI needs to filter, transform, format, or extract JSON data in the terminal. x-cmd version provides zero-dependency auto-installation.
license: MIT
---

# x jq - JSON Processor (AI Optimized)

`x jq` is an enhanced module based on jq, with the core advantage of **zero-dependency auto-installation** and **optimization for scripted tasks**. It ensures JSON can be processed immediately in any environment.

## When to Activate
- When specific fields need to be extracted from complex JSON responses.
- When JSON structures need to be filtered, restructured, or transformed.
- When unformatted JSON strings need to be beautified for further analysis.
- When processing results need to be output as raw strings (`-r`) for other commands.

## Core Principles & Rules
- **Non-interactive First**: AI should avoid the interactive `repl` mode and use jq expressions directly.
- **Pipe Integration**: Recommended for use with pipes, e.g., `cat data.json | x jq '.field'`.
- **Format Control**:
  - Use `-r` (raw-output) for raw values without quotes (ideal for getting single string values).
  - Use `-c` (compact-output) for compact one-line JSON to save context Tokens.
- **Environment Isolation**: `x jq` automatically downloads and runs jq when necessary, without polluting the system environment.

## Patterns & Examples

### Extract and Output Raw String
```bash
# Get the value of the 'version' field (no quotes, suitable for script use)
x jq -r '.version' package.json
```

### Filter Array and Compact Output
```bash
# Filter and output as a compact one-line JSON to save Tokens
x jq -c '.items[] | select(.status == "active")' data.json
```

### Construct New JSON Object
```bash
# Construct a new object containing status and timestamp
x jq -n --arg ts "$(date)" '{"status": "ok", "timestamp": $ts}'
```

### Process Multiple Files
```bash
# Merge and process multiple JSON files
x jq -s '.[0] * .[1]' config1.json config2.json
```

## Checklist
- [ ] Confirm that a non-interactive command is used (no `r` or `repl`).
- [ ] Consider if `-r` is needed for plain text values.
- [ ] Consider if `-c` is needed to reduce output volume.
