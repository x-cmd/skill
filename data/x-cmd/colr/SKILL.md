---
name: colr
description: >
  Color lookup tool for finding ANSI and RGB codes, supporting rainbow output.
  Core Scenario: When the user needs to find color codes or generate colorful terminal text for scripts.
license: MIT
---

# colr - Terminal Color Lookup

The `colr` module helps users find ANSI color codes (8-color, 256-color) and RGB equivalents. It also offers a rainbow generator for colorful text output.

## When to Activate
- When looking up the numeric code for a specific color (e.g., green).
- When finding approximate colors for specific RGB or Hex values.
- When generating "rainbow" styled output for messages.

## Patterns & Examples

### Find Color Code
```bash
# List all 256-color codes and their visual approximations
x colr
```

### Search by Hex
```bash
# Find the nearest terminal color for a hex value
x colr #008000
```

### Rainbow Output
```bash
# Output a message in rainbow colors
x cowsay "Hello" | x colr rainbow
```

## Checklist
- [ ] Confirm if the user needs ANSI 8, 256, or RGB values.
- [ ] Verify if a rainbow transformation is requested for output.
