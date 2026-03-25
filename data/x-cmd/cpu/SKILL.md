---
name: cpu
description: >
  Retrieve CPU hardware information and detect system endianness.
  Core Scenario: When the user needs to check processor details or identify byte order (endianness) for compilation or diagnostics.
license: MIT
---

# cpu - CPU Information & Detection

The `cpu` module provides a quick way to retrieve detailed hardware information about the system's processor and determine the system's byte order (endianness).

## When to Activate
- When identifying CPU architecture or hardware specs.
- When detecting system endianness (little-endian vs big-endian) for low-level development or network programming.

## Core Principles & Rules
- **Conciseness**: Designed for rapid hardware checks.
- **Endianness Flags**: Returns `l` for little-endian and `b` for big-endian.

## Patterns & Examples

### View CPU Info
```bash
# Display detailed processor hardware information
x cpu info
```

### Detect Endianness
```bash
# Check system byte order
x cpu endianness
```

## Checklist
- [ ] Confirm if general info or just endianness is needed.
