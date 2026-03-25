---
name: uname
description: >
  Enhanced system information display with structured key-value output.
  Core Scenario: When the user needs a quick, readable summary of kernel version, architecture, and OS name.
license: MIT
---

# uname - Structured System Information

The `uname` module provides a colorful, structured summary of common system information, including hostname, OS name, kernel details, and architecture.

## When to Activate
- When the user wants to identify their system environment details.
- When verifying kernel versions or system architecture for software compatibility.

## Core Principles & Rules
- **Clarity**: Consolidates key system data into an easy-to-read format.
- **TTY Awareness**: Automatically disables colors when the output is piped.

## Patterns & Examples

### System Summary
```bash
# Display hostname, system name, kernel, and architecture
x uname
```

## Checklist
- [ ] Confirm if the user needs specific uname flags or just a general summary.
