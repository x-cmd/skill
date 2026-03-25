---
name: is
description: >
  Verify values and environment states against expected conditions, supporting bulk checks and type detection.
  Core Scenario: When the user needs to validate inputs (integers, IPs) or detect environment suitability (TTY, WSL) in scripts.
license: MIT
---

# is - Value & Environment Validation

The `is` module is a powerful validation tool used to check if values or environment states meet specific criteria. It supports batch checking multiple values and is ideal for robust shell scripting.

## When to Activate
- When validating if inputs are of a specific type (e.g., integer, float, IP).
- When detecting terminal states (interactive, TTY, shell type).
- When performing environment suitability checks for features like `advise`.
- When comparing file ages or checking if variables are unset.

## Core Principles & Rules
- **Bulk Validation**: Support batching multiple values; the command succeeds only if all values pass the check.
- **Exit Code Logic**: Returns 0 on success and 1 on failure, making it perfect for `&&` and `||` chains.
- **Context Sensitivity**: Use environmental checks (e.g., `interactive_tty`) to determine script behavior.

## Additional Scenarios
- **File Comparison**: Check if a specific file is the newest or oldest in a set.
- **Network Validation**: Batch verify multiple IPv4 addresses.

## Patterns & Examples

### Type Check
```bash
# Batch verify if multiple values are integers
x is int 42 100 -5
```

### Environment Detection
```bash
# Check if running in an interactive TTY
x is interactive_tty && echo "In Terminal" || echo "Piped"
```

### Variable Check
```bash
# Ensure multiple temporary variables are not set
x is unset TEMP_VAR1 TEMP_VAR2
```

## Checklist
- [ ] Confirm if multiple values need to be validated simultaneously.
- [ ] Ensure the correct environment check is used (e.g., `interactive` vs `interactiveshell`).
