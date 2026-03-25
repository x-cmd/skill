---
name: assert
description: >
  Test assertion module for validating expected behaviors, outputs, and environment states.
  Core Scenario: When the user is writing tests or scripts that require verification of commands, files, or variables.
license: MIT
---

# assert - Robust Test Assertions

The `assert` module is a powerful tool for verifying that test results meet expectations. It is widely used in automated testing and CI/CD pipelines to ensure code correctness.

## When to Activate
- When writing shell scripts that need to verify command success or failure.
- When performing regression tests on file systems (checks for non-empty, existence, etc.).
- When validating variable states or detecting global variable leaks.
- When checking stdout/stderr against expected patterns or content.

## Core Principles & Rules
- **Exit Code Focus**: Commands return 0 on success and non-zero on failure to break script execution if combined with `set -e`.
- **Negation Support**: Use `!` for negative command checks or `^` for negative type/file checks.
- **Bulk Verification**: Many subcommands support checking multiple values or files in one go.

## Additional Scenarios
- **Variable Snapshots**: Use `var save` and `var cmp` to detect unintended global variable modifications in functions.
- **Output Matching**: Use `stdout` with heredocs to verify complex multi-line command outputs.

## Patterns & Examples

### Command and Type Verification
```bash
# Verify command success and multiple integer values
x assert true [ 1 -eq 1 ]
x assert is-int 10 20 30
```

### File System Checks
```bash
# Ensure specific files exist and are not empty
x assert is-file /etc/hosts
x assert is-nonempty ./log.txt
```

### Detecting Variable Leaks
```bash
# Capture state, run function, and check for leaks
x assert var save
my_function
x assert var cmp
```

## Checklist
- [ ] Ensure the correct operator (`is-file`, `is-int`, etc.) is used.
- [ ] Confirm if negation (`!` or `^`) is required for the test logic.
- [ ] Verify that multi-line expectations match the stdin/heredoc format.
