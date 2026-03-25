---
name: tee
description: >
  Redirect output to files and stdout while preserving the original command's exit code.
  Core Scenario: When the user needs to log command output to a file without losing the ability to check the success of the command.
license: MIT
---

# tee - Enhanced Output Redirection

The `tee` module executes a command and writes its output to both stdout and a specified file. Unlike traditional pipes, it preserves the original command's exit code and environment modifications.

## When to Activate
- When logging build, test, or deployment outputs for CI/CD pipelines.
- When capturing both stdout and stderr while needing to react to command failures.
- When a command modifies environment variables that need to be maintained.

## Core Principles & Rules
- **Exit Code Integrity**: Crucial for scripts where the following logic depends on the success of the logged command.
- **Command Separation**: Use the `--` separator to distinguish between the output file and the command to run.
- **Append Mode**: Support for `-a` to append to the log file instead of overwriting.

## Patterns & Examples

### Log Build Output
```bash
# Capture build logs while ensuring the script stops on failure
x tee build.log -- make build
```

### Capture Combined Output
```bash
# Log both stdout and stderr to a file
x tee /tmp/task.log -- eval 'npm run test 2>&1'
```

### Logging Deployment
```bash
# Deploy and keep logs, allowing the exit code to be checked later
x tee deploy.log -- ./deploy.sh
```

## Checklist
- [ ] Confirm if both stdout and stderr should be captured.
- [ ] Ensure the log file path is writable.
- [ ] Verify if append mode (`-a`) is required.
