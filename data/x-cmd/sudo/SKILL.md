---
name: sudo
description: >
  Execute commands with elevated privileges (sudo/doas/su) while preserving PATH and x-cmd environment.
  Core Scenario: When the user needs to run system commands or x-cmd tools with root or specific user privileges.
license: MIT
---

# sudo - Enhanced Privilege Elevation

The `sudo` module provides a smarter way to escalate privileges, automatically choosing between `sudo`, `doas`, or `su`. It ensures that custom PATH settings and the x-cmd environment are preserved in the elevated session.

## When to Activate
- When the user needs to perform system-level tasks (e.g., editing `/etc/hosts`).
- When running x-cmd modules that require root access while keeping the x-cmd environment intact.
- When an automatic fallback between different privilege elevation tools is required.

## Core Principles & Rules
- **PATH Preservation**: Always use `x sudo` instead of raw `sudo` to ensure x-cmd commands and custom binaries remain available.
- **Auto-Fallback**: The tool automatically tries `sudo` → `doas` → `su` to find the best available method.
- **Environment Consistency**: Variables like `___X_CMD_ROOT` are retained for seamless operation.

## Patterns & Examples

### Run System Command
```bash
# Update package index with root privileges
x sudo apt update
```

### Edit System Files
```bash
# Open a system file with root privileges
x sudo vim /etc/hosts
```

### Run as Specific User
```bash
# Execute a command as 'admin' using su
x sudo --suuser admin whoami
```

## Checklist
- [ ] Confirm if the command requires elevated privileges.
- [ ] Verify if the user has the necessary permissions to use sudo/su.
- [ ] Ensure specific environment variables are needed or should be preserved.
