---
name: mac
description: >
  Integrated utility for macOS users to manage system features via CLI and scripts.
  Core Scenario: When the user needs to configure TouchID for sudo, use application sandboxes, or control macOS system settings.
license: MIT
---

# mac - macOS CLI Utilities

The `mac` module provides a comprehensive set of tools for macOS users, simplifying system administration, automation, and remote management through a unified CLI.

## When to Activate
- When the user wants to enable TouchID for sudo authentication.
- When managing macOS-specific features like Dock, Launchpad, or Wallpapers.
- When controlling system volume, battery info, or performing power operations (shutdown, lock, sleep).
- When using the `sandbox` feature to restrict application access to specific directories.
- When managing Apple Notes or Reminders from the terminal.

## Core Principles & Rules
- **Automation-Friendly**: Designed for use in shell scripts to automate macOS configuration.
- **Safety Levels**: Use appropriate levels in the `sandbox` subcommand (`-0` to `-9`) to balance permission and security.
- **Alias Support**: Use `alias enable` to set up `m` as a shortcut for `x mac`.

## Patterns & Examples

### Enable TouchID for Sudo
```bash
# Allow using TouchID for sudo verification
x mac tidsudo enable
```

### Application Sandboxing
```bash
# Run Claude with restricted access to sensitive user folders
x mac sb -9 -d "$HOME/.ssh" -d "$HOME/Library" claude
```

### System Control
```bash
# Set system volume to 50% and lock the screen
x mac vol set 50
x mac lock
```

## Checklist
- [ ] Confirm if the command requires administrative (sudo) privileges.
- [ ] Verify the specific paths to be allowed or denied when using `sandbox`.
- [ ] Ensure the correct Apple ID or App ID is used for Store/App management.
