---
name: pwsh
description: >
  PowerShell enhancement module providing interactive REPL and system management tools.
  Core Scenario: When the user needs to manage system resources (disk, IP, processes) or use PowerShell via x-cmd.
license: MIT
---

# pwsh - PowerShell CLI Enhancement

The `pwsh` module improves the PowerShell experience on supported platforms, offering an interactive REPL and powerful system management subcommands for disks, IPs, and processes.

## When to Activate
- When the user needs to manage Windows/PowerShell-specific system resources.
- When entering a PowerShell interactive REPL via x-cmd.
- When performing rapid system diagnostics (processes, services, logs) using PowerShell.

## Patterns & Examples

### Interactive REPL
```bash
# Enter the interactive PowerShell REPL
x pwsh --repl
```

### System Management
```bash
# List system processes using PowerShell backend
x pwsh ps
```

## Checklist
- [ ] Confirm if the environment supports PowerShell (primarily for Git Bash/Windows).
- [ ] Verify the specific system resource being managed.
